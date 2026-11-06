"""Las capabilities de TalentBot: KPIs de reclutamiento de Andes Talent (empresa ficticia).

Una Capability junta tools + instrucciones en una pieza que se enchufa al agente (asistente.py).
El docstring y los tipos de cada tool son lo único que el modelo lee para decidir cuándo usarla.
"""

import csv
import sqlite3
from datetime import date
from functools import cache
from statistics import mean
from typing import Literal

import chromadb
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic_ai.capabilities import Capability
from pypdf import PdfReader

from config import DATOS, settings

# ═══ Datos: vacantes y candidatos en SQLite (abril a octubre de 2026, ficticios) ════════
CSV_VACANTES = DATOS / "vacantes.csv"
CSV_CANDIDATOS = DATOS / "candidatos.csv"
DB = DATOS / "reclutamiento.db"  # se crea desde los CSV la primera vez

Recruiter = Literal["Lucia Fernandez", "Diego Rojas", "Mariana Paz", "Javier Quispe"]
Fuente = Literal["LinkedIn", "Referido", "Computrabajo", "Base interna", "GetOnBoard"]
ETAPAS = ["aplicado", "screening", "entrevista_tecnica", "entrevista_cliente", "oferta", "contratado"]

# Las metas 2026 (las mismas del PDF definiciones_kpi.pdf)
METAS = {
    "time_to_fill_dias": {"meta": 35, "tipo": "maximo"},
    "time_to_hire_dias": {"meta": 25, "tipo": "maximo"},
    "offer_acceptance_pct": {"meta": 80, "tipo": "minimo"},
    "aplicaciones_por_contratacion": {"meta": 15, "tipo": "maximo"},
}


def crear_base() -> None:
    """Crea reclutamiento.db desde cero con los dos CSV."""
    DB.unlink(missing_ok=True)
    with CSV_VACANTES.open(encoding="utf-8") as a:
        vacantes = list(csv.DictReader(a))
    with CSV_CANDIDATOS.open(encoding="utf-8") as a:
        candidatos = list(csv.DictReader(a))
    conexion = sqlite3.connect(DB)
    with conexion:
        conexion.execute(
            "CREATE TABLE vacantes (id INTEGER PRIMARY KEY, puesto TEXT, cliente TEXT, recruiter TEXT, "
            "prioridad TEXT, fecha_apertura TEXT, fecha_cierre TEXT, estado TEXT)"
        )
        conexion.execute(
            "CREATE TABLE candidatos (id INTEGER PRIMARY KEY, vacante_id INTEGER REFERENCES vacantes(id), "
            "nombre TEXT, fuente TEXT, etapa_maxima TEXT, estado TEXT, fecha_aplicacion TEXT, "
            "fecha_oferta TEXT, fecha_respuesta_oferta TEXT, salario_usd REAL CHECK (salario_usd > 0))"
        )
        conexion.executemany(
            "INSERT INTO vacantes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(v[k] or None for k in v) for v in vacantes],
        )
        conexion.executemany(
            "INSERT INTO candidatos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(c[k] or None for k in c) for c in candidatos],
        )
    conexion.close()


def leer(sql: str, parametros: tuple = ()) -> list[dict]:
    """Ejecuta un SELECT en modo solo lectura y devuelve cada fila como diccionario."""
    if not DB.exists():
        crear_base()
    conexion = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    conexion.row_factory = sqlite3.Row
    filas = [dict(f) for f in conexion.execute(sql, parametros).fetchall()]
    conexion.close()
    return filas


def dias(desde: str, hasta: str) -> int:
    return (date.fromisoformat(hasta) - date.fromisoformat(desde)).days


def fechas_validas(desde: str, hasta: str) -> dict | None:
    """Devuelve un error legible si las fechas no son AAAA-MM-DD o están al revés."""
    try:
        if date.fromisoformat(desde) > date.fromisoformat(hasta):
            return {"error": f"La fecha inicial {desde} es posterior a la final {hasta}"}
    except ValueError:
        return {"error": f"Fechas inválidas: usa AAAA-MM-DD (recibí {desde} y {hasta})"}
    return None


def calcular_kpis(desde: str, hasta: str, recruiter: str | None = None) -> dict:
    """Los KPIs de un periodo, opcionalmente de un solo recruiter."""
    filtro, extra = ("AND v.recruiter = ?", (recruiter,)) if recruiter else ("", ())

    cubiertas = leer(
        "SELECT fecha_apertura, fecha_cierre FROM vacantes v "
        f"WHERE estado = 'cubierta' AND fecha_cierre BETWEEN ? AND ? {filtro}",
        (desde, hasta, *extra),
    )
    abiertas_periodo = leer(
        f"SELECT COUNT(*) AS n FROM vacantes v WHERE fecha_apertura BETWEEN ? AND ? {filtro}",
        (desde, hasta, *extra),
    )[0]["n"]
    aplicaciones = leer(
        "SELECT c.etapa_maxima, c.fuente FROM candidatos c JOIN vacantes v ON v.id = c.vacante_id "
        f"WHERE c.fecha_aplicacion BETWEEN ? AND ? {filtro}",
        (desde, hasta, *extra),
    )
    ofertas = leer(
        "SELECT c.estado, c.fuente, c.fecha_aplicacion, c.fecha_respuesta_oferta, c.salario_usd "
        "FROM candidatos c JOIN vacantes v ON v.id = c.vacante_id "
        f"WHERE c.fecha_respuesta_oferta BETWEEN ? AND ? {filtro}",
        (desde, hasta, *extra),
    )
    contratados = [o for o in ofertas if o["estado"] == "contratado"]

    # Embudo: cuántas de las aplicaciones del periodo llegaron al menos a cada etapa
    embudo = {e: sum(ETAPAS.index(a["etapa_maxima"]) >= i for a in aplicaciones) for i, e in enumerate(ETAPAS)}
    fuentes: dict[str, int] = {}
    for c in contratados:
        fuentes[c["fuente"]] = fuentes.get(c["fuente"], 0) + 1

    ttf = round(mean(dias(v["fecha_apertura"], v["fecha_cierre"]) for v in cubiertas), 1) if cubiertas else None
    tth = round(mean(dias(c["fecha_aplicacion"], c["fecha_respuesta_oferta"]) for c in contratados), 1) if contratados else None
    oar = round(100 * len(contratados) / len(ofertas), 1) if ofertas else None
    apc = round(len(aplicaciones) / len(contratados), 1) if contratados else None

    valores = {
        "time_to_fill_dias": ttf,
        "time_to_hire_dias": tth,
        "offer_acceptance_pct": oar,
        "aplicaciones_por_contratacion": apc,
    }
    cumplimiento = {}
    for kpi, valor in valores.items():
        meta = METAS[kpi]
        if valor is None:
            cumplimiento[kpi] = "sin datos"
        else:
            ok = valor <= meta["meta"] if meta["tipo"] == "maximo" else valor >= meta["meta"]
            cumplimiento[kpi] = f"{'cumple' if ok else 'no cumple'} (meta {meta['tipo']} {meta['meta']})"

    return {
        "periodo": f"{desde} a {hasta}",
        "recruiter": recruiter or "todo el equipo",
        "vacantes_abiertas_en_periodo": abiertas_periodo,
        "vacantes_cubiertas": len(cubiertas),
        "aplicaciones": len(aplicaciones),
        "ofertas_respondidas": len(ofertas),
        "ofertas_aceptadas": len(contratados),
        **valores,
        "cumplimiento_metas": cumplimiento,
        "embudo_de_las_aplicaciones_del_periodo": embudo,
        "contrataciones_por_fuente": fuentes,
        "salario_promedio_contratados_usd": round(mean(c["salario_usd"] for c in contratados)) if contratados else None,
        "fuente": "base de reclutamiento de Andes Talent (SQLite)",
    }


def hoy() -> str:
    """Instrucción dinámica: el modelo no sabe qué día es; se lo decimos en cada mensaje."""
    return f"Hoy es {date.today().isoformat()}."


# ═══ Consulta de KPIs (solo lectura) ══════════════════════════════════════════════════
kpis = Capability(
    id="kpis_reclutamiento",
    description="KPIs del equipo de reclutamiento: time-to-fill, time-to-hire, aceptación de ofertas, embudo y fuentes",
    instructions=[
        "Para métricas usa las tools; nunca inventes cifras. Los trimestres de 2026 son: "
        "Q2 = 2026-04-01 a 2026-06-30, Q3 = 2026-07-01 a 2026-09-30, Q4 = 2026-10-01 a 2026-12-31. "
        "Los datos van de abril a octubre de 2026. Al dar resultados, compáralos con la meta y, si te lo piden, "
        "con el trimestre anterior (llama a la tool para cada periodo).",
        hoy,
    ],
)


@kpis.tool_plain
def kpis_periodo(desde: str, hasta: str, recruiter: Recruiter | None = None) -> dict:
    """KPIs de reclutamiento en un periodo, con el cumplimiento de las metas 2026.

    Args:
        desde: fecha inicial AAAA-MM-DD, por ejemplo '2026-07-01' para el inicio de Q3
        hasta: fecha final AAAA-MM-DD, por ejemplo '2026-09-30' para el fin de Q3
        recruiter: nombre de un recruiter para ver solo sus KPIs; vacío = todo el equipo
    """
    if error := fechas_validas(desde, hasta):
        return error
    return calcular_kpis(desde, hasta, recruiter)


@kpis.tool_plain
def ranking_recruiters(desde: str, hasta: str) -> list[dict] | dict:
    """Compara a los cuatro recruiters en un periodo: contrataciones, time-to-fill y aceptación de ofertas.

    Args:
        desde: fecha inicial AAAA-MM-DD
        hasta: fecha final AAAA-MM-DD
    """
    if error := fechas_validas(desde, hasta):
        return error
    filas = []
    for nombre in Recruiter.__args__:
        k = calcular_kpis(desde, hasta, nombre)
        filas.append({
            "recruiter": nombre,
            "contrataciones": k["ofertas_aceptadas"],
            "time_to_fill_dias": k["time_to_fill_dias"],
            "offer_acceptance_pct": k["offer_acceptance_pct"],
            "aplicaciones": k["aplicaciones"],
        })
    return sorted(filas, key=lambda f: f["contrataciones"], reverse=True)


@kpis.tool_plain
def vacantes_abiertas() -> list[dict]:
    """Vacantes abiertas hoy, con días abiertas, candidatos en proceso y alerta si superan la meta de 35 días."""
    filas = leer(
        "SELECT v.id, v.puesto, v.cliente, v.recruiter, v.prioridad, v.fecha_apertura, "
        "(SELECT COUNT(*) FROM candidatos c WHERE c.vacante_id = v.id AND c.estado = 'en_proceso') AS en_proceso "
        "FROM vacantes v WHERE v.estado = 'abierta' ORDER BY v.fecha_apertura"
    )
    hoy_iso = date.today().isoformat()
    for f in filas:
        f["dias_abierta"] = dias(f["fecha_apertura"], hoy_iso)
        f["alerta"] = "supera la meta de 35 días" if f["dias_abierta"] > METAS["time_to_fill_dias"]["meta"] else ""
    return filas


@kpis.tool_plain
def candidatos_de_vacante(vacante_id: int) -> dict:
    """Los candidatos de una vacante con su etapa, estado y salario pretendido (para ver el pipeline o buscar un id)."""
    vacante = leer("SELECT * FROM vacantes WHERE id = ?", (vacante_id,))
    if not vacante:
        return {"error": f"No existe la vacante {vacante_id}"}
    candidatos = leer(
        "SELECT id, nombre, fuente, etapa_maxima, estado, fecha_aplicacion, salario_usd "
        "FROM candidatos WHERE vacante_id = ? ORDER BY estado = 'descartado', fecha_aplicacion",
        (vacante_id,),
    )
    return {"vacante": vacante[0], "candidatos": candidatos}


# ═══ Registro (escritura): nuevos candidatos y contrataciones ══════════════════════════
registro = Capability(
    id="registro_reclutamiento",
    description="Registra candidatos nuevos y contrataciones en la base SQLite",
    instructions=[
        "Para registrar, confirma que tienes todos los datos; si falta alguno, pídelo. "
        "Después de registrar, informa exactamente lo que devolvió la tool.",
        hoy,
    ],
)


@registro.tool_plain
def registrar_candidato(vacante_id: int, nombre: str, fuente: Fuente, salario_usd: float, fecha: str) -> str:
    """Agrega un candidato nuevo (etapa 'aplicado') a una vacante abierta.

    Args:
        vacante_id: número de la vacante
        nombre: nombre y apellido del candidato
        fuente: de dónde llegó el candidato
        salario_usd: salario mensual pretendido en dólares
        fecha: fecha de aplicación AAAA-MM-DD
    """
    if error := fechas_validas(fecha, fecha):
        return error["error"]
    vacante = leer("SELECT estado FROM vacantes WHERE id = ?", (vacante_id,))
    if not vacante or vacante[0]["estado"] != "abierta":
        return f"No se guardó: la vacante {vacante_id} no existe o no está abierta."
    conexion = sqlite3.connect(DB)
    try:
        with conexion:
            cursor = conexion.execute(
                "INSERT INTO candidatos (vacante_id, nombre, fuente, etapa_maxima, estado, fecha_aplicacion, "
                "salario_usd) VALUES (?, ?, ?, 'aplicado', 'en_proceso', ?, ?)",
                (vacante_id, nombre, fuente, fecha, salario_usd),
            )
    except sqlite3.IntegrityError:
        return f"No se guardó: el salario debe ser mayor que 0 (recibí {salario_usd})."
    finally:
        conexion.close()
    return f"Candidato registrado con id {cursor.lastrowid}: {nombre} ({fuente}) en la vacante {vacante_id}."


@registro.tool_plain
def registrar_contratacion(candidato_id: int, fecha: str) -> str:
    """Marca a un candidato como contratado (aceptó la oferta) y cierra su vacante como cubierta.

    Args:
        candidato_id: id del candidato (búscalo con candidatos_de_vacante si no lo sabes)
        fecha: fecha en que aceptó la oferta, AAAA-MM-DD
    """
    if error := fechas_validas(fecha, fecha):
        return error["error"]
    filas = leer(
        "SELECT c.nombre, c.estado, c.vacante_id, v.estado AS estado_vacante, v.puesto, v.fecha_apertura "
        "FROM candidatos c JOIN vacantes v ON v.id = c.vacante_id WHERE c.id = ?",
        (candidato_id,),
    )
    if not filas:
        return f"No se guardó: no existe el candidato {candidato_id}."
    c = filas[0]
    if c["estado"] != "en_proceso" or c["estado_vacante"] != "abierta":
        return f"No se guardó: {c['nombre']} está '{c['estado']}' y su vacante está '{c['estado_vacante']}'."
    conexion = sqlite3.connect(DB)
    with conexion:  # transacción: todo o nada
        conexion.execute(
            "UPDATE candidatos SET estado = 'contratado', etapa_maxima = 'contratado', "
            "fecha_oferta = COALESCE(fecha_oferta, ?), fecha_respuesta_oferta = ? WHERE id = ?",
            (fecha, fecha, candidato_id),
        )
        conexion.execute(
            "UPDATE candidatos SET estado = 'descartado' WHERE vacante_id = ? AND estado = 'en_proceso'",
            (c["vacante_id"],),
        )
        conexion.execute(
            "UPDATE vacantes SET estado = 'cubierta', fecha_cierre = ? WHERE id = ?", (fecha, c["vacante_id"])
        )
    conexion.close()
    ttf = dias(c["fecha_apertura"], fecha)
    return (f"Contratación registrada: {c['nombre']} para {c['puesto']} (vacante {c['vacante_id']}), "
            f"cubierta en {ttf} días.")


# ═══ API pública: tipo de cambio (ExchangeRate-API, sin clave) ═════════════════════════
tipo_cambio = Capability(
    id="tipo_cambio",
    description="Convierte salarios entre monedas con tipos de cambio actuales (ExchangeRate-API)",
    instructions="Para convertir salarios u ofertas entre monedas usa convertir_moneda y menciona la fuente.",
)


@tipo_cambio.tool_plain
def convertir_moneda(monto: float, de: str, a: str) -> dict:
    """Convierte un monto de una moneda a otra con el tipo de cambio del día.

    Args:
        monto: cantidad a convertir, por ejemplo 3000
        de: código ISO de la moneda de origen, por ejemplo 'USD'
        a: código ISO de la moneda de destino, por ejemplo 'BOB' (boliviano), 'COP', 'MXN', 'ARS', 'PEN'
    """
    de, a = de.upper(), a.upper()
    respuesta = httpx.get(f"https://open.er-api.com/v6/latest/{de}", timeout=10)
    if respuesta.status_code != 200 or respuesta.json().get("result") != "success":
        return {"error": f"No encontré el tipo de cambio para {de}"}
    datos = respuesta.json()
    tasa = datos["rates"].get(a)
    if tasa is None:
        return {"error": f"No hay tipo de cambio de {de} a {a}"}
    return {
        "monto": monto, "de": de, "a": a, "tasa": tasa,
        "resultado": round(monto * tasa, 2),
        "actualizado": datos.get("time_last_update_utc"),
        "fuente": "ExchangeRate-API (open.er-api.com)",
    }


# ═══ RAG: los documentos del proceso de reclutamiento (PDFs) y citas ═══════════════════
DOCUMENTOS = DATOS / "documentos"  # PDFs ficticios de Andes Talent


def embeddings_de(textos: list[str]) -> list[list[float]]:
    """Convierte cada texto en un vector: textos de significado parecido dan vectores cercanos."""
    cliente = OpenAI(
        api_key=settings.gemini_api_key_local.get_secret_value(),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    respuesta = cliente.embeddings.create(model="gemini-embedding-001", input=textos)
    return [item.embedding for item in respuesta.data]


@cache  # se indexa la primera vez que alguien busca, y una sola vez mientras el programa corre
def indice():
    coleccion = chromadb.EphemeralClient().get_or_create_collection(
        "documentos", configuration={"hnsw": {"space": "cosine"}}
    )
    for pdf in sorted(DOCUMENTOS.glob("*.pdf")):
        texto = "\n".join(pagina.extract_text() for pagina in PdfReader(pdf).pages)
        trozos = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_text(texto)
        coleccion.upsert(
            ids=[f"{pdf.stem}-{i}" for i in range(len(trozos))],
            documents=trozos,
            embeddings=embeddings_de(trozos),
        )
    return coleccion


def buscar(pregunta: str, k: int = 4) -> list[dict]:
    """Los k trozos más parecidos a la pregunta, con su fuente y su similitud (1 = idéntico)."""
    r = indice().query(query_embeddings=embeddings_de([pregunta]), n_results=k)
    return [
        {"fuente": fuente, "similitud": round(1 - distancia, 2), "texto": texto}
        for fuente, distancia, texto in zip(r["ids"][0], r["distances"][0], r["documents"][0])
    ]


documentos = Capability(
    id="documentos_proceso",
    description="Manual del proceso de reclutamiento, definiciones y metas de KPIs, política de ofertas y SLA con clientes",
    instructions=(
        "Para preguntas sobre el proceso, etapas, plazos, cómo se calcula un KPI, metas, ofertas, bonos por referido "
        "o garantías a clientes usa buscar_documentos y cita la fuente entre corchetes, por ejemplo "
        "[proceso_reclutamiento-2]. Si no hay información suficiente, di que no la encuentras; no la inventes."
    ),
)


@documentos.tool_plain
def buscar_documentos(pregunta: str) -> list[dict]:
    """Busca en los documentos internos de reclutamiento los fragmentos que responden a la pregunta."""
    resultados = buscar(pregunta)
    for r in resultados:
        print(f"  📄 {r['fuente']} (similitud {r['similitud']})", flush=True)
    return resultados
