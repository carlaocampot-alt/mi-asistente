"""Pruebas sin modelo real ni internet: uv run pytest

Un modelo falso (FunctionModel) pide la tool que queremos, y comprobamos lo que hace el código.
"""

import os
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY_LOCAL", "clave-de-prueba")  # para importar sin .env (no se llama a Gemini)

import httpx  # noqa: E402
import pytest  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

import api  # noqa: E402
import asistente  # noqa: E402
import bot  # noqa: E402
import capacidades  # noqa: E402
from config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def base_temporal(tmp_path, monkeypatch):
    """Cada prueba usa su propia reclutamiento.db, no la de datos/."""
    monkeypatch.setattr(capacidades, "DB", tmp_path / "reclutamiento.db")


def pide(tool: str, args: dict) -> FunctionModel:
    """Un modelo falso: la 1.ª vez pide la tool; cuando recibe su resultado, responde texto."""

    def responder(mensajes, info):
        ultimo = mensajes[-1].parts[-1]
        if isinstance(ultimo, ToolReturnPart):
            return ModelResponse(parts=[TextPart(f"Resultado: {ultimo.content}")])
        return ModelResponse(parts=[ToolCallPart(tool, args)])

    return FunctionModel(responder)


def test_kpis_q3_sin_modelo():
    k = capacidades.kpis_periodo("2026-07-01", "2026-09-30")  # una tool es una función normal
    assert k["vacantes_cubiertas"] == 19
    assert k["offer_acceptance_pct"] == 90.5
    assert k["cumplimiento_metas"]["time_to_fill_dias"].startswith("no cumple")


def test_kpis_fechas_invalidas():
    assert "error" in capacidades.kpis_periodo("2026-09-30", "2026-07-01")
    assert "error" in capacidades.kpis_periodo("julio", "2026-09-30")


def test_ranking_tiene_los_cuatro_recruiters():
    ranking = capacidades.ranking_recruiters("2026-07-01", "2026-09-30")
    assert len(ranking) == 4
    assert sum(r["contrataciones"] for r in ranking) == 19


async def test_registrar_contratacion_cierra_la_vacante():
    abierta = capacidades.vacantes_abiertas()[-1]  # la más reciente tiene candidatos en proceso
    candidato = next(
        c for c in capacidades.candidatos_de_vacante(abierta["id"])["candidatos"] if c["estado"] == "en_proceso"
    )
    with asistente.asistente.override(model=pide("registrar_contratacion", {
        "candidato_id": candidato["id"], "fecha": "2026-11-05",
    })):
        respuesta = await asistente.responder("Registra la contratación.")
    assert "Contratación registrada" in respuesta
    assert abierta["id"] not in [v["id"] for v in capacidades.vacantes_abiertas()]


async def test_no_registra_candidato_en_vacante_cerrada():
    with asistente.asistente.override(model=pide("registrar_candidato", {
        "vacante_id": 1, "nombre": "Ana Prueba", "fuente": "LinkedIn", "salario_usd": 2500, "fecha": "2026-11-05",
    })):
        respuesta = await asistente.responder("Agrega a Ana a la vacante 1.")
    assert "No se guardó" in respuesta


def test_convertir_moneda_sin_internet(monkeypatch):
    def get_falso(url, timeout):
        return httpx.Response(200, json={"result": "success", "rates": {"BOB": 6.96}}, request=httpx.Request("GET", url))

    monkeypatch.setattr(capacidades.httpx, "get", get_falso)
    r = capacidades.convertir_moneda(3000, "usd", "bob")
    assert r["resultado"] == 20880.0


async def test_chat_ajeno_se_rechaza_sin_llamar_al_modelo(monkeypatch):
    monkeypatch.setattr(settings, "telegram_permitidos", "123")
    respuestas = []

    async def reply_text(texto):
        respuestas.append(texto)

    def modelo_prohibido(mensajes, info):
        raise AssertionError("no debía llamarse al modelo")

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=999), message=SimpleNamespace(text="hola", reply_text=reply_text)
    )
    with asistente.asistente.override(model=FunctionModel(modelo_prohibido)):
        await bot.mensaje(update, None)
    assert "No autorizado" in respuestas[0] and "999" in respuestas[0]


def test_permitidos(monkeypatch):
    monkeypatch.setattr(settings, "telegram_permitidos", " 123, 456 ,")
    assert settings.permitidos == {"123", "456"}
    monkeypatch.setattr(settings, "telegram_permitidos", "")
    assert settings.permitidos == set()  # vacío = nadie


# ── Extra opcional: la API (API.md) ──


async def post_chat(cuerpo: dict, token: str | None = None) -> httpx.Response:
    """Llama a la API en el mismo proceso, sin servidor ni red."""
    cabeceras = {"Authorization": f"Bearer {token}"} if token else {}
    transporte = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as cliente:
        return await cliente.post("/chat", json=cuerpo, headers=cabeceras)


async def test_api_sin_token_valido_no_llama_al_modelo(monkeypatch):
    def modelo_prohibido(mensajes, info):
        raise AssertionError("no debía llamarse al modelo")

    with asistente.asistente.override(model=FunctionModel(modelo_prohibido)):
        monkeypatch.setattr(settings, "api_tokens", SecretStr(""))
        assert (await post_chat({"mensaje": "hola"}, "cualquiera")).status_code == 401  # vacío = nadie
        monkeypatch.setattr(settings, "api_tokens", SecretStr("abc, def"))
        assert (await post_chat({"mensaje": "hola"})).status_code == 401
        assert (await post_chat({"mensaje": "hola"}, "xyz")).status_code == 401
        assert (await post_chat({"mensaje": ""}, "abc")).status_code == 422  # Pydantic valida el pedido


async def test_api_con_token_responde(monkeypatch):
    monkeypatch.setattr(settings, "api_tokens", SecretStr("abc,def"))
    with asistente.asistente.override(model=pide("kpis_periodo", {"desde": "2026-07-01", "hasta": "2026-09-30"})):
        r = await post_chat({"mensaje": "¿Cómo nos fue en Q3?"}, "def")
    assert r.status_code == 200
    assert "90.5" in r.json()["texto"]
