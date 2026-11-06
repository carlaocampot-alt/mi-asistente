"""El agente: modelo + instrucciones + capabilities. Sin memoria: cada mensaje empieza de cero.

Pruébalo en la terminal, sin Telegram:  uv run asistente.py
"""

import asyncio

from pydantic_ai import Agent

from capacidades import documentos, kpis, registro, tipo_cambio
from config import modelo

asistente = Agent(
    modelo,
    instructions=(
        "Eres TalentBot, el asistente del equipo de reclutamiento de Andes Talent, una empresa (ficticia) de "
        "staffing tecnológico nearshore en Bolivia. Ayudas a líderes y recruiters a: consultar los KPIs del "
        "proceso (time-to-fill, time-to-hire, aceptación de ofertas, embudo, fuentes) y los resultados del "
        "Q3 2026 frente a sus metas; ver vacantes abiertas y alertas; registrar candidatos y contrataciones; "
        "convertir salarios entre monedas; y explicar el proceso y las políticas citando los documentos. "
        "Si te preguntan algo ajeno al reclutamiento de Andes Talent, di amablemente que solo ayudas con eso. "
        "Responde en el idioma del usuario (por defecto español), breve, en texto plano (sin Markdown)."
    ),
    capabilities=[kpis, registro, tipo_cambio, documentos],  # ← aquí enchufas tus capabilities
)


async def responder(mensaje: str) -> str:
    """Lo único que necesita el bot: un mensaje entra, una respuesta sale."""
    resultado = await asistente.run(mensaje)  # await: mientras espera al modelo, el bot atiende a otros
    return resultado.output


async def main() -> None:
    while (mensaje := input("tú> ").strip()) not in {"", "salir"}:
        print("asistente>", await responder(mensaje))


if __name__ == "__main__":
    asyncio.run(main())  # un solo asyncio.run: todo lo async vive dentro
