# Sirius Compass – Backend (IRK)

Estructura base para el backend hexagonal de Sirius Compass. Incluye capas para adaptadores MCP, agentes LangGraph, modelos de dominio y puertos, junto con un punto de entrada listo para extender.

## Estructura
- `app/adapters/`: conectores de infraestructura (GitHub, Trello).
- `app/core/`: lógica de negocio pura (agentes, modelos, RAG).
- `app/ports/`: contratos que implementan los adaptadores.
- `app/main.py`: arranque de la aplicación/servidor.
- `tests/`: pruebas unitarias.

## Próximos pasos
1. Configurar variables en `.env` (Pinecone, GitHub/Trello tokens, claves LLM).
2. Implementar los modelos Pydantic en `app/core/models/`.
3. Codificar los flujos LangGraph en `app/core/agents/`.
4. Agregar tests en `tests/` para cubrir adaptadores y lógica de negocio.
