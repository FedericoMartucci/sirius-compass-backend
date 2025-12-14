# Sirius Compass – Backend (IRK)

Estructura base para el backend hexagonal de Sirius Compass. Incluye capas para adaptadores MCP, agentes LangGraph, modelos de dominio y puertos, junto con un punto de entrada listo para extender.

## Estructura

- `app/adapters/`: conectores de infraestructura (GitHub, Trello).
- `app/core/`: lógica de negocio pura (agentes, modelos, RAG).
- `app/ports/`: contratos que implementan los adaptadores.
- `app/main.py`: arranque de la aplicación/servidor.
- `tests/`: pruebas unitarias.

## Base de Datos y Migraciones

Este proyecto utiliza **PostgreSQL** como base de datos principal y **Alembic** para manejar las migraciones.

### Configuración Local (Docker)

Para levantar la base de datos localmente:

```bash
docker-compose up -d
```

Esto iniciará un contenedor de PostgreSQL en el puerto `5332` con las credenciales definidas en `.env` (o por defecto en `docker-compose.yml`).

### Comandos de Migración

- **Crear una nueva migración** (después de modificar `models.py`):

  ```bash
  alembic revision --autogenerate -m "descripcion_del_cambio"
  ```

- **Aplicar cambios a la DB**:

  ```bash
  alembic upgrade head
  ```

## Próximos pasos

1. Configurar variables en `.env` (Pinecone, GitHub/Trello tokens, claves LLM).
2. Implementar los modelos Pydantic en `app/core/models/`.
3. Codificar los flujos LangGraph en `app/core/agents/`.
4. Agregar tests en `tests/` para cubrir adaptadores y lógica de negocio.
