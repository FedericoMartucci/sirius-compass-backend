# Use Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (gcc often needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure poetry to not create a virtual environment (install in system python)
# This makes it easier to run commands directly, though poetry run still works
RUN poetry config virtualenvs.create false

# Copy dependency definitions
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the application
COPY . .

# Expose port (default for Uvicorn)
EXPOSE 8000

# Command to run the application
# We add --host 0.0.0.0 to ensure it's accessible outside the container
CMD ["poetry", "run", "uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
