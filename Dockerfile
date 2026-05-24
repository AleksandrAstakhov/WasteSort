FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install runtime dependencies only (no train/dev groups)
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main

# Copy package
COPY waste_sort/ waste_sort/
COPY configs/ configs/
COPY artifacts/ artifacts/

EXPOSE 8000

# Default: run CLI inference
ENTRYPOINT ["python", "-m", "waste_sort.cli"]
CMD ["infer"]
