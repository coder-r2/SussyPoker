# Beat the Detector API for a Hugging Face Space (Docker SDK). Build from the repository root:
#   docker build -f api/Dockerfile --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t suspoker-api .
#   docker run -p 7860:7860 --cpus 2 --memory 16g suspoker-api        # http://localhost:7860/docs
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app

COPY pyproject.toml README.md LICENSE config.yaml ./
COPY suspoker ./suspoker
# editable install: suspoker.config resolves models/ and config.yaml relative to /app
RUN pip install -e ".[game]"
COPY models ./models
COPY api ./api

ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}
USER user
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
