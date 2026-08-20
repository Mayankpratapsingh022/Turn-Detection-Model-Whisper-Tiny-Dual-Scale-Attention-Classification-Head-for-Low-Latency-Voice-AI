FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[data,train,tracking,hub,eval,export,demo]"

COPY configs ./configs
COPY infra ./infra

ENTRYPOINT ["turn-detector"]
CMD ["--help"]
