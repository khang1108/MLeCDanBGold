FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY thundercompute ./thundercompute
COPY configs ./configs
COPY offline ./offline

# The shared backend calls hosted inference. Keeping local Torch out of this
# image avoids downloading GPU packages that serving does not use.
RUN pip install --no-deps . \
    && pip install \
        "fastapi>=0.139.2" \
        "uvicorn>=0.51.0" \
        "httpx>=0.28.1" \
        "pydantic>=2.0,<3.0" \
        "pydantic-settings>=2.14.2" \
        "python-dotenv>=1.0" \
        "pandas>=2.0,<4.0" \
        "pillow>=10.0,<13.0" \
        "pyarrow>=14.0,<26.0" \
        "pyyaml>=6.0,<7.0" \
        "numpy>=1.24" \
        "faiss-cpu>=1.7.3" \
        "python-multipart>=0.0.20" \
        "tqdm>=4.66,<5.0"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "hcmai.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
