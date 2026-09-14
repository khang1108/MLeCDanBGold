FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DRES_MOCK_HOST=0.0.0.0 \
    DRES_MOCK_PORT=8090

WORKDIR /opt/dres-mock-server

RUN adduser --disabled-password --gecos "" --uid 10001 mockuser

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

USER 10001:10001
EXPOSE 8090

CMD ["dres-mock-server"]
