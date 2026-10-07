FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .
RUN adduser --disabled-password --gecos "" --home /app medicore \
    && chown -R medicore:medicore /app

USER medicore

EXPOSE 7860

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers ${WEB_CONCURRENCY:-1} --timeout ${GUNICORN_TIMEOUT:-120} app:app"]
