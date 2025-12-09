FROM python:3.12-alpine3.22

LABEL maintainer="modeemi.fi"
LABEL description="Python server for door events with Telegram integration and SpaceAPI"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apk add --no-cache --virtual .build-deps \
        build-base \
        libffi-dev \
        openssl-dev \
        python3-dev \
        pkgconfig \
        cargo \
        sqlite-dev \
    && apk add --no-cache \
        libffi \
        openssl \
        sqlite-libs \
        tzdata \
    && pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && apk del .build-deps

COPY . /app

RUN adduser -D -u 1000 -h /app app \
    && chown -R app:app /app \
    && mkdir -p /app/database \
    && chown -R app:app /app/database

USER app

EXPOSE 8000

# Run migrations on startup, then start server
CMD sh -c "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"