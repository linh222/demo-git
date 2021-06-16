FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-2020-12-19

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100
ENV PYTHONOPTIMIZE=2

COPY ./app /app
COPY requirements.txt /app/
# COPY .env /app/

WORKDIR /app

ENV PYTHONPATH=/

RUN pip install --upgrade pip \
    && pip install --ignore-installed -r requirements.txt \
    && rm -rf /root/.cache

EXPOSE 80