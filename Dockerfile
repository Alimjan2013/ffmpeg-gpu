# Dockerfile

FROM jrottenberg/ffmpeg:8.0-ubuntu2404-edge


ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    MY_VARIABLE="This is a default value"

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.12 -m venv $VIRTUAL_ENV

RUN . $VIRTUAL_ENV/bin/activate && \
    pip install --upgrade pip && \
    pip install flask requests boto3 ffmpeg-python==0.2.0

COPY server.py /app/server.py

EXPOSE 5000

ENTRYPOINT ["python3.12", "server.py"]
