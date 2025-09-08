# Stage 1: Python builder
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN python -m venv /venv && \
    . /venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Get ffmpeg with NVENC
FROM jrottenberg/ffmpeg:8.0-nvidia as ffmpeg

# Stage 3: Final runtime image
FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"
WORKDIR /app

# Install Python runtime + minimal system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy Python environment and app
COPY --from=builder /venv /venv
COPY --from=builder /app /app

# Copy ffmpeg binaries and libs
COPY --from=ffmpeg /usr/local/ /usr/local/

COPY server.py /app/server.py

EXPOSE 5000
ENTRYPOINT ["/venv/bin/python", "server.py"]