FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04

WORKDIR /app

# Install Python and ffmpeg with GPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment and install dependencies
RUN python3 -m venv /venv \
 && /venv/bin/pip install --upgrade pip \
 && /venv/bin/pip install flask ffmpeg-python boto3 requests

# Ensure venv is always used
ENV PATH="/venv/bin:$PATH"

COPY server.py /app/server.py

EXPOSE 5000


CMD ["python", "server.py"]