FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04

WORKDIR /app

# Install Python and ffmpeg with GPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Flask + ffmpeg-python for testing
RUN pip install flask ffmpeg-python boto3 requests

COPY server.py /app/server.py

EXPOSE 5000
CMD ["python3", "server.py"]