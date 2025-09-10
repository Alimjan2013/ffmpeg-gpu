FROM jrottenberg/ffmpeg:8.0-nvidia2404

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git curl \
 && rm -rf /var/lib/apt/lists/*

# Virtual environment + Python deps
RUN python3 -m venv /venv \
 && /venv/bin/pip install --upgrade pip \
 && /venv/bin/pip install flask ffmpeg-python boto3 requests

ENV PATH="/venv/bin:$PATH"

COPY server.py /app/server.py

EXPOSE 5000

# Reset entrypoint so CMD runs normally
ENTRYPOINT []

CMD ["python", "server.py"]