FROM python:3.10-slim

# System deps required by OpenCV (headless) and general libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
# Use headless variant so OpenCV works without a display
RUN pip install --no-cache-dir opencv-python-headless && \
    grep -v '^opencv-python$' requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY app/ app/

ENV PYTHONPATH=/workspace

WORKDIR /workspace/app
CMD ["python", "main.py"]
