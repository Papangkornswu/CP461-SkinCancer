FROM python:3.11-slim

ARG BACKBONE="ViT-B/32"
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/app DATA_DIR=/app/data

RUN apt-get update && apt-get install -y --no-install-recommends git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && useradd -m -d /home/app app
WORKDIR /app

# CPU-only torch keeps the image small; swap the index for a CUDA build if you deploy on a GPU host
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/data && chown -R app:app /app
USER app
# bake the original CLIP weights into the image so the container does not download at start-up
RUN python -c "import clip; clip.load('${BACKBONE}', device='cpu')"

EXPOSE 8501
VOLUME ["/app/data"]
HEALTHCHECK CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
