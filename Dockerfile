# Hugging Face Spaces (Docker SDK). Spaces run the container as uid 1000 and route traffic to port 7860.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch wheels keep the image several GB smaller than the default CUDA build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# The app only loads CLIP from the local cache, so fetch it once at build time (~1.7 GB).
# Same download as `python -m src.inference.download_models`; done before COPY so code edits reuse this layer.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
m = 'openai/clip-vit-large-patch14'; CLIPModel.from_pretrained(m); CLIPProcessor.from_pretrained(m)"

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
