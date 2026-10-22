---
title: PlantVision API
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# PlantVision API

FastAPI backend for [PlantVision AI](https://github.com/your-username/plantvision-web): leaf disease
classification (MobileNetV2, 38 PlantVillage classes), image-quality and unfamiliar-image checks,
Grad-CAM, zero-shot CLIP pest and symptom checks, care guidance and analysis history.

This repo holds the inference code only. Training, evaluation and the Streamlit app live in the
original `plant-health-detector` repo; `models/` contains the trained checkpoint and calibration it produced.

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.inference.download_models     # one-off CLIP download (~1.7 GB)
uvicorn api.main:app --port 8000
pytest
```

The frontend expects the API at `http://localhost:8000` by default.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PLANT_API_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated frontend origins allowed by CORS |

## Deploy to Hugging Face Spaces

1. Create a Space at <https://huggingface.co/new-space> with SDK **Docker** (blank template).
2. In the Space's **Settings → Variables and secrets**, add `PLANT_API_CORS_ORIGINS=https://your-app.vercel.app`.
3. Push this repo to the Space. Either push directly:
   ```bash
   git remote add space https://huggingface.co/spaces/your-username/plantvision-api
   git push space main
   ```
   or let GitHub do it: the `Sync to Hugging Face Space` workflow pushes `main` on every commit once the
   GitHub repo has a `HF_TOKEN` secret (write token) and a `HF_SPACE` variable (`your-username/plantvision-api`).
4. The API is then served at `https://your-username-plantvision-api.hf.space` (try `/api/health`).

Model files and images are stored with Git LFS (`.gitattributes`), which Hugging Face requires for binaries.

**History is ephemeral and shared.** `data/history.db` lives in the container, so it is wiped whenever the
Space restarts or rebuilds, and every visitor sees the same history list.
