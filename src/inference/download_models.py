"""One-off download of the optional CLIP pest-screen model (~1.7 GB, free, no token).

Run:  python -m src.inference.download_models

The app itself never downloads anything; without this model it falls back to
the weaker ImageNet insect screen.
"""

from transformers import CLIPModel, CLIPProcessor

from src import config


def main():
    print(f"Downloading {config.PEST_CLIP_MODEL} into the Hugging Face cache...")
    CLIPModel.from_pretrained(config.PEST_CLIP_MODEL)
    CLIPProcessor.from_pretrained(config.PEST_CLIP_MODEL)
    print("Done. The app will now use the CLIP pest screen.")


if __name__ == "__main__":
    main()
