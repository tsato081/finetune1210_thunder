"""
Download training/test CSVs from a Hugging Face dataset repo using HF_TOKEN.

Usage:
  1) Copy .env.example to .env and fill HF_TOKEN, HF_DATASET_REPO, HF_FILES.
  2) uv run bert_thunder/finetune_1210/fetch_assets.py

Notes:
  - Files are downloaded into bert_thunder/data/ by default.
  - Adjust DEST_DIR or HF_FILES to match your repo layout.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "your-username/finetune1210-data")
HF_FILES = os.getenv(
    "HF_FILES",
    "train/task1_cleaned.csv,train/task2_cleaned.csv,test/Hawks4.0正解データ.csv,"
    "test/Hawks ver 5.0 csv出力用.csv,test/Hawks_Revenge_test_2.csv",
)

DEST_DIR = Path("data")


def main():
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN must be set in .env")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    files = [f.strip() for f in HF_FILES.split(",") if f.strip()]
    if not files:
        raise SystemExit("HF_FILES is empty.")

    for fname in files:
        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=fname,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        dest_path = DEST_DIR / Path(fname)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).replace(dest_path)
        print(f"Downloaded {fname} -> {dest_path}")


if __name__ == "__main__":
    main()
