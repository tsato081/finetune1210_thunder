"""
Download training/test CSVs from a Hugging Face dataset repo.

Usage:
  1) Copy .env.example to .env and fill HF_TOKEN or HF_AUTH_TOKEN.
  2) uv run bert_thunder/finetune1210_thunder/fetch_assets.py

Notes:
  - Files are placed under bert_thunder/finetune1210_thunder/data/ regardless of
    the current working directory.
  - Adjust HF_DATASET_REPO or HF_FILES to match your repo layout.
"""

from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import hf_hub_download

# Load .env from the nearest location upward (repo root-friendly)
load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_AUTH_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "teru00801/finetune1210")
HF_FILES = os.getenv(
    "HF_FILES",
    "train/task1_cleaned.csv,train/task2_cleaned.csv,test/Hawks4.0正解データ.csv,"
    "test/Hawks ver 5.0 csv出力用.csv,test/Hawks_Revenge_test_2.csv",
)

# Always drop files next to this script (avoids cwd confusion on cloud)
DEST_DIR = BASE_DIR / "data"


def main():
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN or HF_AUTH_TOKEN must be set in .env")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    files = [f.strip() for f in HF_FILES.split(",") if f.strip()]
    if not files:
        raise SystemExit("HF_FILES is empty.")

    for fname in files:
        dest_path = DEST_DIR / Path(fname)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() or dest_path.is_symlink():
            dest_path.unlink()

        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=fname,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=DEST_DIR,
            local_dir_use_symlinks=False,  # write real files, not cache symlinks
        )
        print(f"Downloaded {fname} -> {local_path}")


if __name__ == "__main__":
    main()
