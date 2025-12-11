"""
Upload data/train and data/test CSVs to a Hugging Face dataset repo.
Skips creating an empty commit if no files are found or nothing to upload.

Usage:
  - Set HF_AUTH_TOKEN in .env (or environment)
  - Optionally set HF_DATASET_REPO (default: teru00801/finetune1210)
  - uv run bert_thunder/finetune1210_thunder/upload_to_hf.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, CommitOperationAdd

load_dotenv(".env")

HF_TOKEN = os.getenv("HF_AUTH_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "teru00801/finetune1210")
DATA_ROOT = Path("bert_thunder/finetune1210_thunder/data")


def gather_files():
    targets = []
    for sub in ["train", "test"]:
        folder = DATA_ROOT / sub
        if not folder.exists():
            continue
        for p in folder.glob("*.csv"):
            targets.append(p)
    return targets


def main():
    if not HF_TOKEN:
        raise SystemExit("HF_AUTH_TOKEN not set")
    files = gather_files()
    if not files:
        raise SystemExit("No CSV files found under data/train or data/test.")

    ops = [
        CommitOperationAdd(path_in_repo=f"{p.relative_to(DATA_ROOT)}", path_or_fileobj=p)
        for p in files
    ]
    if not ops:
        raise SystemExit("No operations to commit; nothing uploaded.")

    api = HfApi(token=HF_TOKEN)
    api.create_commit(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message="Upload finetune1210 data",
    )
    print(f"Uploaded {len(ops)} files to {HF_DATASET_REPO}")


if __name__ == "__main__":
    main()
