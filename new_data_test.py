from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding

# -----------------------------------------------------------------------------
# User settings (EDIT HERE)
# -----------------------------------------------------------------------------
# 学習コードが保存した「best checkpoint」の epochN ディレクトリ
CKPT_DIR = Path("finetune1210_thunder/output/output_20251211_210209/epoch10")

# 推論・評価したい v5 正解付きCSV
V5_CSV = Path("finetune1210_thunder/data/test/Hawks_正解データマスター - ver 5.0 csv出力用 (4).csv")

# Task1 (Pick) の判定しきい値（学習ログの "Best Task1 threshold (F1) on val" に合わせる）
TASK1_THRESHOLD = 0.70

# 推論設定（学習時と合わせるのが無難）
MAX_LENGTH = 384
BATCH_SIZE = 64
NUM_WORKERS = 4
POOLING = "mean"  # "mean" or "cls"

# 出力先（空なら CKPT_DIR の親 output_xxx の直下に出る）
OUTPUT_ROOT_OVERRIDE: Optional[Path] = None

# -----------------------------------------------------------------------------
# Logging (same style)
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

def setup_logging(run_dir: Path) -> Path:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"infer_{ts}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return log_path


# -----------------------------------------------------------------------------
# Config-like constants (for mapping)
# -----------------------------------------------------------------------------
PICK_MAP = {"Pick": 1, "Decline": 0}
PICK_MAP_REV = {1: "Pick", 0: "Decline"}


# -----------------------------------------------------------------------------
# Device (same preference)
# -----------------------------------------------------------------------------
def setup_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("Using CUDA.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")
    return device


# -----------------------------------------------------------------------------
# CSV read / preprocess (robust for JP encodings)
# -----------------------------------------------------------------------------
def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    title_col = (
        df["title_original"]
        if "title_original" in df.columns
        else df["title"]
        if "title" in df.columns
        else pd.Series([""] * len(df))
    )
    body_col = (
        df["body_original"]
        if "body_original" in df.columns
        else df["body"]
        if "body" in df.columns
        else pd.Series([""] * len(df))
    )
    df["title"] = title_col.fillna("").astype(str)
    df["body"] = body_col.fillna("").astype(str)
    return df


# -----------------------------------------------------------------------------
# Dataset (same shape)
# -----------------------------------------------------------------------------
class MultiTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, label2id: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["title"],
            row["body"],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Task1 label
        pick_val = row.get("pick", None)
        label_bin = -100
        if pick_val is not None and not pd.isna(pick_val):
            val = str(pick_val).strip()
            if val in PICK_MAP:
                label_bin = PICK_MAP[val]
            elif val in {"1", "1.0"}:
                label_bin = 1
            elif val in {"0", "0.0"}:
                label_bin = 0

        # Task2 label
        label_cat = -100
        cat_val = row.get("category", None)
        if cat_val is not None and not pd.isna(cat_val):
            cat_val = str(cat_val).strip()
            if cat_val != "" and cat_val != "-1" and cat_val in self.label2id:
                label_cat = self.label2id[cat_val]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels_binary": torch.tensor(label_bin, dtype=torch.long),
            "labels_category": torch.tensor(label_cat, dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# Model (same heads)
# -----------------------------------------------------------------------------
class DebertaMultiTask(nn.Module):
    def __init__(self, base_model: Union[str, Path], num_task2_labels: int, pooling: str = "mean"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model, config=self.config)
        self.pooling = pooling

        hidden = self.config.hidden_size
        dropout_prob = getattr(self.config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(self.config, "hidden_dropout_prob", 0.1)

        self.dropout = nn.Dropout(dropout_prob)
        self.head_task1 = nn.Linear(hidden, 2)
        self.head_task2 = nn.Linear(hidden, num_task2_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        logits_task1 = self.head_task1(pooled)
        logits_task2 = self.head_task2(pooled)
        return logits_task1, logits_task2


# -----------------------------------------------------------------------------
# Inference/Eval (same logging style as training script)
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_test_inference(
    model: nn.Module,
    tokenizer,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: torch.device,
    csv_path: Path,
    name: str,
    threshold: float,
    run_dir: Path,
):
    if not csv_path.exists():
        logger.info("CSV not found, skip: %s", csv_path)
        return

    logger.info("Running test inference (%s) using %s", name, csv_path)

    df_test_raw = read_csv_smart(csv_path)
    df_test = preprocess_dataframe(df_test_raw)

    test_ds = MultiTaskDataset(df_test, tokenizer, MAX_LENGTH, label2id)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=MAX_LENGTH)
    pin_memory = device.type == "cuda"
    persistent_workers = NUM_WORKERS > 0

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model.eval()
    t1_preds, t1_trues, t2_preds, t2_trues = [], [], [], []
    pred_pick_all, pred_cat_all, prob_pick_all = [], [], []
    topk_hits = {1: 0, 2: 0, 3: 0}
    topk_total = 0

    for batch in tqdm(test_loader, desc=f"Test-{name}"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels_b = batch["labels_binary"].to(device, non_blocking=True)
        labels_c = batch["labels_category"].to(device, non_blocking=True)

        logits_t1, logits_t2 = model(input_ids=input_ids, attention_mask=attention_mask)

        prob_b = torch.softmax(logits_t1, dim=-1)[:, 1]
        pred_b = (prob_b >= threshold).long()
        pred_pick_all.extend(pred_b.cpu().tolist())
        prob_pick_all.extend(prob_b.cpu().tolist())

        mask_b = labels_b != -100
        if mask_b.any():
            t1_preds.extend(pred_b[mask_b].cpu().tolist())
            t1_trues.extend(labels_b[mask_b].cpu().tolist())

        pred_c = torch.argmax(logits_t2, dim=1)
        pred_cat_all.extend(pred_c.cpu().tolist())

        mask_c = labels_c != -100
        if mask_c.any():
            t2_preds.extend(pred_c[mask_c].cpu().tolist())
            t2_trues.extend(labels_c[mask_c].cpu().tolist())
            # top-k accuracy for Task2
            k_max = min(3, logits_t2.size(-1))
            topk = torch.topk(logits_t2[mask_c], k=k_max, dim=1).indices.cpu()
            true_ids = labels_c[mask_c].cpu()
            topk_total += len(true_ids)
            for k in [1, 2, 3]:
                if k > k_max:
                    continue
                hits = (topk[:, :k] == true_ids.unsqueeze(1)).any(dim=1).sum().item()
                topk_hits[k] += hits

    # ---- logging (same style) ----
    if t1_trues:
        acc_t1 = accuracy_score(t1_trues, t1_preds)
        logger.info("[%s] Test Task1 accuracy=%.4f (n=%d)", name, acc_t1, len(t1_trues))
        logger.info(
            "[%s] Test Task1 report:\n%s",
            name,
            classification_report(
                t1_trues,
                t1_preds,
                target_names=[PICK_MAP_REV[0], PICK_MAP_REV[1]],
                digits=4,
                zero_division=0,
            ),
        )
    else:
        logger.info("[%s] Test Task1: no ground-truth labels found (pick column empty?)", name)

    if t2_trues:
        unique_labels_test = sorted(list(set(t2_trues) | set(t2_preds)))
        logger.info("[%s] Test Task2 accuracy=%.4f (n=%d)", name, accuracy_score(t2_trues, t2_preds), len(t2_trues))
        logger.info(
            "[%s] Test Task2 report:\n%s",
            name,
            classification_report(
                t2_trues,
                t2_preds,
                labels=unique_labels_test,
                target_names=[id2label[i] for i in unique_labels_test],
                digits=4,
                zero_division=0,
            ),
        )
        if topk_total > 0:
            topk_acc = {k: topk_hits[k] / topk_total for k in [1, 2, 3]}
            logger.info(
                "[%s] Task2 top-k accuracy: k=1 %.4f, k=2 %.4f, k=3 %.4f",
                name, topk_acc[1], topk_acc[2], topk_acc[3]
            )
    else:
        logger.info("[%s] Test Task2: no ground-truth labels found (category column empty?)", name)

    # ---- save predictions csv (same behavior) ----
    out_df = df_test_raw.copy()
    if len(pred_pick_all) == len(out_df):
        out_df["model_pick"] = [PICK_MAP_REV.get(int(p), str(p)) for p in pred_pick_all]
        out_df["model_pick_prob"] = prob_pick_all
    if len(pred_cat_all) == len(out_df):
        out_df["model_category"] = [id2label.get(int(p), str(p)) for p in pred_cat_all]

    pred_path = run_dir / f"test_predictions_{name}_thr{threshold:.2f}.csv"
    out_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    logger.info("Saved test predictions to %s", pred_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if not CKPT_DIR.exists():
        raise FileNotFoundError(f"CKPT_DIR not found: {CKPT_DIR}")

    ckpt_path = CKPT_DIR / "checkpoint.pt"
    enc_dir = CKPT_DIR / "encoder"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint.pt not found: {ckpt_path}")
    if not enc_dir.exists():
        raise FileNotFoundError(f"encoder dir not found: {enc_dir}")

    # output dir: trainingと同じく run_dir を作る（logs / 予測csv をまとめる）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (OUTPUT_ROOT_OVERRIDE or CKPT_DIR.parent) / f"infer_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(run_dir)
    logger.info("Logging to %s", log_path)

    device = setup_device()

    # load checkpoint + label map
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    id2label_raw = ckpt.get("label_map", None)
    if id2label_raw is None:
        lm = CKPT_DIR / "label_map.json"
        if not lm.exists():
            raise FileNotFoundError("label_map not found in checkpoint and label_map.json not found.")
        with open(lm, "r", encoding="utf-8") as f:
            id2label_raw = json.load(f)

    # json経由だと key が str になりがちなので int化
    id2label: Dict[int, str] = {int(k): v for k, v in dict(id2label_raw).items()}
    label2id: Dict[str, int] = {v: k for k, v in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(enc_dir)

    model = DebertaMultiTask(enc_dir, num_task2_labels=len(label2id), pooling=POOLING)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)

    logger.info("Loaded model from %s", CKPT_DIR)
    logger.info("Inference CSV: %s", V5_CSV)
    logger.info("Task1 threshold: %.3f", TASK1_THRESHOLD)

    run_test_inference(
        model=model,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        device=device,
        csv_path=V5_CSV,
        name="v5_master",
        threshold=TASK1_THRESHOLD,
        run_dir=run_dir,
    )

    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
