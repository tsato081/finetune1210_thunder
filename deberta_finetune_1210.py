from __future__ import annotations

"""
Stage2 multi-task fine-tuning for DeBERTa (Pick/Decline + 96-category).

Design goals (per finetune_architecture.md):
- Start from Stage1 encoder checkpoint (Pick large-scale pretrain; default local models/deberta_pick_pretrain_cuda)
- Simple heads: mean pooling -> linear for Task1/Task2
- Loss: plain CE with minimal regularization
  * Task1 label smoothing = 0.0 (sharp boundary)
  * Task2 label smoothing ~0.1, inverse-freq class weights clipped to avoid extremes
  * Total loss = lambda1 * L_task1 + lambda2 * L_task2 (no uncertainty weighting)
  * R-Drop optional: symmetric KL on both tasks (Task2主体、Task1は係数小さめ推奨)
- Training: up to 10 epochs + early stopping, LLRD (0.8), lr 2e-5, warmup 0.06, bs 32-64

Inputs (same split style as deberta_finetune_mps.py):
  task1_csv: Pick/Decline data (e.g., bert_thunder/data/train/task1_cleaned.csv)
  task2_csv: Category data (e.g., bert_thunder/data/train/task2_cleaned.csv)
  Each CSV needs title/body (+ _original) and pick/category columns as applicable.
"""

import copy
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.cuda.amp import autocast, GradScaler


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

# Prefer MPS -> CUDA -> CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

logger = logging.getLogger(__name__)


def setup_logging(run_dir: Path) -> Path:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{ts}.log"

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

# Enable Japanese font rendering for matplotlib outputs (if available)
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    logger.warning("japanize_matplotlib is not installed; plots may not render Japanese labels correctly.")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    model_name: Union[str, Path] = Path(__file__).resolve().parent / "deberta_v3_mlm"  # default to local Stage1 checkpoint
    base_dir: Path = Path(__file__).resolve().parent
    
    output_root: Path = base_dir / "output"

    task1_csv: Path = base_dir / "data/train/task1_cleaned_all.csv"
    task2_csv: Path = base_dir / "data/train/task2_cleaned.csv"
    test_csv: Path = base_dir / "data/test/Hawks4.0正解データ.csv"
    test_csv_v5: Path = base_dir / "data/test/Hawks ver 5.0 csv出力用.csv"
    test2_csv: Path = base_dir / "data/test/Hawks_Revenge_test_2.csv"
    val_size: float = 0.1
    max_length: int = 384
    seed: int = 42

    # Optim / schedule
    base_learning_rate: float = 2e-5
    lr_decay: float = 0.8  # LLRD decay per layer
    weight_decay: float = 0.01
    num_train_epochs: int = 10
    early_stopping_patience: int = 1  # stop if no macro F1 improvement for this many epochs
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 4

    # Loss balance
    lambda_task1: float = 0.4
    lambda_task2: float = 1.0

    hierarchy_weight: float = 0.1  # Decline時にTask2分布をフラット化する正則化係数
    rdrop_alpha_task1: float = 0.0  # 0〜0.3くらいで様子見
    rdrop_alpha_task2: float = 0.7  # 0.5〜1.0推奨（Task2主体）

    # Loss details
    label_smoothing_task1: float = 0.0
    label_smoothing_task2: float = 0.1
    min_category_weight: float = 0.6  # floor for frequent classes
    max_category_weight: Optional[float] = 5.0  # cap for extremely rare classes
    pick_class_weights: Optional[List[float]] = None  # [Decline, Pick]; if None, auto from data

    pooling: str = "mean"  # "mean" or "cls"
    save_total_limit: int = 2

    pick_map: Dict[str, int] = None
    pick_map_rev: Dict[int, str] = None
    use_amp: bool = True  
    torch_compile: bool = False 

CFG = Config(
    pick_map={"Pick": 1, "Decline": 0},
    pick_map_rev={1: "Pick", 0: "Decline"},
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def setup_device_and_seed(cfg: Config) -> torch.device:
    set_seed(cfg.seed)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS.")
        torch.mps.manual_seed(cfg.seed)  # type: ignore[attr-defined]
        cfg.use_amp = False  # このスクリプトでは MPS で AMP は使わない
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # 入力サイズが固定なので ON で OK
        logger.info("Using CUDA.")
        cfg.use_amp = bool(cfg.use_amp)       # CUDA のときだけ AMP
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")
        cfg.use_amp = False                   # CPU では AMP 無効
    return device



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


def create_label_mapping(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    cats = df["category"].dropna().astype(str)
    unique = sorted([c for c in cats.unique() if c.strip() != "" and c != "-1"])
    label2id = {c: i for i, c in enumerate(unique)}
    id2label = {i: c for c, i in label2id.items()}
    logger.info("Detected %d Task2 categories.", len(label2id))
    return label2id, id2label


def compute_inverse_freq_weights(
    values: List[int],
    num_classes: int,
    min_clip: Optional[float] = None,
    max_clip: Optional[float] = None,
) -> torch.Tensor:
    arr = np.array([v for v in values if v != -100], dtype=np.int64)
    if arr.size == 0:
        return torch.ones(num_classes, dtype=torch.float)
    counts = np.bincount(arr, minlength=num_classes)
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.mean()
    if min_clip is not None:
        weights = np.maximum(weights, min_clip)
    if max_clip is not None:
        weights = np.minimum(weights, max_clip)
    return torch.tensor(weights, dtype=torch.float)


# -----------------------------------------------------------------------------
# Dataset
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
            if val in CFG.pick_map:
                label_bin = CFG.pick_map[val]
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
# Model
# -----------------------------------------------------------------------------
class DebertaMultiTask(nn.Module):
    def __init__(
        self,
        base_model: str,
        label2id: Dict[str, int],
        pooling: str = "mean",
    ):
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
        self.head_task2 = nn.Linear(hidden, len(label2id))

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
# Training helpers
# -----------------------------------------------------------------------------
def compute_hierarchy_regularizer(logits_task2, labels_binary, flatten_weight: float = 1.0) -> torch.Tensor:
    """Encourage Task2 distribution to be flat for Decline samples."""
    decline_mask = labels_binary == 0
    if not decline_mask.any():
        return logits_task2.new_tensor(0.0)
    logits_decline = logits_task2[decline_mask]
    if logits_decline.numel() == 0:
        return logits_task2.new_tensor(0.0)
    log_probs = F.log_softmax(logits_decline, dim=-1)
    num_classes = log_probs.size(-1)
    uniform = torch.full_like(log_probs, fill_value=1.0 / num_classes)
    kl = F.kl_div(log_probs, uniform, reduction="batchmean")
    return flatten_weight * kl


def compute_losses(
    logits_task1,
    logits_task2,
    labels_binary,
    labels_category,
    ce_t1: nn.CrossEntropyLoss,
    ce_t2: nn.CrossEntropyLoss,
    lambda1: float,
    lambda2: float,
    hierarchy_weight: float = 0.0,
    logits_task1_b=None,
    logits_task2_b=None,
    alpha_t1: float = 0.0,
    alpha_t2: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loss1 = torch.tensor(0.0, device=logits_task1.device)
    loss2 = torch.tensor(0.0, device=logits_task1.device)

    mask1 = labels_binary != -100
    if mask1.any():
        if logits_task1_b is not None:
            loss1 = 0.5 * (
                ce_t1(logits_task1[mask1], labels_binary[mask1])
                + ce_t1(logits_task1_b[mask1], labels_binary[mask1])
            )
        else:
            loss1 = ce_t1(logits_task1[mask1], labels_binary[mask1])

    mask2 = labels_category != -100
    if mask2.any():
        if logits_task2_b is not None:
            loss2 = 0.5 * (
                ce_t2(logits_task2[mask2], labels_category[mask2])
                + ce_t2(logits_task2_b[mask2], labels_category[mask2])
            )
        else:
            loss2 = ce_t2(logits_task2[mask2], labels_category[mask2])

    kl1 = torch.tensor(0.0, device=logits_task1.device)
    kl2 = torch.tensor(0.0, device=logits_task1.device)
    if alpha_t1 > 0 and logits_task1_b is not None and mask1.any():
        kl1 = symmetric_kl(logits_task1[mask1], logits_task1_b[mask1]) * alpha_t1
    if alpha_t2 > 0 and logits_task2_b is not None and mask2.any():
        kl2 = symmetric_kl(logits_task2[mask2], logits_task2_b[mask2]) * alpha_t2

    hier = compute_hierarchy_regularizer(logits_task2, labels_binary, flatten_weight=hierarchy_weight)
    total = lambda1 * (loss1 + kl1) + lambda2 * (loss2 + kl2) + hier
    return total, loss1.detach(), loss2.detach(), hier.detach()


def build_optimizer(model: nn.Module, cfg: Config):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    lr = cfg.base_learning_rate
    decay = cfg.lr_decay
    num_layers = model.config.num_hidden_layers

    def lr_for(name: str) -> float:
        if "encoder.layer." in name:
            try:
                layer_id = int(name.split("encoder.layer.")[1].split(".")[0])
                return lr * (decay ** (num_layers - layer_id - 1))
            except Exception:
                return lr
        if "embeddings" in name:
            return lr * (decay ** num_layers)
        return lr

    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_lr = lr_for(name)
        wd = 0.0 if any(nd in name for nd in no_decay) else cfg.weight_decay
        params.append({"params": [param], "lr": group_lr, "weight_decay": wd})
    fused = torch.cuda.is_available()
    try:
        optimizer = torch.optim.AdamW(params, fused=fused)
    except TypeError:
        optimizer = torch.optim.AdamW(params)
    return optimizer


def evaluate(model, dataloader, device, ce_t1, ce_t2, lambda1, lambda2, hierarchy_weight: float, use_amp: bool = False):
    model.eval()
    all_labels_t1, all_preds_t1 = [], []
    all_labels_t2, all_preds_t2 = [], []
    losses_t1, losses_t2, losses_total, losses_hier = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels_binary = batch["labels_binary"].to(device, non_blocking=True)
            labels_category = batch["labels_category"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits_t1, logits_t2 = model(input_ids=input_ids, attention_mask=attention_mask)
                total_loss, l1, l2, hier = compute_losses(
                    logits_t1,
                    logits_t2,
                    labels_binary,
                    labels_category,
                    ce_t1,
                    ce_t2,
                    lambda1,
                    lambda2,
                    hierarchy_weight,
                )

            losses_total.append(total_loss.item())
            losses_hier.append(hier.item())
            if labels_binary.ne(-100).any():
                losses_t1.append(l1.item())
                pred_t1 = logits_t1.argmax(dim=-1)
                mask = labels_binary != -100
                all_labels_t1.extend(labels_binary[mask].cpu().tolist())
                all_preds_t1.extend(pred_t1[mask].cpu().tolist())
            if labels_category.ne(-100).any():
                losses_t2.append(l2.item())
                pred_t2 = logits_t2.argmax(dim=-1)
                mask = labels_category != -100
                all_labels_t2.extend(labels_category[mask].cpu().tolist())
                all_preds_t2.extend(pred_t2[mask].cpu().tolist())

    metrics = {}
    if all_labels_t1:
        acc1 = accuracy_score(all_labels_t1, all_preds_t1)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels_t1, all_preds_t1, labels=[1], zero_division=0
        )
        metrics.update(
            {
                "task1_acc": acc1,
                "task1_precision_pick": prec[0],
                "task1_recall_pick": rec[0],
                "task1_f1_pick": f1[0],
                "task1_loss": float(np.mean(losses_t1)) if losses_t1 else 0.0,
            }
        )
    if all_labels_t2:
        acc2 = accuracy_score(all_labels_t2, all_preds_t2)
        metrics.update(
            {
                "task2_acc": acc2,
                "task2_macro_f1": f1_score(all_labels_t2, all_preds_t2, average="macro"),
                "task2_weighted_f1": f1_score(all_labels_t2, all_preds_t2, average="weighted"),
                "task2_loss": float(np.mean(losses_t2)) if losses_t2 else 0.0,
            }
        )
    metrics["total_loss"] = float(np.mean(losses_total)) if losses_total else 0.0
    metrics["hier_loss"] = float(np.mean(losses_hier)) if losses_hier else 0.0

    return metrics, all_labels_t2, all_preds_t2

def compute_hierarchy_stats(model, dataloader, device, pick_low_thresh: float = 0.4, decline_high_thresh: float = 0.6, use_amp: bool = False):
    model.eval()
    pick_low, pick_total = 0, 0
    decline_high, decline_total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Hierarchy", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits_t1, logits_t2 = model(input_ids=input_ids, attention_mask=attention_mask)
            prob_t1 = F.softmax(logits_t1, dim=-1)  # [:,1] is Pick
            prob_t2 = F.softmax(logits_t2, dim=-1)

            pred_t1 = prob_t1.argmax(dim=-1)
            max_t2 = prob_t2.max(dim=-1).values

            pick_mask = pred_t1 == 1
            decline_mask = pred_t1 == 0

            pick_total += pick_mask.sum().item()
            decline_total += decline_mask.sum().item()

            if pick_mask.any():
                pick_low += (max_t2[pick_mask] < pick_low_thresh).sum().item()
            if decline_mask.any():
                decline_high += (max_t2[decline_mask] > decline_high_thresh).sum().item()

    stats = {
        "pick_total": pick_total,
        "pick_low_conf_count": pick_low,
        "pick_low_conf_ratio": pick_low / pick_total if pick_total else 0.0,
        "decline_total": decline_total,
        "decline_high_conf_count": decline_high,
        "decline_high_conf_ratio": decline_high / decline_total if decline_total else 0.0,
        "pick_low_thresh": pick_low_thresh,
        "decline_high_thresh": decline_high_thresh,
    }
    return stats


def plot_curves(history: List[Dict[str, float]], out_dir: Path):
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history) + 1))

    def get(key, default=0.0):
        return [h.get(key, default) for h in history]

    curves = [
        ("Task1 Loss", get("train_task1_loss"), get("val_task1_loss")),
        ("Task2 Loss", get("train_task2_loss"), get("val_task2_loss")),
        ("Task1 Acc/F1", get("train_task1_acc"), get("val_task1_acc")),
        ("Task2 Acc/MacroF1", get("train_task2_acc"), get("val_task2_macro_f1")),
    ]

    for title, train_vals, val_vals in curves:
        plt.figure()
        plt.plot(epochs, train_vals, label="train")
        plt.plot(epochs, val_vals, label="val")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = out_dir / f"{title.lower().replace(' ', '_').replace('/', '_')}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def compute_task1_scores(labels: List[int], preds: List[int]) -> Dict[str, float]:
    if not labels:
        return {"acc": 0.0, "precision_pick": 0.0, "recall_pick": 0.0, "f1_pick": 0.0}
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, labels=[1], zero_division=0)
    return {
        "acc": acc,
        "precision_pick": prec[0],
        "recall_pick": rec[0],
        "f1_pick": f1[0],
    }


def compute_task2_scores(labels: List[int], preds: List[int]) -> Dict[str, float]:
    if not labels:
        return {"acc": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    return {
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def symmetric_kl(logits_p, logits_q):
    p_log = F.log_softmax(logits_p, dim=-1)
    q_log = F.log_softmax(logits_q, dim=-1)
    p = p_log.exp()
    q = q_log.exp()
    kl_pq = F.kl_div(p_log, q, reduction="batchmean")
    kl_qp = F.kl_div(q_log, p, reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)


def collect_task1_probs(model, dataloader, device, use_amp: bool = False):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collect Task1 probs", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels_b = batch["labels_binary"].to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits_t1, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            prob_b = torch.softmax(logits_t1, dim=-1)[:, 1]
            mask_b = labels_b != -100
            if mask_b.any():
                probs.extend(prob_b[mask_b].cpu().tolist())
                labels.extend(labels_b[mask_b].cpu().tolist())
    return probs, labels


def grid_search_threshold(probs: List[float], labels: List[int], thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.3, 0.7, 41)  # 0.3〜0.7を細かく
    best_f1 = {"threshold": 0.5, "acc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    best_acc = {"threshold": 0.5, "acc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    labels_arr = np.array(labels)
    probs_arr = np.array(probs)
    for th in thresholds:
        preds = (probs_arr >= th).astype(int)
        acc = accuracy_score(labels_arr, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels_arr, preds, labels=[1], zero_division=0)
        prec, rec, f1 = float(prec[0]), float(rec[0]), float(f1[0])
        if f1 > best_f1["f1"] or (f1 == best_f1["f1"] and acc > best_f1["acc"]):
            best_f1 = {"threshold": float(th), "acc": float(acc), "f1": float(f1), "precision": prec, "recall": rec}
        if acc > best_acc["acc"] or (acc == best_acc["acc"] and f1 > best_acc["f1"]):
            best_acc = {"threshold": float(th), "acc": float(acc), "f1": float(f1), "precision": prec, "recall": rec}
    return best_f1, best_acc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Prepare run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(CFG.output_root) / f"output_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(run_dir)
    logger.info("Logging to %s", log_path)

    device = setup_device_and_seed(CFG)

    model_name = CFG.model_name
    resolved_local = None
    if isinstance(model_name, (str, Path)):
        candidates = []
        p = Path(model_name).expanduser()
        candidates.append(p)
        if not p.is_absolute():
            candidates.append((CFG.base_dir / p).expanduser())
        resolved_local = next((c for c in candidates if c.exists()), None)

    if resolved_local:
        model_name = resolved_local
        logger.info("Loading model from local path: %s", model_name)
    elif isinstance(CFG.model_name, Path):
        raise FileNotFoundError(
            f"Local model path not found: {Path(CFG.model_name).resolve()} (set CFG.model_name to an HF repo id if you want to fetch from hub)"
        )
    else:
        model_name = str(model_name)
        logger.info("Loading model from hub repo id: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not os.path.exists(CFG.task1_csv) or not os.path.exists(CFG.task2_csv):
        raise FileNotFoundError("Task CSV not found.")

    df_t1 = preprocess_dataframe(pd.read_csv(CFG.task1_csv))
    df_t2 = preprocess_dataframe(pd.read_csv(CFG.task2_csv))
    label2id, id2label = create_label_mapping(df_t2)

    # Save label map for downstream use
    with open(run_dir / "label_map.json", "w") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    # Split train/val separately then concatenate (keeps both task labels present)
    t1_train, t1_val = train_test_split(
        df_t1, test_size=CFG.val_size, stratify=df_t1["pick"], random_state=CFG.seed
    )
    try:
        t2_train, t2_val = train_test_split(
            df_t2, test_size=CFG.val_size, stratify=df_t2["category"], random_state=CFG.seed
        )
    except Exception:
        t2_train, t2_val = train_test_split(df_t2, test_size=CFG.val_size, random_state=CFG.seed)

    df_train = pd.concat([t1_train, t2_train], ignore_index=True).sample(
        frac=1.0, random_state=CFG.seed
    ).reset_index(drop=True)
    df_val = pd.concat([t1_val, t2_val], ignore_index=True).reset_index(drop=True)

    train_ds = MultiTaskDataset(df_train, tokenizer, CFG.max_length, label2id)
    val_ds = MultiTaskDataset(df_val, tokenizer, CFG.max_length, label2id)

    pin_memory = device.type == "cuda"
    persistent_workers = CFG.dataloader_num_workers > 0

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=CFG.max_length)
    cat_series = df_train.get("category", pd.Series([""] * len(df_train))).astype(str).str.strip()
    mask_t2 = cat_series.notna() & (cat_series != "") & (cat_series != "-1") & cat_series.isin(label2id)
    cat_ids = cat_series.map(label2id)  # NaN含む

    # Task1 (=category無し) は等確率。Task2だけ rare を多めに引く
    weights_np = np.ones(len(df_train), dtype=np.float32)
    if mask_t2.any():
        counts = cat_ids[mask_t2].astype(int).value_counts()
        inv = (1.0 / counts).to_dict()  # {class_id: inv_freq}
        weights_np[mask_t2.values] = (
            cat_ids[mask_t2].astype(int).map(inv).astype(np.float32).values
        )
        # Task2 の平均重みを 1 に正規化（t1:t2比率を極端に崩さない）
        weights_np[mask_t2.values] /= weights_np[mask_t2.values].mean()

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights_np, dtype=torch.double),
        num_samples=len(train_ds),  # 1 epoch あたりのサンプル数（元と同じ）
        replacement=True,           # バランス取りのため必須
        generator=torch.Generator().manual_seed(CFG.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.per_device_train_batch_size,
        shuffle=False,              # sampler使用時はFalse
        sampler=sampler,
        num_workers=CFG.dataloader_num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.per_device_eval_batch_size,
        shuffle=False,
        num_workers=CFG.dataloader_num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = DebertaMultiTask(model_name, label2id, pooling=CFG.pooling).to(device)

    # Losses
    if CFG.pick_class_weights is not None:
        pick_weights = torch.tensor(CFG.pick_class_weights, dtype=torch.float)
    else:
        counts = t1_train["pick"].map(CFG.pick_map).dropna().astype(int).value_counts()
        total = counts.sum()
        w0 = total / (2 * counts.get(0, 1))
        w1 = total / (2 * counts.get(1, 1))
        pick_weights = torch.tensor([w0, w1], dtype=torch.float)

    cat_labels_train = t2_train["category"].map(label2id).fillna(-100).astype(int).tolist()
    cat_weights = compute_inverse_freq_weights(
        cat_labels_train,
        num_classes=len(label2id),
        min_clip=CFG.min_category_weight,
        max_clip=CFG.max_category_weight,
    )
    pick_weights = pick_weights.to(device)
    cat_weights = cat_weights.to(device)

    ce_t1 = nn.CrossEntropyLoss(weight=pick_weights, label_smoothing=CFG.label_smoothing_task1)
    ce_t2 = nn.CrossEntropyLoss(weight=cat_weights, label_smoothing=CFG.label_smoothing_task2)

    optimizer = build_optimizer(model, CFG)
    num_update_steps_per_epoch = max(1, len(train_loader) // CFG.gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * CFG.num_train_epochs
    num_warmup = int(num_training_steps * CFG.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_training_steps)

    scaler = GradScaler(enabled=CFG.use_amp)

    best_macro_f1 = -1.0
    best_path = None
    best_state = None
    best_epoch = -1
    history: List[Dict[str, float]] = []
    patience_counter = 0

    for epoch in range(CFG.num_train_epochs):
        model.train()
        running_loss, running_t1, running_t2, running_hier = [], [], [], []
        train_labels_t1, train_preds_t1 = [], []
        train_labels_t2, train_preds_t2 = [], []

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{CFG.num_train_epochs}", leave=False)
        ):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels_binary = batch["labels_binary"].to(device, non_blocking=True)
            labels_category = batch["labels_category"].to(device, non_blocking=True)

            # forward passes for R-Drop (second pass only if alpha > 0)
            with autocast(enabled=CFG.use_amp):
                logits_t1, logits_t2 = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_t1_b = logits_t2_b = None
                if CFG.rdrop_alpha_task1 > 0 or CFG.rdrop_alpha_task2 > 0:
                    logits_t1_b, logits_t2_b = model(input_ids=input_ids, attention_mask=attention_mask)

                loss, l1, l2, hier = compute_losses(
                    logits_t1,
                    logits_t2,
                    labels_binary,
                    labels_category,
                    ce_t1,
                    ce_t2,
                    CFG.lambda_task1,
                    CFG.lambda_task2,
                    CFG.hierarchy_weight,
                    logits_task1_b=logits_t1_b,
                    logits_task2_b=logits_t2_b,
                    alpha_t1=CFG.rdrop_alpha_task1,
                    alpha_t2=CFG.rdrop_alpha_task2,
                )

            loss = loss / CFG.gradient_accumulation_steps

            if CFG.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % CFG.gradient_accumulation_steps == 0:
                if CFG.use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss.append(loss.item())
            running_t1.append(l1.item())
            running_t2.append(l2.item())
            running_hier.append(hier.item())

            # Train metrics (rough, on-the-fly)
            with torch.no_grad():
                pred_t1 = logits_t1.argmax(dim=-1)
                mask1 = labels_binary != -100
                train_labels_t1.extend(labels_binary[mask1].cpu().tolist())
                train_preds_t1.extend(pred_t1[mask1].cpu().tolist())

                pred_t2 = logits_t2.argmax(dim=-1)
                mask2 = labels_category != -100
                train_labels_t2.extend(labels_category[mask2].cpu().tolist())
                train_preds_t2.extend(pred_t2[mask2].cpu().tolist())

        train_loss = float(np.mean(running_loss)) if running_loss else 0.0
        train_t1 = float(np.mean(running_t1)) if running_t1 else 0.0
        train_t2 = float(np.mean(running_t2)) if running_t2 else 0.0
        train_hier = float(np.mean(running_hier)) if running_hier else 0.0

        val_metrics, val_labels_t2, val_preds_t2 = evaluate(
            model,
            val_loader,
            device,
            ce_t1,
            ce_t2,
            CFG.lambda_task1,
            CFG.lambda_task2,
            CFG.hierarchy_weight,
            CFG.use_amp,
        )

        # Train-side metrics for logging
        train_t1_scores = compute_task1_scores(train_labels_t1, train_preds_t1)
        train_t2_scores = compute_task2_scores(train_labels_t2, train_preds_t2)

        epoch_record = {
            "train_loss": train_loss,
            "train_task1_loss": train_t1,
            "train_task2_loss": train_t2,
            "train_hier_loss": train_hier,
            "train_task1_acc": train_t1_scores["acc"],
            "train_task1_f1_pick": train_t1_scores["f1_pick"],
            "train_task2_acc": train_t2_scores["acc"],
            "train_task2_macro_f1": train_t2_scores["macro_f1"],
            "val_task1_loss": val_metrics.get("task1_loss", 0.0),
            "val_task2_loss": val_metrics.get("task2_loss", 0.0),
            "val_total_loss": val_metrics.get("total_loss", 0.0),
            "val_task1_acc": val_metrics.get("task1_acc", 0.0),
            "val_task1_f1_pick": val_metrics.get("task1_f1_pick", 0.0),
            "val_task2_acc": val_metrics.get("task2_acc", 0.0),
            "val_task2_macro_f1": val_metrics.get("task2_macro_f1", 0.0),
            "val_task2_weighted_f1": val_metrics.get("task2_weighted_f1", 0.0),
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %d | train_loss=%.4f (t1=%.4f, t2=%.4f) | val: %s",
            epoch + 1,
            train_loss,
            train_t1,
            train_t2,
            {k: round(v, 4) for k, v in val_metrics.items()},
        )
        logger.info("lambda_task1=%.2f lambda_task2=%.2f", CFG.lambda_task1, CFG.lambda_task2)

        macro_f1 = val_metrics.get("task2_macro_f1", -1.0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_path = run_dir / f"epoch{epoch+1}"
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            best_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": id2label,
                    "config": CFG.__dict__,
                },
                best_path / "checkpoint.pt",
            )
            model.encoder.save_pretrained(best_path / "encoder")
            tokenizer.save_pretrained(best_path / "encoder")
            with open(best_path / "label_map.json", "w") as f:
                json.dump(id2label, f, ensure_ascii=False, indent=2)
            patience_counter = 0
        else:
            patience_counter += 1

        # Optional per-class report for Task2 at epoch end
        if val_labels_t2:
            report = classification_report(
                val_labels_t2, val_preds_t2, digits=4, output_dict=False, zero_division=0
            )
            logger.info("Task2 per-class summary (val):\n%s", report)

        if CFG.early_stopping_patience > 0 and patience_counter >= CFG.early_stopping_patience:
            logger.info(
                "Early stopping triggered (no val Task2 macro F1 improvement for %d epochs).", CFG.early_stopping_patience
            )
            break

    plot_curves(history, run_dir / "plots")

    logger.info("Best val macro F1 (Task2): %.4f (epoch %d)", best_macro_f1, best_epoch)
    if best_path:
        logger.info("Best checkpoint saved to %s", best_path)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Validation detailed stats with best model
    val_metrics, val_labels_t2_best, val_preds_t2_best = evaluate(
    model, val_loader, device, ce_t1, ce_t2, CFG.lambda_task1, CFG.lambda_task2, CFG.hierarchy_weight, CFG.use_amp
    )

    logger.info("Best epoch val metrics: %s", {k: round(v, 4) for k, v in val_metrics.items()})

    # Task1 threshold search on val
    best_t1_threshold = 0.5
    best_t1_threshold_acc = 0.5
    probs_val, labels_val = collect_task1_probs(model, val_loader, device, CFG.use_amp)
    if labels_val:
        best_f1_th, best_acc_th = grid_search_threshold(probs_val, labels_val)
        best_t1_threshold = best_f1_th["threshold"]
        best_t1_threshold_acc = best_acc_th["threshold"]
        logger.info(
            "Best Task1 threshold (F1) on val: %.3f (acc=%.4f, f1=%.4f, prec=%.4f, rec=%.4f)",
            best_f1_th["threshold"],
            best_f1_th["acc"],
            best_f1_th["f1"],
            best_f1_th["precision"],
            best_f1_th["recall"],
        )
        logger.info(
            "Best Task1 threshold (Acc) on val: %.3f (acc=%.4f, f1=%.4f, prec=%.4f, rec=%.4f)",
            best_acc_th["threshold"],
            best_acc_th["acc"],
            best_acc_th["f1"],
            best_acc_th["precision"],
            best_acc_th["recall"],
        )

    if val_labels_t2_best:
        unique_labels_val = sorted(list(set(val_labels_t2_best) | set(val_preds_t2_best)))
        report_dict = classification_report(
            val_labels_t2_best,
            val_preds_t2_best,
            labels=unique_labels_val,
            target_names=[id2label[i] for i in unique_labels_val],
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        with open(run_dir / "val_per_class_metrics.json", "w") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        logger.info("Saved val per-class metrics to %s", run_dir / "val_per_class_metrics.json")

    hier_stats = compute_hierarchy_stats(model, val_loader, device, use_amp=CFG.use_amp)
    with open(run_dir / "hierarchy_stats_val.json", "w") as f:
        json.dump(hier_stats, f, ensure_ascii=False, indent=2)
    logger.info("Hierarchy stats (val): %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in hier_stats.items()})

    def run_test_inference(csv_path: str, name: str, threshold: float, save_probs: bool = False):
        if not csv_path or not os.path.exists(csv_path):
            return
        logger.info("Running test inference (%s) using %s", name, csv_path)
        df_test_raw = pd.read_csv(csv_path)
        df_test = preprocess_dataframe(df_test_raw)
        test_ds = MultiTaskDataset(df_test, tokenizer, CFG.max_length, label2id)
        test_loader = DataLoader(
            test_ds,
            batch_size=CFG.per_device_eval_batch_size,
            shuffle=False,
            num_workers=CFG.dataloader_num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        model.eval()
        t1_preds, t1_trues, t2_preds, t2_trues = [], [], [], []
        pred_pick_all, pred_cat_all, prob_pick_all = [], [], []
        topk_hits = {1: 0, 2: 0, 3: 0}
        topk_total = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Test-{name}"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels_b = batch["labels_binary"].to(device, non_blocking=True)
                labels_c = batch["labels_category"].to(device, non_blocking=True)

                with autocast(enabled=CFG.use_amp):
                    logits_t1, logits_t2 = model(input_ids=input_ids, attention_mask=attention_mask)

                prob_b = torch.softmax(logits_t1, dim=-1)[:, 1]
                pred_b = (prob_b >= threshold).long()
                pred_pick_all.extend(pred_b.cpu().tolist())
                if save_probs:
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

        if t1_trues:
            acc_t1 = accuracy_score(t1_trues, t1_preds)
            logger.info("[%s] Test Task1 accuracy=%.4f (n=%d)", name, acc_t1, len(t1_trues))
            logger.info(
                "[%s] Test Task1 report:\n%s",
                name,
                classification_report(
                    t1_trues,
                    t1_preds,
                    target_names=[CFG.pick_map_rev[0], CFG.pick_map_rev[1]],
                    digits=4,
                    zero_division=0,
                ),
            )
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
                logger.info("[%s] Task2 top-k accuracy: k=1 %.4f, k=2 %.4f, k=3 %.4f", name, topk_acc[1], topk_acc[2], topk_acc[3])

        out_df = df_test_raw.copy()
        if len(pred_pick_all) == len(out_df):
            out_df["model_pick"] = [CFG.pick_map_rev.get(int(p), str(p)) for p in pred_pick_all]
        if len(pred_cat_all) == len(out_df):
            out_df["model_category"] = [id2label.get(int(p), str(p)) for p in pred_cat_all]
        if save_probs and len(prob_pick_all) == len(out_df):
            out_df["model_pick_prob"] = prob_pick_all
        pred_path = run_dir / f"test_predictions_{name}_thr{threshold:.2f}.csv"
        out_df.to_csv(pred_path, index=False)
        logger.info("Saved test predictions to %s", pred_path)

    run_test_inference(CFG.test_csv, "test1_f1", threshold=best_t1_threshold, save_probs=True)
    run_test_inference(CFG.test_csv_v5, "test_v5_f1", threshold=best_t1_threshold, save_probs=True)
    if best_t1_threshold_acc != best_t1_threshold:
        run_test_inference(CFG.test_csv, "test1_acc", threshold=best_t1_threshold_acc, save_probs=False)
        run_test_inference(CFG.test_csv_v5, "test_v5_acc", threshold=best_t1_threshold_acc, save_probs=False)

    # Secondary test (Riskdog subsets) following deberta_finetune_mps.py behavior
    if CFG.test2_csv and os.path.exists(CFG.test2_csv):
        logger.info("Running secondary test (Riskdog subsets) using %s", CFG.test2_csv)
        df_test2_raw = pd.read_csv(CFG.test2_csv)
        df_test2 = preprocess_dataframe(df_test2_raw)
        test2_ds = MultiTaskDataset(df_test2, tokenizer, CFG.max_length, label2id)
        test2_loader = DataLoader(
            test2_ds,
            batch_size=CFG.per_device_eval_batch_size,
            shuffle=False,
            num_workers=CFG.dataloader_num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        model.eval()
        pred_pick_all2, pred_cat_all2 = [], []
        with torch.no_grad():
            for batch in tqdm(test2_loader, desc="Test2"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with autocast(enabled=CFG.use_amp):
                    logits_t1, logits_t2 = model(input_ids=input_ids, attention_mask=attention_mask)
                prob_b = torch.softmax(logits_t1, dim=-1)[:, 1]
                pred_b = (prob_b >= best_t1_threshold).long()
                pred_pick_all2.extend(pred_b.cpu().tolist())
                pred_c = torch.argmax(logits_t2, dim=1)
                pred_cat_all2.extend(pred_c.cpu().tolist())

        pred_pick_all2 = np.array(pred_pick_all2)
        pred_cat_all2 = np.array(pred_cat_all2)

        def _encode_pick_label(val: str) -> Optional[int]:
            s = str(val).strip()
            if s in CFG.pick_map:
                return CFG.pick_map[s]
            if s in {"1", "1.0"}:
                return 1
            if s in {"0", "0.0"}:
                return 0
            return None

        # Task1 subset: test_purpose contains "pick（Riskdog）"
        if "test_purpose" in df_test2_raw.columns and "pick" in df_test2_raw.columns:
            tp = df_test2_raw["test_purpose"].astype(str)
            mask_t1 = tp.str.contains("pick（Riskdog）", na=False)
            mask_t1 &= df_test2_raw["pick"].notna()
            mask_t1 &= df_test2_raw["pick"].astype(str).str.strip() != ""
            idx_t1 = np.where(mask_t1.values)[0]
            true_pick_labels, pred_pick_labels = [], []
            for i in idx_t1:
                enc = _encode_pick_label(df_test2_raw.iloc[i]["pick"])
                if enc is None:
                    continue
                true_pick_labels.append(enc)
                pred_pick_labels.append(int(pred_pick_all2[i]))
            if true_pick_labels:
                acc_t1_rd = accuracy_score(true_pick_labels, pred_pick_labels)
                logger.info("Secondary test (Riskdog) Task1 accuracy=%.4f (n=%d)", acc_t1_rd, len(true_pick_labels))

        # Task2 subset: test_purpose contains "カテゴリ（Riskdog）"
        if "test_purpose" in df_test2_raw.columns and "category" in df_test2_raw.columns:
            tp = df_test2_raw["test_purpose"].astype(str)
            mask_t2 = tp.str.contains("カテゴリ（Riskdog）", na=False)
            mask_t2 &= df_test2_raw["category"].notna()
            mask_t2 &= df_test2_raw["category"].astype(str).str.strip() != ""
            idx_t2 = np.where(mask_t2.values)[0]
            true_cat_ids, pred_cat_ids = [], []
            for i in idx_t2:
                cat_str = str(df_test2_raw.iloc[i]["category"]).strip()
                true_id = label2id.get(cat_str)
                if true_id is None:
                    continue
                true_cat_ids.append(int(true_id))
                pred_cat_ids.append(int(pred_cat_all2[i]))
            if true_cat_ids:
                acc_t2_rd = accuracy_score(true_cat_ids, pred_cat_ids)
                logger.info("Secondary test (Riskdog) Task2 accuracy=%.4f (n=%d)", acc_t2_rd, len(true_cat_ids))

        out_df2 = df_test2_raw.copy()
        if len(pred_pick_all2) == len(out_df2):
            out_df2["model_pick"] = [CFG.pick_map_rev.get(int(p), str(p)) for p in pred_pick_all2]
        if len(pred_cat_all2) == len(out_df2):
            out_df2["model_category"] = [id2label.get(int(p), str(p)) for p in pred_cat_all2]
        out_df2.to_csv(run_dir / "test2_predictions.csv", index=False)
        logger.info("Saved secondary test predictions to %s", run_dir / "test2_predictions.csv")


if __name__ == "__main__":
    main()
