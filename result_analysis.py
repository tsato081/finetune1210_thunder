"""
Per-category performance analysis for finetune_1210 outputs.

Usage:
  - Set RUN_DIR to the specific run folder (e.g., "bert_thunder/finetune_1210/output/output_20251210_175748")
  - Run: uv run bert_thunder/finetune_1210/result_analysis.py

What it does:
  - Load val_per_class_metrics.json (per-class precision/recall/f1/support)
  - Load test_predictions_*.csv and test2_predictions.csv if present
  - Compute per-category support and F1, sort, and highlight:
      * "変態クラス": supportが少ないのにF1が高い
      * "要注意クラス": supportがそこそこあるのにF1が低い
  - Prints top/bottom lists to stdout. Adjust thresholds as needed.
"""

from pathlib import Path
import pandas as pd

# ==== Config: point to a specific run directory ====
RUN_DIR = Path("bert_thunder/finetune_1210/output/output_20251210_175748")

# Thresholds for flagging classes
LOW_SUPPORT_MAX = 5           # support <= 5 → small class
HIGH_F1_MIN = 0.7            # F1 >= 0.7 for "変態クラス"
MID_SUPPORT_MIN = 10         # support >= 10 → not too tiny
LOW_F1_MAX = 0.5             # F1 <= 0.5 for "要注意クラス"


def load_val_per_class(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "val_per_class_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run training first.")
    raw = pd.read_json(path, typ="series")
    rows = []
    for label, metrics in raw.items():
        if not isinstance(metrics, dict):
            continue
        if "support" not in metrics:
            continue
        rows.append(
            {
                "label": label,
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1-score": metrics.get("f1-score", 0.0),
                "support": metrics.get("support", 0),
            }
        )
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].notna() & (df["label"] != "") & (df["label"] != "-1")]
    df = df[df["support"] > 0]
    return df


def extract_test_per_class(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["label", "precision", "recall", "f1-score", "support"])
    df = pd.read_csv(csv_path)
    if "category" not in df.columns or "model_category" not in df.columns:
        return pd.DataFrame(columns=["label", "precision", "recall", "f1-score", "support"])
    true = df["category"].astype(str)
    pred = df["model_category"].astype(str)

    def _clean_label(x: str):
        s = str(x).strip()
        if s == "" or s == "-1" or s.lower() == "nan":
            return None
        return s

    true = true.apply(_clean_label)
    pred = pred.apply(_clean_label)
    mask = true.notna() | pred.notna()
    true = true[mask]
    pred = pred[mask]
    # compute per-class metrics
    labels = sorted(set(true.dropna().unique()) | set(pred.dropna().unique()))
    rows = []
    for label in labels:
        mask_true = true == label
        mask_pred = pred == label
        tp = ((mask_true) & (mask_pred)).sum()
        fp = ((~mask_true) & (mask_pred)).sum()
        fn = ((mask_true) & (~mask_pred)).sum()
        support = mask_true.sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": support,
            }
        )
    return pd.DataFrame(rows)


def extract_true_pred(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["true", "pred"])
    df = pd.read_csv(csv_path)
    if "category" not in df.columns or "model_category" not in df.columns:
        return pd.DataFrame(columns=["true", "pred"])

    def _clean_label(x: str):
        s = str(x).strip()
        if s == "" or s == "-1" or s.lower() == "nan":
            return None
        return s

    tpairs = pd.DataFrame(
        {
            "true": df["category"].astype(str).apply(_clean_label),
            "pred": df["model_category"].astype(str).apply(_clean_label),
        }
    )
    tpairs = tpairs[(tpairs["true"].notna()) | (tpairs["pred"].notna())]
    return tpairs


def top_misclassifications(tpairs: pd.DataFrame, focus_labels, topn=3):
    rows = []
    for label in focus_labels:
        sub = tpairs[tpairs["true"] == label]
        if sub.empty:
            continue
        counts = sub["pred"].value_counts(dropna=False)
        # exclude correct predictions from "mislabels"
        if label in counts.index:
            counts = counts.drop(label)
        topk = counts.head(topn)
        rows.append(
            {
                "label": label,
                "support": int(len(sub)),
                "top_mislabels": "; ".join([f"{k if k is not None else 'None'}({v})" for k, v in topk.items()]),
            }
        )
    return pd.DataFrame(rows)


def flag_classes(df: pd.DataFrame):
    df = df.copy()
    df["support"] = df["support"].astype(int)
    df["f1"] = df["f1-score"]

    hen_tai = df[(df["support"] <= LOW_SUPPORT_MAX) & (df["f1"] >= HIGH_F1_MIN)].sort_values(
        ["f1", "support"], ascending=[False, True]
    )
    you_chui = df[(df["support"] >= MID_SUPPORT_MIN) & (df["f1"] <= LOW_F1_MAX)].sort_values(
        ["f1", "support"], ascending=[True, False]
    )
    return hen_tai, you_chui


def main():
    print(f"Run dir: {RUN_DIR}")
    # Val per-class metrics
    val_df = load_val_per_class(RUN_DIR)
    hen, bad = flag_classes(val_df)
    print("\n[Val] 変態クラス（support少ないのにF1高い）")
    print(hen[["label", "support", "f1-score"]].head(20).to_string(index=False))
    print("\n[Val] 要注意クラス（supportそこそこなのにF1低い）")
    print(bad[["label", "support", "f1-score"]].head(20).to_string(index=False))

    # Test sets
    test_files = sorted(RUN_DIR.glob("test_predictions_*.csv"))
    for csv_path in test_files:
        name = csv_path.stem.replace("test_predictions_", "")
        test_df = extract_test_per_class(csv_path)
        tpairs = extract_true_pred(csv_path)
        if test_df.empty:
            continue
        hen_t, bad_t = flag_classes(test_df)
        print(f"\n[Test {name}] 変態クラス")
        print(hen_t[["label", "support", "f1-score"]].head(20).to_string(index=False))
        print(f"\n[Test {name}] 要注意クラス")
        print(bad_t[["label", "support", "f1-score"]].head(20).to_string(index=False))

        if not bad_t.empty and not tpairs.empty:
            mis_top = top_misclassifications(tpairs, bad_t["label"].tolist(), topn=3)
            if not mis_top.empty:
                print(f"\n[Test {name}] 要注意クラスの誤分類先トップ3")
                print(mis_top.to_string(index=False))

    # Secondary test (Riskdog)
    test2_csv = RUN_DIR / "test2_predictions.csv"
    if test2_csv.exists():
        test2_df = extract_test_per_class(test2_csv)
        tpairs2 = extract_true_pred(test2_csv)
        hen_t2, bad_t2 = flag_classes(test2_df)
        print("\n[Test2 Riskdog] 変態クラス")
        print(hen_t2[["label", "support", "f1-score"]].head(20).to_string(index=False))
        print("\n[Test2 Riskdog] 要注意クラス")
        print(bad_t2[["label", "support", "f1-score"]].head(20).to_string(index=False))
        if not bad_t2.empty and not tpairs2.empty:
            mis_top2 = top_misclassifications(tpairs2, bad_t2["label"].tolist(), topn=3)
            if not mis_top2.empty:
                print("\n[Test2 Riskdog] 要注意クラスの誤分類先トップ3")
                print(mis_top2.to_string(index=False))


if __name__ == "__main__":
    main()
