# finetune1210_thunder (DeBERTa multitask for Pick/Category, MPS friendly)

Minimal, self-contained version of the Stage2 pipeline (Stage1 encoder already学習済みを前提)。

## 構成
- `deberta_finetune_1210.py`: 学習・検証・テスト推論（Task1/Task2）、閾値探索、top-k精度、階層正則化、R-Drop対応。
- `result_analysis.py`: 出力フォルダを指定して per-class メトリクスや誤分類先トップ3を集計。
- `finetune_architecture.md`: 設計メモ。
- `fetch_assets.py`: Hugging Face から学習/テスト CSV をダウンロードする補助スクリプト（`.env` 必須）。
- `upload_to_hf.py`: data/train・data/test の CSV を Hugging Face Dataset にアップロード（空コミットを避ける）。

## 事前準備
- Python: 3.11+
- 依存: `requirements.txt` を参照（torch, transformers, pandas, scikit-learn, matplotlib, tqdm など）。
- モデル: `models/deberta_pick_pretrain_mps` に Stage1 エンコーダチェックポイントを配置。
- データ:
  - Task1: `data/train/task1_cleaned.csv`（`pick`, `title/body`）
  - Task2: `data/train/task2_cleaned.csv`（`category`, `title/body`）
  - テスト（任意）: `data/test/Hawks4.0正解データ.csv`, `Hawks ver 5.0 csv出力用.csv`, `Hawks_Revenge_test_2.csv`
- Hugging Face 経由でデータ取得する場合：
  - `.env.example` をコピーして `.env` を作成し、`HF_TOKEN` のみセット（デフォルトで data/train・data/test 配下に配置）
  - `uv run bert_thunder/finetune1210_thunder/fetch_assets.py`
- Hugging Face にデータをアップする場合：
  - `.env` に `HF_AUTH_TOKEN`（必要なら `HF_DATASET_REPO`）を設定
  - `uv run bert_thunder/finetune1210_thunder/upload_to_hf.py`（data/train・data/test 内のCSVのみアップ、空コミットなし）

## 使い方
学習・推論（デフォルト設定のまま run ごとに `output/output_<timestamp>/` へ成果物保存）:
```bash
uv run bert_thunder/finetune1210_thunder/deberta_finetune_1210.py
```
- エポック中に早期終了あり（Task2 macro F1 基準）。
- Valで Task1 しきい値を F1/Acc 両方で探索し、Test 推論に適用。
- 出力物: ログ、label_map、学習曲線、val_per_class_metrics.json、hierarchy_stats、テスト予測CSV（閾値種別別）。

分析（特定 run を指定して誤分類先トップ3などを表示）:
```bash
uv run bert_thunder/finetune1210_thunder/result_analysis.py
```
スクリプト冒頭の `RUN_DIR` を対象の `output/output_<timestamp>/` に変更してください。

## メモ
- Task2重視: λ2=1.0, λ1=0.4（調整可）。
- 階層正則化: Decline で Task2 分布をフラット化（`hierarchy_weight`）。
- R-Drop: Task2 主体（`rdrop_alpha_task2` 0.5）、Task1 は任意で小さく。
- Top-k: テスト時に Task2 の top-1/2/3 accuracy をログ出力。

## ディレクトリ管理
- `data/` と `output/` は `.gitignore` で中身を除外しつつ `.gitkeep` で構造を保持。
- 公開時はデータ/モデルを含めず、README＋fetch_assets.py で再取得できるようにする想定。
