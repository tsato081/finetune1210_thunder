bert_thunder/deberta_finetune_mps.py について、様々な技術を盛り込みすぎた結果よくわからなくなったので一旦素の状態に戻してみる。

Stage1：Pick 大規模（150k+）で 2epoch だけ回して encoder の「Pick/Declineの感覚」をざっくり入れる（済み。bert_thunder/deberta_pick_pretrain_mps）

Stage2：  
入力：Stage1 の checkpoint  
出力：Task1（Pick/Decline）＋Task2（96カテゴリ）を同時学習  
5epoch 固定で回す  
正則化は「最低限だけ」に絞る  
「Task2 を 85% に持って行く」が最重要。Task1 はその補助。

---

## 1-2. 素の構成：モデル・損失・スケジュール

### モデル構成
- 共通 encoder：Stage1 で作った DeBERTa
- ヘッド：  
  - Task1：2クラス用 linear（Pick / Decline）  
  - Task2：96クラス用 linear
- Pooling：まずは CLS か mean pooling どちらかに固定（今は pooling オプションも弄っているので）  
  個人的には mean pooling の方が安定しやすいことが多いので、「素の構成」では mean に寄せておくのが無難。

### 損失（最初は超シンプル）
- Task1：普通の CrossEntropyLoss  
  - クラス重み：今の逆数ベース（Pick/Decline のバランス調整）はそのままでOK  
  - Label smoothing：いったん 0.0（境界をシャープにしたい）
- Task2：CrossEntropyLoss  
  - クラス重み：逆数ベースは使いつつも、「極端なレアクラスの重みはクリップする」くらいが良さそう  
    例えば「最小重みは 0.5〜0.7 程度で打ち切る」設計イメージ（実装は後で考えるとして、概念的には「極端に重くしない」）  
  - Label smoothing：0.1〜0.15 を軽く入れる（96クラスで、近い概念同士が多いので、過剰な overconfidence を抑える意味で）

### タスクの重み付け
- 合計 loss を L = λ1 Ltask1 + λ2 Ltask2 で合成するだけにする（不確実性重み付けは封印）  
  最初のデフォルト案：λ1 = 0.3∼0.5 / λ2 = 1.0  
  これで Task2 を「本命」として強く最適化しつつ Task1 で「Pick/Declineの感覚」はそれなりに維持する、というバランスになる。

### 使わないテク（素の構成では一旦封印）
- R-Drop：OFF  
- multi-sample dropout：OFF（msd_samples=1 のイメージ）  
- AWP：OFF  
- 不確実性重み付け（loss_log_vars）：OFF

### Training スケジュール
- エポック数：5 固定  
- LLRD：そのまま採用（layer-wise lr decay は割と無難に効く）  
  - decay 係数は 0.8 でも 0.9 でも大きな差にはなりにくいので、ひとまず現状の 0.8 で良い
- 学習率：2e-5 くらい（今の設定を踏襲）  
- warmup：0.06  
- バッチサイズ：32〜64（MPS のメモリと相談）  
- gradient_accumulation：1（MPS だし、ここも無理に弄らない）

---

## 2. Stage2 に「階層マルチタスク」をどう入れるか
### 2-1. 前提：Task1 と Task2 の関係
- Task1：「Task2の96カテゴリのどれかに該当するかどうか」  
  要するに「ニュースとしてカテゴリ付与対象か / そうでないか」の2値判定
- Task2：ニュースを 96カテゴリに分類
- つまり本質的には、「96カテゴリ ＋ その他(Decline)」の 97クラス分類。Task1 は「その 97クラスを [その他] vs [その他以外] でざっくり2値化したもの」とも言い換えられる。

### 2-2. 案A：Loss レベルで階層性を持たせる（構造はそのまま）
- モデルは今と同じく：Encoder → Binary head（Task1） / Encoder → 96-class head（Task2）
- 損失の設計だけ以下のようにする：  
  - Pick ラベル付き & カテゴリ付きのサンプル：Task1 binary CE / Task2 CE（96クラス）  
  - Decline のみラベルのサンプル：Task1 binary CE のみ（Task2 はラベルがない & そもそもカテゴリも定義されていないので学習しない）  
  - Task2 カテゴリのみラベルのサンプル（Pick/Decline 無し）：Task2 の CE のみ使う  
  ※ すでに labels_binary = -100 / labels_category = -100 を使ってマスクしているので、その方針を明確に「階層っぽいデータ利用」と認識して整理するイメージ。

#### 階層らしさを足すための追加項目
- 整合性 Regularizer（任意）  
  Pick とラベリングされているサンプルでは、Task1 の「Pick 確率」が高い & Task2 の「正しいカテゴリの確率」も高い、という2つが同時に満たされてほしい。  
  例えば：Pick サンプルについてだけ、「Task1 の Pick ログit」と「Task2 の正解カテゴリ logits」を近づけるみたいな項を loss に小さく足す（L2 とか、KL とか）。これは「Pick の確信度が高いときは、カテゴリもちゃんと一貫してほしい」というゆるい階層制約を入れるイメージ。  
  逆に Decline サンプルでは、Task2 の出力分布が あまり尖らずフラット気味 の方が自然なので、Decline サンプルに対して Task2 の KL を「一様分布に近づける」ような regularizer を入れる案もある（オプション）。  
  まずは「構造変えずに、loss の設計だけで階層らしさを足す」パターンからやるのが、リスク少なくてよさそう。

---

## 3. ログに出すと“後で効いてくる”情報

今のログ：各 epoch ごとの val accuracy（Task1/Task2）、最後に test accuracy（Task1/Task2）、一部 cartography / worst-K 出力。精度頭打ちの原因を切るには、もう少し “どこで負けているか”が分かるログが欲しい。

### 3-1. 各 epoch ごとに見たいもの（テキストログ）
- (A) 損失系：train_loss_task1, train_loss_task2 / val_loss_task1, val_loss_task2 / （階層ロスを入れた場合）loss_hier_regularizer  
  ※「両方のタスクを同時に最適化しているつもりが、実は片方の loss しか下がっていない」というのがよくあるので最低限ほしい。
- (B) メトリクス系：val_acc_task1, val_acc_task2（既にある）  
  - Task2：val_macro_f1_task2、可能なら val_weighted_f1_task2 も  
  - Task1：precision, recall, f1（Pickクラスに対して）
- (C) タスクの寄与：手動タスク重みを使う場合は lambda_task1, lambda_task2（定数でもログに出しておくと、あとで見返しやすい）  
  不確実性重み付けに戻す気があるなら：各 epoch ごとに sigma_task1, sigma_task2 をログに出す（今も custom log に入れているが、epoch単位の summaryにも出しておく）
- (D) 学習の安定性：すでに出している grad_norm を epoch summary にも載せる（急に爆上がり・爆下がりしてるepochがないかを見る用）

### 3-2. 学習終了時に一度だけ出すと嬉しい統計
- (E) Task2 のクラス別メトリクス：per_class_support（クラスごとの件数）、per_class_accuracy or recall、可能なら「頻度帯ごと」の集約も
- 階層整合性の統計：  
  「Task1=Pick なのに、Task2 の max-prob が低いサンプルの割合」  
  「Task1=Decline なのに、Task2 の max-prob がめちゃ高いサンプルの割合」  
  → 階層マルチタスクの「噛み合い」を見る指標。

---

## 見たい学習曲線（プロット）

最低限これがあると判断が楽。

### 4-1. まず絶対欲しい 4枚
- Task1：train loss / val loss vs epoch
- Task1：train accuracy / val accuracy vs epoch
- Task2：train loss / val loss vs epoch
- Task2：train accuracy / val accuracy vs epoch（＋ macro F1）

これで、underfitting なのか / 過学習なのか / そもそも learning rate が高すぎ/低すぎなのか、がかなり見えてくる。

### 4-2. Task2 のクラス頻度別カーブ
- クラスを頻度帯で 3〜4 グループに分けておいて、High / Mid / Low ごとに Accuracy または Recall の平均を epoch ごとに記録し、プロット  
  → 「Low freq だけが全く改善していない」のか、「全部そこそこ伸びているが 0.8 を超えない」のか、など対策の方向性が分かる。
