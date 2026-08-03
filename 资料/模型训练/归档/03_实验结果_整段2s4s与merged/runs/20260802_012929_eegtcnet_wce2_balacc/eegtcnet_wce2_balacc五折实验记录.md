# 特异度套件（20260802_012929 / eegtcnet_wce2_balacc）

- 开始：`2026-08-02T01:29:29`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`eegtcnet` | 臂：`A20` — 加权CE w0=2.0 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegtcnet_wce2_balacc\merged_2s\run_20260802_012929`

---
## 最终结论

### Task — A20
- Val BalAcc：`0.5795 ± 0.0284`
- Test Spec：`0.3456 ± 0.1780`
- Test Rec：`0.7772 ± 0.1538`
- Test BalAcc：`0.5614 ± 0.0383`
- Test F1：`0.7539 ± 0.0776`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`48`
- Val 选模分数（Balanced Acc）：`0.6219`
- Val F1（最优 checkpoint 时，附报）：`0.6034`
- Val loss（最优时）：`0.6716`

**Test（overall）**
- Accuracy：`0.6052`
- Recall：`0.6638`
- Specificity：`0.4604`
- Precision：`0.7524`
- F1：`0.7053`
- Balanced Acc：`0.5621`
- 混淆矩阵：TP=`3351` TN=`941` FP=`1103` FN=`1697`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6616` F1=`0.7341` BalAcc=`0.6449`
- `stieger_only`：Acc=`0.6019` F1=`0.7037` BalAcc=`0.5567`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`54`
- 验证最优轮次（best_epoch）：`36`
- Val 选模分数（Balanced Acc）：`0.5606`
- Val F1（最优 checkpoint 时，附报）：`0.7784`
- Val loss（最优时）：`0.6762`

**Test（overall）**
- Accuracy：`0.5497`
- Recall：`0.5288`
- Specificity：`0.6029`
- Precision：`0.7725`
- F1：`0.6278`
- Balanced Acc：`0.5659`
- 混淆矩阵：TP=`2305` TN=`1031` FP=`679` FN=`2054`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3970` F1=`0.2157` BalAcc=`0.5458`
- `stieger_only`：Acc=`0.5604` F1=`0.6457` BalAcc=`0.5643`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5383`
- Val F1（最优 checkpoint 时，附报）：`0.7543`
- Val loss（最优时）：`0.6860`

**Test（overall）**
- Accuracy：`0.7573`
- Recall：`0.8892`
- Specificity：`0.3696`
- Precision：`0.8056`
- F1：`0.8453`
- Balanced Acc：`0.6294`
- 混淆矩阵：TP=`4228` TN=`598` FP=`1020` FN=`527`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6949` F1=`0.8132` BalAcc=`0.5411`
- `stieger_only`：Acc=`0.7613` F1=`0.8475` BalAcc=`0.6383`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`57`
- 验证最优轮次（best_epoch）：`39`
- Val 选模分数（Balanced Acc）：`0.5925`
- Val F1（最优 checkpoint 时，附报）：`0.7765`
- Val loss（最优时）：`0.6514`

**Test（overall）**
- Accuracy：`0.6734`
- Recall：`0.9092`
- Specificity：`0.1442`
- Precision：`0.7046`
- F1：`0.7939`
- Balanced Acc：`0.5267`
- 混淆矩阵：TP=`5154` TN=`364` FP=`2161` FN=`515`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6141` F1=`0.7370` BalAcc=`0.5130`
- `stieger_only`：Acc=`0.6762` F1=`0.7964` BalAcc=`0.5271`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc）：`0.5840`
- Val F1（最优 checkpoint 时，附报）：`0.8203`
- Val loss（最优时）：`0.6502`

**Test（overall）**
- Accuracy：`0.6774`
- Recall：`0.8948`
- Specificity：`0.1508`
- Precision：`0.7184`
- F1：`0.7970`
- Balanced Acc：`0.5228`
- 混淆矩阵：TP=`5274` TN=`367` FP=`2067` FN=`620`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3413` F1=`0.2568` BalAcc=`0.4544`
- `stieger_only`：Acc=`0.6842` F1=`0.8031` BalAcc=`0.5238`

- 结束：`2026-08-02T01:50:59`
