# 特异度套件（20260802_033850 / conformer_wce2_balbatch_balacc）

- 开始：`2026-08-02T03:38:50`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`conformer` | 臂：`B2` — w0=2 + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\conformer_wce2_balbatch_balacc\merged_2s\run_20260802_033850`

---
## 最终结论

### Task — B2
- Val BalAcc：`0.5704 ± 0.0097`
- Test Spec：`0.5860 ± 0.2192`
- Test Rec：`0.5358 ± 0.2420`
- Test BalAcc：`0.5609 ± 0.0542`
- Test F1：`0.5940 ± 0.1880`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc）：`0.5641`
- Val F1（最优 checkpoint 时，附报）：`0.3547`
- Val loss（最优时）：`0.8116`

**Test（overall）**
- Accuracy：`0.3879`
- Recall：`0.2044`
- Specificity：`0.8410`
- Precision：`0.7605`
- F1：`0.3222`
- Balanced Acc：`0.5227`
- 混淆矩阵：TP=`1032` TN=`1719` FP=`325` FN=`4016`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4747` F1=`0.4555` BalAcc=`0.5544`
- `stieger_only`：Acc=`0.3828` F1=`0.3138` BalAcc=`0.5213`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5674`
- Val F1（最优 checkpoint 时，附报）：`0.7111`
- Val loss（最优时）：`0.7011`

**Test（overall）**
- Accuracy：`0.4217`
- Recall：`0.2810`
- Specificity：`0.7801`
- Precision：`0.7651`
- F1：`0.4111`
- Balanced Acc：`0.5306`
- 混淆矩阵：TP=`1225` TN=`1334` FP=`376` FN=`3134`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3693` F1=`0.1375` BalAcc=`0.5294`
- `stieger_only`：Acc=`0.4253` F1=`0.4251` BalAcc=`0.5290`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val 选模分数（Balanced Acc）：`0.5580`
- Val F1（最优 checkpoint 时，附报）：`0.5260`
- Val loss（最优时）：`0.7657`

**Test（overall）**
- Accuracy：`0.6758`
- Recall：`0.6837`
- Specificity：`0.6527`
- Precision：`0.8526`
- F1：`0.7589`
- Balanced Acc：`0.6682`
- 混淆矩阵：TP=`3251` TN=`1056` FP=`562` FN=`1504`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7692` F1=`0.8404` BalAcc=`0.6984`
- `stieger_only`：Acc=`0.6697` F1=`0.7531` BalAcc=`0.6686`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5776`
- Val F1（最优 checkpoint 时，附报）：`0.6166`
- Val loss（最优时）：`0.7302`

**Test（overall）**
- Accuracy：`0.6162`
- Recall：`0.7428`
- Specificity：`0.3319`
- Precision：`0.7140`
- F1：`0.7281`
- Balanced Acc：`0.5373`
- 混淆矩阵：TP=`4211` TN=`838` FP=`1687` FN=`1458`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5326` F1=`0.5905` BalAcc=`0.5515`
- `stieger_only`：Acc=`0.6201` F1=`0.7333` BalAcc=`0.5362`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.5851`
- Val F1（最优 checkpoint 时，附报）：`0.7004`
- Val loss（最优时）：`0.6719`

**Test（overall）**
- Accuracy：`0.6376`
- Recall：`0.7671`
- Specificity：`0.3242`
- Precision：`0.7332`
- F1：`0.7498`
- Balanced Acc：`0.5456`
- 混淆矩阵：TP=`4521` TN=`789` FP=`1645` FN=`1373`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3952` F1=`0.2937` BalAcc=`0.5317`
- `stieger_only`：Acc=`0.6426` F1=`0.7552` BalAcc=`0.5455`

- 结束：`2026-08-02T03:56:27`
