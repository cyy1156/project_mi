# 特异度套件（20260801_220609 / eegnet_balbatch_balacc）

- 开始：`2026-08-01T22:06:09`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`eegnet` | 臂：`B1` — 普通CE + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegnet_balbatch_balacc\merged_2s\run_20260801_220609`

---
## 最终结论

### Task — B1
- Val BalAcc：`0.5831 ± 0.0271`
- Test Spec：`0.5286 ± 0.1491`
- Test Rec：`0.6215 ± 0.1539`
- Test BalAcc：`0.5751 ± 0.0516`
- Test F1：`0.6740 ± 0.1116`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5892`
- Val F1（最优 checkpoint 时，附报）：`0.6276`
- Val loss（最优时）：`0.6877`

**Test（overall）**
- Accuracy：`0.5761`
- Recall：`0.5939`
- Specificity：`0.5323`
- Precision：`0.7582`
- F1：`0.6661`
- Balanced Acc：`0.5631`
- 混淆矩阵：TP=`2998` TN=`1088` FP=`956` FN=`2050`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6641` F1=`0.7476` BalAcc=`0.6247`
- `stieger_only`：Acc=`0.5709` F1=`0.6610` BalAcc=`0.5598`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.5785`
- Val F1（最优 checkpoint 时，附报）：`0.7452`
- Val loss（最优时）：`0.6566`

**Test（overall）**
- Accuracy：`0.4503`
- Recall：`0.3317`
- Specificity：`0.7526`
- Precision：`0.7737`
- F1：`0.4644`
- Balanced Acc：`0.5422`
- 混淆矩阵：TP=`1446` TN=`1287` FP=`423` FN=`2913`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3518` F1=`0.0979` BalAcc=`0.5144`
- `stieger_only`：Acc=`0.4572` F1=`0.4820` BalAcc=`0.5422`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5335`
- Val F1（最优 checkpoint 时，附报）：`0.6302`
- Val loss（最优时）：`0.6902`

**Test（overall）**
- Accuracy：`0.7033`
- Recall：`0.7308`
- Specificity：`0.6224`
- Precision：`0.8505`
- F1：`0.7861`
- Balanced Acc：`0.6766`
- 混淆矩阵：TP=`3475` TN=`1007` FP=`611` FN=`1280`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7462` F1=`0.8319` BalAcc=`0.6472`
- `stieger_only`：Acc=`0.7005` F1=`0.7828` BalAcc=`0.6819`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`51`
- 验证最优轮次（best_epoch）：`33`
- Val 选模分数（Balanced Acc）：`0.6058`
- Val F1（最优 checkpoint 时，附报）：`0.6261`
- Val loss（最优时）：`0.6917`

**Test（overall）**
- Accuracy：`0.6087`
- Recall：`0.7231`
- Specificity：`0.3521`
- Precision：`0.7147`
- F1：`0.7189`
- Balanced Acc：`0.5376`
- 混淆矩阵：TP=`4099` TN=`889` FP=`1636` FN=`1570`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5978` F1=`0.6929` BalAcc=`0.5580`
- `stieger_only`：Acc=`0.6093` F1=`0.7200` BalAcc=`0.5365`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc）：`0.6084`
- Val F1（最优 checkpoint 时，附报）：`0.7037`
- Val loss（最优时）：`0.6741`

**Test（overall）**
- Accuracy：`0.6275`
- Recall：`0.7282`
- Specificity：`0.3837`
- Precision：`0.7410`
- F1：`0.7346`
- Balanced Acc：`0.5560`
- 混淆矩阵：TP=`4292` TN=`934` FP=`1500` FN=`1602`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4491` F1=`0.4390` BalAcc=`0.5375`
- `stieger_only`：Acc=`0.6312` F1=`0.7388` BalAcc=`0.5561`

- 结束：`2026-08-01T22:30:07`
