# 特异度套件（20260801_210911 / eegnet_wce22_balacc）

- 开始：`2026-08-01T21:09:11`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`eegnet` | 臂：`A22` — 加权CE w0=2.2 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegnet_wce22_balacc\merged_2s\run_20260801_210911`

---
## 最终结论

### Task — A22
- Val BalAcc：`0.5884 ± 0.0237`
- Test Spec：`0.4389 ± 0.1874`
- Test Rec：`0.6972 ± 0.1734`
- Test BalAcc：`0.5680 ± 0.0414`
- Test F1：`0.7111 ± 0.1104`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.6011`
- Val F1（最优 checkpoint 时，附报）：`0.7480`
- Val loss（最优时）：`0.6786`

**Test（overall）**
- Accuracy：`0.6506`
- Recall：`0.7763`
- Specificity：`0.3400`
- Precision：`0.7439`
- F1：`0.7598`
- Balanced Acc：`0.5582`
- 混淆矩阵：TP=`3919` TN=`695` FP=`1349` FN=`1129`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6869` F1=`0.7947` BalAcc=`0.5735`
- `stieger_only`：Acc=`0.6484` F1=`0.7576` BalAcc=`0.5579`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`43`
- 验证最优轮次（best_epoch）：`25`
- Val 选模分数（Balanced Acc）：`0.5713`
- Val F1（最优 checkpoint 时，附报）：`0.7514`
- Val loss（最优时）：`0.6731`

**Test（overall）**
- Accuracy：`0.4661`
- Recall：`0.3590`
- Specificity：`0.7392`
- Precision：`0.7782`
- F1：`0.4914`
- Balanced Acc：`0.5491`
- 混淆矩阵：TP=`1565` TN=`1264` FP=`446` FN=`2794`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3618` F1=`0.1301` BalAcc=`0.5198`
- `stieger_only`：Acc=`0.4735` F1=`0.5087` BalAcc=`0.5492`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.5508`
- Val F1（最优 checkpoint 时，附报）：`0.7010`
- Val loss（最优时）：`0.6873`

**Test（overall）**
- Accuracy：`0.6925`
- Recall：`0.7373`
- Specificity：`0.5606`
- Precision：`0.8314`
- F1：`0.7815`
- Balanced Acc：`0.6489`
- 混淆矩阵：TP=`3506` TN=`907` FP=`711` FN=`1249`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7308` F1=`0.8235` BalAcc=`0.6238`
- `stieger_only`：Acc=`0.6900` F1=`0.7786` BalAcc=`0.6538`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.6149`
- Val F1（最优 checkpoint 时，附报）：`0.7184`
- Val loss（最优时）：`0.6567`

**Test（overall）**
- Accuracy：`0.6547`
- Recall：`0.8522`
- Specificity：`0.2115`
- Precision：`0.7082`
- F1：`0.7735`
- Balanced Acc：`0.5318`
- 混淆矩阵：TP=`4831` TN=`534` FP=`1991` FN=`838`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6549` F1=`0.7635` BalAcc=`0.5629`
- `stieger_only`：Acc=`0.6547` F1=`0.7740` BalAcc=`0.5302`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`57`
- 验证最优轮次（best_epoch）：`39`
- Val 选模分数（Balanced Acc）：`0.6040`
- Val F1（最优 checkpoint 时，附报）：`0.7368`
- Val loss（最优时）：`0.6674`

**Test（overall）**
- Accuracy：`0.6390`
- Recall：`0.7613`
- Specificity：`0.3431`
- Precision：`0.7373`
- F1：`0.7491`
- Balanced Acc：`0.5522`
- 混淆矩阵：TP=`4487` TN=`835` FP=`1599` FN=`1407`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4731` F1=`0.4943` BalAcc=`0.5383`
- `stieger_only`：Acc=`0.6424` F1=`0.7528` BalAcc=`0.5522`

- 结束：`2026-08-01T21:37:36`
