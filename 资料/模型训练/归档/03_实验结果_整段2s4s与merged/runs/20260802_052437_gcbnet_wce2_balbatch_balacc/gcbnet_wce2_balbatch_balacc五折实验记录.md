# 特异度套件（20260802_052437 / gcbnet_wce2_balbatch_balacc）

- 开始：`2026-08-02T05:24:37`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`gcbnet` | 臂：`B2` — w0=2 + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\gcbnet_wce2_balbatch_balacc\merged_2s\run_20260802_052437`

---
## 最终结论

### Task — B2
- Val BalAcc：`0.5071 ± 0.0049`
- Test Spec：`0.9288 ± 0.0728`
- Test Rec：`0.0845 ± 0.0857`
- Test BalAcc：`0.5066 ± 0.0079`
- Test F1：`0.1375 ± 0.1337`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`55`
- 验证最优轮次（best_epoch）：`37`
- Val 选模分数（Balanced Acc）：`0.5033`
- Val F1（最优 checkpoint 时，附报）：`0.0582`
- Val loss（最优时）：`0.7923`

**Test（overall）**
- Accuracy：`0.2917`
- Recall：`0.0075`
- Specificity：`0.9936`
- Precision：`0.7451`
- F1：`0.0149`
- Balanced Acc：`0.5006`
- 混淆矩阵：TP=`38` TN=`2031` FP=`13` FN=`5010`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3333` F1=`0.0435` BalAcc=`0.4996`
- `stieger_only`：Acc=`0.2893` F1=`0.0133` BalAcc=`0.5007`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val 选模分数（Balanced Acc）：`0.5078`
- Val F1（最优 checkpoint 时，附报）：`0.2708`
- Val loss（最优时）：`0.7208`

**Test（overall）**
- Accuracy：`0.3009`
- Recall：`0.0420`
- Specificity：`0.9608`
- Precision：`0.7320`
- F1：`0.0794`
- Balanced Acc：`0.5014`
- 混淆矩阵：TP=`183` TN=`1643` FP=`67` FN=`4176`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3216` F1=`0.0000` BalAcc=`0.4961`
- `stieger_only`：Acc=`0.2994` F1=`0.0844` BalAcc=`0.5015`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.7436`

**Test（overall）**
- Accuracy：`0.2539`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`1618` FP=`0` FN=`4755`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3282` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2490` F1=`0.0000` BalAcc=`0.5000`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`36`
- 验证最优轮次（best_epoch）：`18`
- Val 选模分数（Balanced Acc）：`0.5140`
- Val F1（最优 checkpoint 时，附报）：`0.1208`
- Val loss（最优时）：`0.7933`

**Test（overall）**
- Accuracy：`0.3952`
- Recall：`0.2092`
- Specificity：`0.8127`
- Precision：`0.7149`
- F1：`0.3237`
- Balanced Acc：`0.5109`
- 混淆矩阵：TP=`1186` TN=`2052` FP=`473` FN=`4483`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3315` F1=`0.0538` BalAcc=`0.4972`
- `stieger_only`：Acc=`0.3982` F1=`0.3336` BalAcc=`0.5113`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`54`
- 验证最优轮次（best_epoch）：`36`
- Val 选模分数（Balanced Acc）：`0.5101`
- Val F1（最优 checkpoint 时，附报）：`0.1279`
- Val loss（最优时）：`0.7323`

**Test（overall）**
- Accuracy：`0.3721`
- Recall：`0.1637`
- Specificity：`0.8767`
- Precision：`0.7628`
- F1：`0.2696`
- Balanced Acc：`0.5202`
- 混淆矩阵：TP=`965` TN=`2134` FP=`300` FN=`4929`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3353` F1=`0.0826` BalAcc=`0.5216`
- `stieger_only`：Acc=`0.3729` F1=`0.2728` BalAcc=`0.5201`

- 结束：`2026-08-02T05:34:56`
