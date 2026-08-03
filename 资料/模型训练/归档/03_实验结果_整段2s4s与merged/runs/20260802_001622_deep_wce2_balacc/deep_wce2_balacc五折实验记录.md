# 特异度套件（20260802_001622 / deep_wce2_balacc）

- 开始：`2026-08-02T00:16:22`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`deep` | 臂：`A20` — 加权CE w0=2.0 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\deep_wce2_balacc\merged_2s\run_20260802_001622`

---
## 最终结论

### Task — A20
- Val BalAcc：`0.5133 ± 0.0254`
- Test Spec：`0.9529 ± 0.0497`
- Test Rec：`0.0491 ± 0.0543`
- Test BalAcc：`0.5010 ± 0.0058`
- Test F1：`0.0855 ± 0.0925`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5007`
- Val F1（最优 checkpoint 时，附报）：`0.0409`
- Val loss（最优时）：`0.7474`

**Test（overall）**
- Accuracy：`0.2965`
- Recall：`0.0202`
- Specificity：`0.9790`
- Precision：`0.7034`
- F1：`0.0393`
- Balanced Acc：`0.4996`
- 混淆矩阵：TP=`102` TN=`2001` FP=`43` FN=`4946`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3207` F1=`0.0074` BalAcc=`0.4902`
- `stieger_only`：Acc=`0.2951` F1=`0.0410` BalAcc=`0.5001`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`112`
- 验证最优轮次（best_epoch）：`94`
- Val 选模分数（Balanced Acc）：`0.5641`
- Val F1（最优 checkpoint 时，附报）：`0.5968`
- Val loss（最优时）：`0.6954`

**Test（overall）**
- Accuracy：`0.3472`
- Recall：`0.1349`
- Specificity：`0.8883`
- Precision：`0.7548`
- F1：`0.2289`
- Balanced Acc：`0.5116`
- 混淆矩阵：TP=`588` TN=`1519` FP=`191` FN=`3771`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3643` F1=`0.1185` BalAcc=`0.5277`
- `stieger_only`：Acc=`0.3460` F1=`0.2354` BalAcc=`0.5097`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.8812`

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

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5019`
- Val F1（最优 checkpoint 时，附报）：`0.0216`
- Val loss（最优时）：`0.7513`

**Test（overall）**
- Accuracy：`0.3392`
- Recall：`0.0905`
- Specificity：`0.8974`
- Precision：`0.6645`
- F1：`0.1593`
- Balanced Acc：`0.4940`
- 混淆矩阵：TP=`513` TN=`2266` FP=`259` FN=`5156`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3261` F1=`0.0080` BalAcc=`0.5020`
- `stieger_only`：Acc=`0.3398` F1=`0.1654` BalAcc=`0.4934`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.9061`

**Test（overall）**
- Accuracy：`0.2923`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`2434` FP=`0` FN=`5894`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3054` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2920` F1=`0.0000` BalAcc=`0.5000`

- 结束：`2026-08-02T00:27:39`
