# 特异度套件（20260802_060307 / dgcnn_balbatch_balacc）

- 开始：`2026-08-02T06:03:07`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`dgcnn` | 臂：`B1` — 普通CE + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dgcnn_balbatch_balacc\merged_2s\run_20260802_060307`

---
## 最终结论

### Task — B1
- Val BalAcc：`0.5504 ± 0.0165`
- Test Spec：`0.5105 ± 0.2238`
- Test Rec：`0.5714 ± 0.2167`
- Test BalAcc：`0.5410 ± 0.0216`
- Test F1：`0.6228 ± 0.1338`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5752`
- Val F1（最优 checkpoint 时，附报）：`0.6723`
- Val loss（最优时）：`0.6846`

**Test（overall）**
- Accuracy：`0.4852`
- Recall：`0.4152`
- Specificity：`0.6580`
- Precision：`0.7499`
- F1：`0.5345`
- Balanced Acc：`0.5366`
- 混淆矩阵：TP=`2096` TN=`1345` FP=`699` FN=`2952`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6465` F1=`0.7318` BalAcc=`0.6096`
- `stieger_only`：Acc=`0.4757` F1=`0.5204` BalAcc=`0.5334`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.5338`
- Val F1（最优 checkpoint 时，附报）：`0.5823`
- Val loss（最优时）：`0.6999`

**Test（overall）**
- Accuracy：`0.4261`
- Recall：`0.2861`
- Specificity：`0.7830`
- Precision：`0.7707`
- F1：`0.4173`
- Balanced Acc：`0.5346`
- 混淆矩阵：TP=`1247` TN=`1339` FP=`371` FN=`3112`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3367` F1=`0.0571` BalAcc=`0.5032`
- `stieger_only`：Acc=`0.4324` F1=`0.4350` BalAcc=`0.5351`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.5426`
- Val F1（最优 checkpoint 时，附报）：`0.5845`
- Val loss（最优时）：`0.6954`

**Test（overall）**
- Accuracy：`0.6148`
- Recall：`0.6477`
- Specificity：`0.5179`
- Precision：`0.7979`
- F1：`0.7150`
- Balanced Acc：`0.5828`
- 混淆矩阵：TP=`3080` TN=`838` FP=`780` FN=`1675`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5846` F1=`0.6284` BalAcc=`0.6169`
- `stieger_only`：Acc=`0.6167` F1=`0.7196` BalAcc=`0.5782`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5359`
- Val F1（最优 checkpoint 时，附报）：`0.6483`
- Val loss（最优时）：`0.6969`

**Test（overall）**
- Accuracy：`0.6746`
- Recall：`0.9220`
- Specificity：`0.1192`
- Precision：`0.7015`
- F1：`0.7968`
- Balanced Acc：`0.5206`
- 混淆矩阵：TP=`5227` TN=`301` FP=`2224` FN=`442`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6005` F1=`0.6697` BalAcc=`0.6017`
- `stieger_only`：Acc=`0.6781` F1=`0.8013` BalAcc=`0.5160`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.5645`
- Val F1（最优 checkpoint 时，附报）：`0.7519`
- Val loss（最优时）：`0.6779`

**Test（overall）**
- Accuracy：`0.5536`
- Recall：`0.5862`
- Specificity：`0.4745`
- Precision：`0.7298`
- F1：`0.6502`
- Balanced Acc：`0.5304`
- 混淆矩阵：TP=`3455` TN=`1155` FP=`1279` FN=`2439`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3174` F1=`0.0339` BalAcc=`0.5086`
- `stieger_only`：Acc=`0.5584` F1=`0.6571` BalAcc=`0.5304`

- 结束：`2026-08-02T06:08:30`
