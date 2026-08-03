# 特异度套件（20260802_043136 / dbn_balbatch_balacc）

- 开始：`2026-08-02T04:31:36`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`dbn` | 臂：`B1` — 普通CE + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dbn_balbatch_balacc\merged_2s\run_20260802_043136`

---
## 最终结论

### Task — B1
- Val BalAcc：`0.5291 ± 0.0101`
- Test Spec：`0.4023 ± 0.1454`
- Test Rec：`0.6574 ± 0.1636`
- Test BalAcc：`0.5299 ± 0.0178`
- Test F1：`0.6830 ± 0.0937`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.5320`
- Val F1（最优 checkpoint 时，附报）：`0.7115`
- Val loss（最优时）：`0.6822`

**Test（overall）**
- Accuracy：`0.5206`
- Recall：`0.5103`
- Specificity：`0.5460`
- Precision：`0.7352`
- F1：`0.6024`
- Balanced Acc：`0.5281`
- 混淆矩阵：TP=`2576` TN=`1116` FP=`928` FN=`2472`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6389` F1=`0.7245` BalAcc=`0.6040`
- `stieger_only`：Acc=`0.5136` F1=`0.5945` BalAcc=`0.5242`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`51`
- 验证最优轮次（best_epoch）：`33`
- Val 选模分数（Balanced Acc）：`0.5213`
- Val F1（最优 checkpoint 时，附报）：`0.6926`
- Val loss（最优时）：`0.6775`

**Test（overall）**
- Accuracy：`0.4961`
- Recall：`0.4687`
- Specificity：`0.5661`
- Precision：`0.7336`
- F1：`0.5719`
- Balanced Acc：`0.5174`
- 混淆矩阵：TP=`2043` TN=`968` FP=`742` FN=`2316`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4121` F1=`0.3777` BalAcc=`0.4924`
- `stieger_only`：Acc=`0.5020` F1=`0.5827` BalAcc=`0.5178`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5210`
- Val F1（最优 checkpoint 时，附报）：`0.7964`
- Val loss（最优时）：`0.6801`

**Test（overall）**
- Accuracy：`0.7147`
- Recall：`0.8696`
- Specificity：`0.2596`
- Precision：`0.7754`
- F1：`0.8198`
- Balanced Acc：`0.5646`
- 混淆矩阵：TP=`4135` TN=`420` FP=`1198` FN=`620`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6974` F1=`0.8103` BalAcc=`0.5590`
- `stieger_only`：Acc=`0.7159` F1=`0.8204` BalAcc=`0.5663`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.5477`
- Val F1（最优 checkpoint 时，附报）：`0.5068`
- Val loss（最优时）：`0.7047`

**Test（overall）**
- Accuracy：`0.6379`
- Recall：`0.8285`
- Specificity：`0.2099`
- Precision：`0.7019`
- F1：`0.7600`
- Balanced Acc：`0.5192`
- 混淆矩阵：TP=`4697` TN=`530` FP=`1995` FN=`972`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5163` F1=`0.5316` BalAcc=`0.5768`
- `stieger_only`：Acc=`0.6436` F1=`0.7672` BalAcc=`0.5156`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val 选模分数（Balanced Acc）：`0.5234`
- Val F1（最优 checkpoint 时，附报）：`0.8182`
- Val loss（最优时）：`0.6592`

**Test（overall）**
- Accuracy：`0.5573`
- Recall：`0.6098`
- Specificity：`0.4302`
- Precision：`0.7215`
- F1：`0.6610`
- Balanced Acc：`0.5200`
- 混淆矩阵：TP=`3594` TN=`1047` FP=`1387` FN=`2300`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3114` F1=`0.0171` BalAcc=`0.5043`
- `stieger_only`：Acc=`0.5623` F1=`0.6680` BalAcc=`0.5199`

- 结束：`2026-08-02T04:35:14`
