# 特异度套件（20260802_002739 / deep_balbatch_balacc）

- 开始：`2026-08-02T00:27:39`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`deep` | 臂：`B1` — 普通CE + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\deep_balbatch_balacc\merged_2s\run_20260802_002739`

---
## 最终结论

### Task — B1
- Val BalAcc：`0.5133 ± 0.0264`
- Test Spec：`0.9583 ± 0.0827`
- Test Rec：`0.0541 ± 0.1076`
- Test BalAcc：`0.5062 ± 0.0125`
- Test F1：`0.0803 ± 0.1593`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.9464`

**Test（overall）**
- Accuracy：`0.2882`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`2044` FP=`0` FN=`5048`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3258` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2860` F1=`0.0000` BalAcc=`0.5000`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`164`
- 验证最优轮次（best_epoch）：`146`
- Val 选模分数（Balanced Acc）：`0.5661`
- Val F1（最优 checkpoint 时，附报）：`0.6877`
- Val loss（最优时）：`0.6881`

**Test（overall）**
- Accuracy：`0.4169`
- Recall：`0.2693`
- Specificity：`0.7930`
- Precision：`0.7683`
- F1：`0.3988`
- Balanced Acc：`0.5312`
- 混淆矩阵：TP=`1174` TN=`1356` FP=`354` FN=`3185`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3869` F1=`0.1974` BalAcc=`0.5364`
- `stieger_only`：Acc=`0.4190` F1=`0.4098` BalAcc=`0.5295`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5003`
- Val F1（最优 checkpoint 时，附报）：`0.0011`
- Val loss（最优时）：`0.8382`

**Test（overall）**
- Accuracy：`0.2545`
- Recall：`0.0013`
- Specificity：`0.9988`
- Precision：`0.7500`
- F1：`0.0025`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`6` TN=`1616` FP=`2` FN=`4749`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3282` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2497` F1=`0.0027` BalAcc=`0.5000`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.9959`

**Test（overall）**
- Accuracy：`0.3082`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`2525` FP=`0` FN=`5669`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3234` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.3074` F1=`0.0000` BalAcc=`0.5000`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`1.0758`

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

- 结束：`2026-08-02T00:41:50`
