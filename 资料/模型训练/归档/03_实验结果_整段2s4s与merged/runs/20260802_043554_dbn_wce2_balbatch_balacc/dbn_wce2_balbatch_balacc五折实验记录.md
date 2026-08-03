# 特异度套件（20260802_043554 / dbn_wce2_balbatch_balacc）

- 开始：`2026-08-02T04:35:54`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`dbn` | 臂：`B2` — w0=2 + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dbn_wce2_balbatch_balacc\merged_2s\run_20260802_043554`

---
## 最终结论

### Task — B2
- Val BalAcc：`0.5029 ± 0.0017`
- Test Spec：`0.9451 ± 0.0726`
- Test Rec：`0.0601 ± 0.0750`
- Test BalAcc：`0.5026 ± 0.0052`
- Test F1：`0.0989 ± 0.1213`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val 选模分数（Balanced Acc）：`0.5045`
- Val F1（最优 checkpoint 时，附报）：`0.0211`
- Val loss（最优时）：`0.7707`

**Test（overall）**
- Accuracy：`0.2886`
- Recall：`0.0008`
- Specificity：`0.9995`
- Precision：`0.8000`
- F1：`0.0016`
- Balanced Acc：`0.5002`
- 混淆矩阵：TP=`4` TN=`2043` FP=`1` FN=`5044`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3258` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2864` F1=`0.0017` BalAcc=`0.5002`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`73`
- 验证最优轮次（best_epoch）：`55`
- Val 选模分数（Balanced Acc）：`0.5047`
- Val F1（最优 checkpoint 时，附报）：`0.0439`
- Val loss（最优时）：`0.7647`

**Test（overall）**
- Accuracy：`0.2834`
- Recall：`0.0030`
- Specificity：`0.9982`
- Precision：`0.8125`
- F1：`0.0059`
- Balanced Acc：`0.5006`
- 混淆矩阵：TP=`13` TN=`1707` FP=`3` FN=`4346`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3241` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2806` F1=`0.0063` BalAcc=`0.5006`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.7326`

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

- 早停/结束轮次（stopped_epoch）：`93`
- 验证最优轮次（best_epoch）：`75`
- Val 选模分数（Balanced Acc）：`0.5033`
- Val F1（最优 checkpoint 时，附报）：`0.0934`
- Val loss（最优时）：`0.7611`

**Test（overall）**
- Accuracy：`0.3773`
- Recall：`0.1815`
- Specificity：`0.8170`
- Precision：`0.6901`
- F1：`0.2874`
- Balanced Acc：`0.4993`
- 混淆矩阵：TP=`1029` TN=`2063` FP=`462` FN=`4640`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3315` F1=`0.0821` BalAcc=`0.4885`
- `stieger_only`：Acc=`0.3795` F1=`0.2954` BalAcc=`0.4996`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`54`
- 验证最优轮次（best_epoch）：`36`
- Val 选模分数（Balanced Acc）：`0.5021`
- Val F1（最优 checkpoint 时，附报）：`0.2863`
- Val loss（最优时）：`0.7157`

**Test（overall）**
- Accuracy：`0.3476`
- Recall：`0.1150`
- Specificity：`0.9108`
- Precision：`0.7575`
- F1：`0.1997`
- Balanced Acc：`0.5129`
- 混淆矩阵：TP=`678` TN=`2217` FP=`217` FN=`5216`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3054` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.3485` F1=`0.2032` BalAcc=`0.5131`

- 结束：`2026-08-02T04:42:13`
