# 特异度套件（20260802_004151 / deep_wce2_balbatch_balacc）

- 开始：`2026-08-02T00:41:51`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`deep` | 臂：`B2` — w0=2 + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\deep_wce2_balbatch_balacc\merged_2s\run_20260802_004151`

---
## 最终结论

### Task — B2
- Val BalAcc：`0.5053 ± 0.0107`
- Test Spec：`0.9949 ± 0.0103`
- Test Rec：`0.0052 ± 0.0105`
- Test BalAcc：`0.5000 ± 0.0001`
- Test F1：`0.0101 ± 0.0202`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.9472`

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

- 早停/结束轮次（stopped_epoch）：`65`
- 验证最优轮次（best_epoch）：`47`
- Val 选模分数（Balanced Acc）：`0.5267`
- Val F1（最优 checkpoint 时，附报）：`0.2613`
- Val loss（最优时）：`0.7638`

**Test（overall）**
- Accuracy：`0.2933`
- Recall：`0.0262`
- Specificity：`0.9743`
- Precision：`0.7215`
- F1：`0.0505`
- Balanced Acc：`0.5002`
- 混淆矩阵：TP=`114` TN=`1666` FP=`44` FN=`4245`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3241` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2911` F1=`0.0537` BalAcc=`0.5000`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.9803`

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

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`1.0191`

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
- Val loss（最优时）：`1.0386`

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

- 结束：`2026-08-02T00:50:09`
