# 特异度套件（20260802_005009 / deep_smote_balacc）

- 开始：`2026-08-02T00:50:09`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`deep` | 臂：`S1` — 普通CE + SMOTE
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\deep_smote_balacc\merged_2s\run_20260802_005009`

---
## 最终结论

### Task — S1
- Val BalAcc：`0.5029 ± 0.0058`
- Test Spec：`0.9982 ± 0.0035`
- Test Rec：`0.0025 ± 0.0050`
- Test BalAcc：`0.5004 ± 0.0007`
- Test F1：`0.0049 ± 0.0098`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`4.1869`

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

- 早停/结束轮次（stopped_epoch）：`95`
- 验证最优轮次（best_epoch）：`77`
- Val 选模分数（Balanced Acc）：`0.5146`
- Val F1（最优 checkpoint 时，附报）：`0.1195`
- Val loss（最优时）：`1.3519`

**Test（overall）**
- Accuracy：`0.2882`
- Recall：`0.0124`
- Specificity：`0.9912`
- Precision：`0.7826`
- F1：`0.0244`
- Balanced Acc：`0.5018`
- 混淆矩阵：TP=`54` TN=`1695` FP=`15` FN=`4305`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3241` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2857` F1=`0.0260` BalAcc=`0.5019`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`3.4121`

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
- Val loss（最优时）：`3.7354`

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
- Val loss（最优时）：`2.8330`

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

- 结束：`2026-08-02T01:04:43`
