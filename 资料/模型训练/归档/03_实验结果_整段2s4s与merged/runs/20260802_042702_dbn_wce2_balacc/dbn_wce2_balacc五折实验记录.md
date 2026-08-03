# 特异度套件（20260802_042702 / dbn_wce2_balacc）

- 开始：`2026-08-02T04:27:02`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`dbn` | 臂：`A20` — 加权CE w0=2.0 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dbn_wce2_balacc\merged_2s\run_20260802_042702`

---
## 最终结论

### Task — A20
- Val BalAcc：`0.5266 ± 0.0115`
- Test Spec：`0.4513 ± 0.2618`
- Test Rec：`0.6026 ± 0.3014`
- Test BalAcc：`0.5269 ± 0.0256`
- Test F1：`0.6089 ± 0.2480`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.5375`
- Val F1（最优 checkpoint 时，附报）：`0.7317`
- Val loss（最优时）：`0.6870`

**Test（overall）**
- Accuracy：`0.5392`
- Recall：`0.5523`
- Specificity：`0.5068`
- Precision：`0.7345`
- F1：`0.6305`
- Balanced Acc：`0.5296`
- 混淆矩阵：TP=`2788` TN=`1036` FP=`1008` FN=`2260`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6414` F1=`0.7321` BalAcc=`0.5959`
- `stieger_only`：Acc=`0.5332` F1=`0.6240` BalAcc=`0.5261`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`60`
- 验证最优轮次（best_epoch）：`42`
- Val 选模分数（Balanced Acc）：`0.5208`
- Val F1（最优 checkpoint 时，附报）：`0.7411`
- Val loss（最优时）：`0.6935`

**Test（overall）**
- Accuracy：`0.5573`
- Recall：`0.5910`
- Specificity：`0.4713`
- Precision：`0.7402`
- F1：`0.6572`
- Balanced Acc：`0.5312`
- 混淆矩阵：TP=`2576` TN=`806` FP=`904` FN=`1783`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4497` F1=`0.4370` BalAcc=`0.5223`
- `stieger_only`：Acc=`0.5648` F1=`0.6687` BalAcc=`0.5297`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc）：`0.5165`
- Val F1（最优 checkpoint 时，附报）：`0.8048`
- Val loss（最优时）：`0.6905`

**Test（overall）**
- Accuracy：`0.7500`
- Recall：`0.9377`
- Specificity：`0.1984`
- Precision：`0.7747`
- F1：`0.8484`
- Balanced Acc：`0.5681`
- 混淆矩阵：TP=`4459` TN=`321` FP=`1297` FN=`296`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6974` F1=`0.8084` BalAcc=`0.5650`
- `stieger_only`：Acc=`0.7535` F1=`0.8509` BalAcc=`0.5685`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val 选模分数（Balanced Acc）：`0.5433`
- Val F1（最优 checkpoint 时，附报）：`0.5265`
- Val loss（最优时）：`0.6931`

**Test（overall）**
- Accuracy：`0.6466`
- Recall：`0.8545`
- Specificity：`0.1798`
- Precision：`0.7005`
- F1：`0.7699`
- Balanced Acc：`0.5171`
- 混淆矩阵：TP=`4844` TN=`454` FP=`2071` FN=`825`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5190` F1=`0.5473` BalAcc=`0.5678`
- `stieger_only`：Acc=`0.6526` F1=`0.7770` BalAcc=`0.5139`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5150`
- Val F1（最优 checkpoint 时，附报）：`0.5892`
- Val loss（最优时）：`0.6933`

**Test（overall）**
- Accuracy：`0.3178`
- Recall：`0.0774`
- Specificity：`0.9002`
- Precision：`0.6524`
- F1：`0.1383`
- Balanced Acc：`0.4888`
- 混淆矩阵：TP=`456` TN=`2191` FP=`243` FN=`5438`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3054` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.3181` F1=`0.1408` BalAcc=`0.4885`

- 结束：`2026-08-02T04:30:56`
