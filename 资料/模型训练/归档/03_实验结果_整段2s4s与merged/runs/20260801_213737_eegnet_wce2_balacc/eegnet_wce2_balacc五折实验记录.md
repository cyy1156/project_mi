# 特异度套件（20260801_213737 / eegnet_wce2_balacc）

- 开始：`2026-08-01T21:37:37`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`eegnet` | 臂：`A20` — 加权CE w0=2.0 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegnet_wce2_balacc\merged_2s\run_20260801_213737`

---
## 最终结论

### Task — A20
- Val BalAcc：`0.5891 ± 0.0220`
- Test Spec：`0.4229 ± 0.1758`
- Test Rec：`0.7188 ± 0.1700`
- Test BalAcc：`0.5709 ± 0.0402`
- Test F1：`0.7241 ± 0.1074`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.6045`
- Val F1（最优 checkpoint 时，附报）：`0.7687`
- Val loss（最优时）：`0.6717`

**Test（overall）**
- Accuracy：`0.6720`
- Recall：`0.8207`
- Specificity：`0.3048`
- Precision：`0.7446`
- F1：`0.7808`
- Balanced Acc：`0.5628`
- 混淆矩阵：TP=`4143` TN=`623` FP=`1421` FN=`905`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6995` F1=`0.8078` BalAcc=`0.5728`
- `stieger_only`：Acc=`0.6704` F1=`0.7791` BalAcc=`0.5627`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`49`
- 验证最优轮次（best_epoch）：`31`
- Val 选模分数（Balanced Acc）：`0.5703`
- Val F1（最优 checkpoint 时，附报）：`0.7590`
- Val loss（最优时）：`0.6719`

**Test（overall）**
- Accuracy：`0.4782`
- Recall：`0.3811`
- Specificity：`0.7257`
- Precision：`0.7798`
- F1：`0.5119`
- Balanced Acc：`0.5534`
- 混淆矩阵：TP=`1661` TN=`1241` FP=`469` FN=`2698`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3668` F1=`0.1310` BalAcc=`0.5276`
- `stieger_only`：Acc=`0.4860` F1=`0.5298` BalAcc=`0.5530`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5567`
- Val F1（最优 checkpoint 时，附报）：`0.7208`
- Val loss（最优时）：`0.6842`

**Test（overall）**
- Accuracy：`0.7171`
- Recall：`0.7868`
- Specificity：`0.5124`
- Precision：`0.8258`
- F1：`0.8058`
- Balanced Acc：`0.6496`
- 混淆矩阵：TP=`3741` TN=`829` FP=`789` FN=`1014`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7282` F1=`0.8257` BalAcc=`0.6079`
- `stieger_only`：Acc=`0.7164` F1=`0.8044` BalAcc=`0.6555`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`43`
- 验证最优轮次（best_epoch）：`25`
- Val 选模分数（Balanced Acc）：`0.6154`
- Val F1（最优 checkpoint 时，附报）：`0.7046`
- Val loss（最优时）：`0.6577`

**Test（overall）**
- Accuracy：`0.6484`
- Recall：`0.8282`
- Specificity：`0.2448`
- Precision：`0.7111`
- F1：`0.7652`
- Balanced Acc：`0.5365`
- 混淆矩阵：TP=`4695` TN=`618` FP=`1907` FN=`974`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6495` F1=`0.7543` BalAcc=`0.5699`
- `stieger_only`：Acc=`0.6484` F1=`0.7657` BalAcc=`0.5348`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`55`
- 验证最优轮次（best_epoch）：`37`
- Val 选模分数（Balanced Acc）：`0.5987`
- Val F1（最优 checkpoint 时，附报）：`0.7435`
- Val loss（最优时）：`0.6666`

**Test（overall）**
- Accuracy：`0.6458`
- Recall：`0.7774`
- Specificity：`0.3270`
- Precision：`0.7367`
- F1：`0.7565`
- Balanced Acc：`0.5522`
- 混淆矩阵：TP=`4582` TN=`796` FP=`1638` FN=`1312`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4910` F1=`0.5198` BalAcc=`0.5512`
- `stieger_only`：Acc=`0.6489` F1=`0.7600` BalAcc=`0.5520`

- 结束：`2026-08-01T22:06:08`
