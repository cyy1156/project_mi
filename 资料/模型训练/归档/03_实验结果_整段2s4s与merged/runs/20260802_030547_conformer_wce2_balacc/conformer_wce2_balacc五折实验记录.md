# 特异度套件（20260802_030547 / conformer_wce2_balacc）

- 开始：`2026-08-02T03:05:47`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`conformer` | 臂：`A20` — 加权CE w0=2.0 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\conformer_wce2_balacc\merged_2s\run_20260802_030547`

---
## 最终结论

### Task — A20
- Val BalAcc：`0.5847 ± 0.0276`
- Test Spec：`0.3742 ± 0.1851`
- Test Rec：`0.7419 ± 0.1705`
- Test BalAcc：`0.5580 ± 0.0293`
- Test F1：`0.7347 ± 0.0884`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.6237`
- Val F1（最优 checkpoint 时，附报）：`0.6631`
- Val loss（最优时）：`0.6667`

**Test（overall）**
- Accuracy：`0.5627`
- Recall：`0.5624`
- Specificity：`0.5636`
- Precision：`0.7609`
- F1：`0.6468`
- Balanced Acc：`0.5630`
- 混淆矩阵：TP=`2839` TN=`1152` FP=`892` FN=`2209`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6035` F1=`0.6609` BalAcc=`0.6199`
- `stieger_only`：Acc=`0.5603` F1=`0.6460` BalAcc=`0.5592`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.5434`
- Val F1（最优 checkpoint 时，附报）：`0.7889`
- Val loss（最优时）：`0.7599`

**Test（overall）**
- Accuracy：`0.5477`
- Recall：`0.5189`
- Specificity：`0.6211`
- Precision：`0.7773`
- F1：`0.6224`
- Balanced Acc：`0.5700`
- 混淆矩阵：TP=`2262` TN=`1062` FP=`648` FN=`2097`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4774` F1=`0.3953` BalAcc=`0.5993`
- `stieger_only`：Acc=`0.5526` F1=`0.6336` BalAcc=`0.5655`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.5819`
- Val F1（最优 checkpoint 时，附报）：`0.7695`
- Val loss（最优时）：`0.6652`

**Test（overall）**
- Accuracy：`0.7632`
- Recall：`0.9274`
- Specificity：`0.2806`
- Precision：`0.7912`
- F1：`0.8539`
- Balanced Acc：`0.6040`
- 混淆矩阵：TP=`4410` TN=`454` FP=`1164` FN=`345`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7128` F1=`0.8239` BalAcc=`0.5625`
- `stieger_only`：Acc=`0.7665` F1=`0.8559` BalAcc=`0.6086`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.6039`
- Val F1（最优 checkpoint 时，附报）：`0.6962`
- Val loss（最优时）：`0.6810`

**Test（overall）**
- Accuracy：`0.6329`
- Recall：`0.7980`
- Specificity：`0.2622`
- Precision：`0.7083`
- F1：`0.7505`
- Balanced Acc：`0.5301`
- 混淆矩阵：TP=`4524` TN=`662` FP=`1863` FN=`1145`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5462` F1=`0.6330` BalAcc=`0.5287`
- `stieger_only`：Acc=`0.6370` F1=`0.7551` BalAcc=`0.5298`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5706`
- Val F1（最优 checkpoint 时，附报）：`0.8073`
- Val loss（最优时）：`0.6635`

**Test（overall）**
- Accuracy：`0.6807`
- Recall：`0.9026`
- Specificity：`0.1434`
- Precision：`0.7184`
- F1：`0.8001`
- Balanced Acc：`0.5230`
- 混淆矩阵：TP=`5320` TN=`349` FP=`2085` FN=`574`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5329` F1=`0.5895` BalAcc=`0.5649`
- `stieger_only`：Acc=`0.6837` F1=`0.8031` BalAcc=`0.5218`

- 结束：`2026-08-02T03:21:39`
