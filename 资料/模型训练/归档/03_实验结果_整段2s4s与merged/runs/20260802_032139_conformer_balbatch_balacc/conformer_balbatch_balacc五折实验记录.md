# 特异度套件（20260802_032139 / conformer_balbatch_balacc）

- 开始：`2026-08-02T03:21:39`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`conformer` | 臂：`B1` — 普通CE + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\conformer_balbatch_balacc\merged_2s\run_20260802_032139`

---
## 最终结论

### Task — B1
- Val BalAcc：`0.5877 ± 0.0238`
- Test Spec：`0.4575 ± 0.1498`
- Test Rec：`0.6906 ± 0.1532`
- Test BalAcc：`0.5740 ± 0.0407`
- Test F1：`0.7144 ± 0.0898`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`13`
- Val 选模分数（Balanced Acc）：`0.6270`
- Val F1（最优 checkpoint 时，附报）：`0.6678`
- Val loss（最优时）：`0.6491`

**Test（overall）**
- Accuracy：`0.5558`
- Recall：`0.5483`
- Specificity：`0.5744`
- Precision：`0.7609`
- F1：`0.6373`
- Balanced Acc：`0.5613`
- 混淆矩阵：TP=`2768` TN=`1174` FP=`870` FN=`2280`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5859` F1=`0.6372` BalAcc=`0.6107`
- `stieger_only`：Acc=`0.5541` F1=`0.6374` BalAcc=`0.5580`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5633`
- Val F1（最优 checkpoint 时，附报）：`0.7805`
- Val loss（最优时）：`0.6245`

**Test（overall）**
- Accuracy：`0.5255`
- Recall：`0.4726`
- Specificity：`0.6602`
- Precision：`0.7800`
- F1：`0.5886`
- Balanced Acc：`0.5664`
- 混淆矩阵：TP=`2060` TN=`1129` FP=`581` FN=`2299`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5000` F1=`0.4363` BalAcc=`0.6160`
- `stieger_only`：Acc=`0.5272` F1=`0.5967` BalAcc=`0.5609`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.5680`
- Val F1（最优 checkpoint 时，附报）：`0.7237`
- Val loss（最优时）：`0.6666`

**Test（overall）**
- Accuracy：`0.7522`
- Recall：`0.8562`
- Specificity：`0.4468`
- Precision：`0.8198`
- F1：`0.8376`
- Balanced Acc：`0.6515`
- 混淆矩阵：TP=`4071` TN=`723` FP=`895` FN=`684`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7641` F1=`0.8497` BalAcc=`0.6446`
- `stieger_only`：Acc=`0.7515` F1=`0.8368` BalAcc=`0.6540`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.6019`
- Val F1（最优 checkpoint 时，附报）：`0.6967`
- Val loss（最优时）：`0.6657`

**Test（overall）**
- Accuracy：`0.6450`
- Recall：`0.8282`
- Specificity：`0.2337`
- Precision：`0.7081`
- F1：`0.7635`
- Balanced Acc：`0.5309`
- 混淆矩阵：TP=`4695` TN=`590` FP=`1935` FN=`974`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5679` F1=`0.6362` BalAcc=`0.5732`
- `stieger_only`：Acc=`0.6486` F1=`0.7682` BalAcc=`0.5284`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc）：`0.5781`
- Val F1（最优 checkpoint 时，附报）：`0.6898`
- Val loss（最优时）：`0.6674`

**Test（overall）**
- Accuracy：`0.6378`
- Recall：`0.7475`
- Specificity：`0.3722`
- Precision：`0.7425`
- F1：`0.7450`
- Balanced Acc：`0.5599`
- 混淆矩阵：TP=`4406` TN=`906` FP=`1528` FN=`1488`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4850` F1=`0.4416` BalAcc=`0.6073`
- `stieger_only`：Acc=`0.6410` F1=`0.7490` BalAcc=`0.5586`

- 结束：`2026-08-02T03:38:49`
