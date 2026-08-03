# 特异度套件（20260802_061518 / dgcnn_smote_balacc）

- 开始：`2026-08-02T06:15:18`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`dgcnn` | 臂：`S1` — 普通CE + SMOTE
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dgcnn_smote_balacc\merged_2s\run_20260802_061518`

---
## 最终结论

### Task — S1
- Val BalAcc：`0.5297 ± 0.0055`
- Test Spec：`0.4957 ± 0.1811`
- Test Rec：`0.5728 ± 0.1674`
- Test BalAcc：`0.5343 ± 0.0155`
- Test F1：`0.6323 ± 0.0872`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5354`
- Val F1（最优 checkpoint 时，附报）：`0.7081`
- Val loss（最优时）：`0.6746`

**Test（overall）**
- Accuracy：`0.5049`
- Recall：`0.4697`
- Specificity：`0.5920`
- Precision：`0.7398`
- F1：`0.5746`
- Balanced Acc：`0.5308`
- 混淆矩阵：TP=`2371` TN=`1210` FP=`834` FN=`2677`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6338` F1=`0.7249` BalAcc=`0.5902`
- `stieger_only`：Acc=`0.4973` F1=`0.5643` BalAcc=`0.5282`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5302`
- Val F1（最优 checkpoint 时，附报）：`0.6591`
- Val loss（最优时）：`0.6853`

**Test（overall）**
- Accuracy：`0.4745`
- Recall：`0.4070`
- Specificity：`0.6468`
- Precision：`0.7460`
- F1：`0.5266`
- Balanced Acc：`0.5269`
- 混淆矩阵：TP=`1774` TN=`1106` FP=`604` FN=`2585`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3568` F1=`0.1579` BalAcc=`0.5020`
- `stieger_only`：Acc=`0.4828` F1=`0.5441` BalAcc=`0.5264`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5197`
- Val F1（最优 checkpoint 时，附报）：`0.4156`
- Val loss（最优时）：`0.7052`

**Test（overall）**
- Accuracy：`0.5624`
- Recall：`0.5628`
- Specificity：`0.5612`
- Precision：`0.7903`
- F1：`0.6574`
- Balanced Acc：`0.5620`
- 混淆矩阵：TP=`2676` TN=`908` FP=`710` FN=`2079`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4103` F1=`0.3275` BalAcc=`0.5131`
- `stieger_only`：Acc=`0.5723` F1=`0.6719` BalAcc=`0.5614`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5340`
- Val F1（最优 checkpoint 时，附报）：`0.5956`
- Val loss（最优时）：`0.7044`

**Test（overall）**
- Accuracy：`0.6590`
- Recall：`0.8898`
- Specificity：`0.1410`
- Precision：`0.6993`
- F1：`0.7831`
- Balanced Acc：`0.5154`
- 混淆矩阵：TP=`5044` TN=`356` FP=`2169` FN=`625`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5924` F1=`0.6544` BalAcc=`0.6045`
- `stieger_only`：Acc=`0.6622` F1=`0.7876` BalAcc=`0.5104`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.5292`
- Val F1（最优 checkpoint 时，附报）：`0.7416`
- Val loss（最优时）：`0.6750`

**Test（overall）**
- Accuracy：`0.5357`
- Recall：`0.5348`
- Specificity：`0.5378`
- Precision：`0.7370`
- F1：`0.6198`
- Balanced Acc：`0.5363`
- 混淆矩阵：TP=`3152` TN=`1309` FP=`1125` FN=`2742`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3234` F1=`0.0504` BalAcc=`0.5129`
- `stieger_only`：Acc=`0.5400` F1=`0.6265` BalAcc=`0.5365`

- 结束：`2026-08-02T06:21:32`
