# 特异度套件（20260802_054411 / dgcnn_wce22_balacc）

- 开始：`2026-08-02T05:44:11`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`dgcnn` | 臂：`A22` — 加权CE w0=2.2 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dgcnn_wce22_balacc\merged_2s\run_20260802_054411`

---
## 最终结论

### Task — A22
- Val BalAcc：`0.5495 ± 0.0139`
- Test Spec：`0.4350 ± 0.2102`
- Test Rec：`0.6485 ± 0.2067`
- Test BalAcc：`0.5418 ± 0.0249`
- Test F1：`0.6731 ± 0.1198`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`81`
- 验证最优轮次（best_epoch）：`63`
- Val 选模分数（Balanced Acc）：`0.5756`
- Val F1（最优 checkpoint 时，附报）：`0.6474`
- Val loss（最优时）：`0.6832`

**Test（overall）**
- Accuracy：`0.4683`
- Recall：`0.3829`
- Specificity：`0.6791`
- Precision：`0.7466`
- F1：`0.5062`
- Balanced Acc：`0.5310`
- 混淆矩阵：TP=`1933` TN=`1388` FP=`656` FN=`3115`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6061` F1=`0.6855` BalAcc=`0.5897`
- `stieger_only`：Acc=`0.4601` F1=`0.4938` BalAcc=`0.5285`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`56`
- 验证最优轮次（best_epoch）：`38`
- Val 选模分数（Balanced Acc）：`0.5358`
- Val F1（最优 checkpoint 时，附报）：`0.6596`
- Val loss（最优时）：`0.6912`

**Test（overall）**
- Accuracy：`0.4989`
- Recall：`0.4393`
- Specificity：`0.6509`
- Precision：`0.7623`
- F1：`0.5574`
- Balanced Acc：`0.5451`
- 混淆矩阵：TP=`1915` TN=`1113` FP=`597` FN=`2444`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3744` F1=`0.1836` BalAcc=`0.5210`
- `stieger_only`：Acc=`0.5077` F1=`0.5748` BalAcc=`0.5444`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`61`
- 验证最优轮次（best_epoch）：`43`
- Val 选模分数（Balanced Acc）：`0.5459`
- Val F1（最优 checkpoint 时，附报）：`0.7075`
- Val loss（最优时）：`0.6885`

**Test（overall）**
- Accuracy：`0.6843`
- Recall：`0.7830`
- Specificity：`0.3943`
- Precision：`0.7916`
- F1：`0.7873`
- Balanced Acc：`0.5886`
- 混淆矩阵：TP=`3723` TN=`638` FP=`980` FN=`1032`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6897` F1=`0.7721` BalAcc=`0.6412`
- `stieger_only`：Acc=`0.6839` F1=`0.7882` BalAcc=`0.5841`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5400`
- Val F1（最优 checkpoint 时，附报）：`0.6603`
- Val loss（最优时）：`0.6875`

**Test（overall）**
- Accuracy：`0.6759`
- Recall：`0.9266`
- Specificity：`0.1129`
- Precision：`0.7011`
- F1：`0.7982`
- Balanced Acc：`0.5197`
- 混淆矩阵：TP=`5253` TN=`285` FP=`2240` FN=`416`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5761` F1=`0.6422` BalAcc=`0.5836`
- `stieger_only`：Acc=`0.6806` F1=`0.8036` BalAcc=`0.5159`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5499`
- Val F1（最优 checkpoint 时，附报）：`0.7942`
- Val loss（最优时）：`0.6786`

**Test（overall）**
- Accuracy：`0.6018`
- Recall：`0.7107`
- Specificity：`0.3381`
- Precision：`0.7222`
- F1：`0.7164`
- Balanced Acc：`0.5244`
- 混淆矩阵：TP=`4189` TN=`823` FP=`1611` FN=`1705`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3234` F1=`0.0504` BalAcc=`0.5129`
- `stieger_only`：Acc=`0.6075` F1=`0.7233` BalAcc=`0.5242`

- 结束：`2026-08-02T05:54:04`
