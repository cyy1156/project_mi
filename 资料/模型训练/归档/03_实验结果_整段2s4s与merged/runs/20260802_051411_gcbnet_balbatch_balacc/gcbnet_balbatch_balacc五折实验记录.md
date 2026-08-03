# 特异度套件（20260802_051411 / gcbnet_balbatch_balacc）

- 开始：`2026-08-02T05:14:11`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`gcbnet` | 臂：`B1` — 普通CE + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\gcbnet_balbatch_balacc\merged_2s\run_20260802_051411`

---
## 最终结论

### Task — B1
- Val BalAcc：`0.5522 ± 0.0164`
- Test Spec：`0.5870 ± 0.1944`
- Test Rec：`0.4967 ± 0.2154`
- Test BalAcc：`0.5418 ± 0.0278`
- Test F1：`0.5702 ± 0.1570`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc）：`0.5725`
- Val F1（最优 checkpoint 时，附报）：`0.5323`
- Val loss（最优时）：`0.6997`

**Test（overall）**
- Accuracy：`0.4143`
- Recall：`0.2591`
- Specificity：`0.7975`
- Precision：`0.7596`
- F1：`0.3864`
- Balanced Acc：`0.5283`
- 混淆矩阵：TP=`1308` TN=`1630` FP=`414` FN=`3740`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5404` F1=`0.5708` BalAcc=`0.5871`
- `stieger_only`：Acc=`0.4068` F1=`0.3741` BalAcc=`0.5254`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`43`
- 验证最优轮次（best_epoch）：`25`
- Val 选模分数（Balanced Acc）：`0.5280`
- Val F1（最优 checkpoint 时，附报）：`0.5729`
- Val loss（最优时）：`0.7034`

**Test（overall）**
- Accuracy：`0.4159`
- Recall：`0.2771`
- Specificity：`0.7696`
- Precision：`0.7541`
- F1：`0.4053`
- Balanced Acc：`0.5234`
- 混淆矩阵：TP=`1208` TN=`1316` FP=`394` FN=`3151`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4095` F1=`0.3305` BalAcc=`0.5148`
- `stieger_only`：Acc=`0.4163` F1=`0.4100` BalAcc=`0.5236`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.5441`
- Val F1（最优 checkpoint 时，附报）：`0.6028`
- Val loss（最优时）：`0.6940`

**Test（overall）**
- Accuracy：`0.6480`
- Recall：`0.7005`
- Specificity：`0.4938`
- Precision：`0.8027`
- F1：`0.7481`
- Balanced Acc：`0.5972`
- 混淆矩阵：TP=`3331` TN=`799` FP=`819` FN=`1424`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5821` F1=`0.6200` BalAcc=`0.6210`
- `stieger_only`：Acc=`0.6523` F1=`0.7546` BalAcc=`0.5925`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5479`
- Val F1（最优 checkpoint 时，附报）：`0.5723`
- Val loss（最优时）：`0.7045`

**Test（overall）**
- Accuracy：`0.6270`
- Recall：`0.7873`
- Specificity：`0.2673`
- Precision：`0.7070`
- F1：`0.7450`
- Balanced Acc：`0.5273`
- 混淆矩阵：TP=`4463` TN=`675` FP=`1850` FN=`1206`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5543` F1=`0.5838` BalAcc=`0.6049`
- `stieger_only`：Acc=`0.6305` F1=`0.7504` BalAcc=`0.5229`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`60`
- 验证最优轮次（best_epoch）：`42`
- Val 选模分数（Balanced Acc）：`0.5687`
- Val F1（最优 checkpoint 时，附报）：`0.6178`
- Val loss（最优时）：`0.6963`

**Test（overall）**
- Accuracy：`0.5024`
- Recall：`0.4593`
- Specificity：`0.6068`
- Precision：`0.7388`
- F1：`0.5664`
- Balanced Acc：`0.5331`
- 混淆矩阵：TP=`2707` TN=`1477` FP=`957` FN=`3187`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4012` F1=`0.2754` BalAcc=`0.5525`
- `stieger_only`：Acc=`0.5045` F1=`0.5707` BalAcc=`0.5324`

- 结束：`2026-08-02T05:23:56`
