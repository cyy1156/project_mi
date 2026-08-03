# 特异度套件（20260802_023459 / eegtcnet_smote_balacc）

- 开始：`2026-08-02T02:34:59`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`eegtcnet` | 臂：`S1` — 普通CE + SMOTE
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegtcnet_smote_balacc\merged_2s\run_20260802_023459`

---
## 最终结论

### Task — S1
- Val BalAcc：`0.5543 ± 0.0243`
- Test Spec：`0.2346 ± 0.2049`
- Test Rec：`0.8525 ± 0.1660`
- Test BalAcc：`0.5436 ± 0.0290`
- Test F1：`0.7807 ± 0.0783`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5883`
- Val F1（最优 checkpoint 时，附报）：`0.7585`
- Val loss（最优时）：`0.6685`

**Test（overall）**
- Accuracy：`0.6990`
- Recall：`0.9010`
- Specificity：`0.2001`
- Precision：`0.7356`
- F1：`0.8099`
- Balanced Acc：`0.5505`
- 混淆矩阵：TP=`4548` TN=`409` FP=`1635` FN=`500`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7071` F1=`0.8073` BalAcc=`0.5985`
- `stieger_only`：Acc=`0.6985` F1=`0.8100` BalAcc=`0.5473`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5335`
- Val F1（最优 checkpoint 时，附报）：`0.7353`
- Val loss（最优时）：`0.6671`

**Test（overall）**
- Accuracy：`0.5507`
- Recall：`0.5240`
- Specificity：`0.6187`
- Precision：`0.7779`
- F1：`0.6262`
- Balanced Acc：`0.5713`
- 混淆矩阵：TP=`2284` TN=`1058` FP=`652` FN=`2075`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3467` F1=`0.1034` BalAcc=`0.5046`
- `stieger_only`：Acc=`0.5650` F1=`0.6478` BalAcc=`0.5731`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.5385`
- Val F1（最优 checkpoint 时，附报）：`0.7767`
- Val loss（最优时）：`0.6546`

**Test（overall）**
- Accuracy：`0.7453`
- Recall：`0.9197`
- Specificity：`0.2330`
- Precision：`0.7789`
- F1：`0.8435`
- Balanced Acc：`0.5763`
- 混淆矩阵：TP=`4373` TN=`377` FP=`1241` FN=`382`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6872` F1=`0.8094` BalAcc=`0.5294`
- `stieger_only`：Acc=`0.7491` F1=`0.8457` BalAcc=`0.5813`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5791`
- Val F1（最优 checkpoint 时，附报）：`0.8025`
- Val loss（最优时）：`0.6081`

**Test（overall）**
- Accuracy：`0.6842`
- Recall：`0.9704`
- Specificity：`0.0416`
- Precision：`0.6945`
- F1：`0.8096`
- Balanced Acc：`0.5060`
- 混淆矩阵：TP=`5501` TN=`105` FP=`2420` FN=`168`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5652` F1=`0.6947` BalAcc=`0.4747`
- `stieger_only`：Acc=`0.6898` F1=`0.8142` BalAcc=`0.5071`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.5320`
- Val F1（最优 checkpoint 时，附报）：`0.8310`
- Val loss（最优时）：`0.5838`

**Test（overall）**
- Accuracy：`0.6939`
- Recall：`0.9476`
- Specificity：`0.0797`
- Precision：`0.7137`
- F1：`0.8142`
- Balanced Acc：`0.5136`
- 混淆矩阵：TP=`5585` TN=`194` FP=`2240` FN=`309`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3413` F1=`0.3605` BalAcc=`0.3885`
- `stieger_only`：Acc=`0.7011` F1=`0.8200` BalAcc=`0.5159`

- 结束：`2026-08-02T02:50:06`
