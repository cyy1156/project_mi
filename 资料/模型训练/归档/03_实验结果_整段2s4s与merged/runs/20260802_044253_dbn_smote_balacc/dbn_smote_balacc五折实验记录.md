# 特异度套件（20260802_044253 / dbn_smote_balacc）

- 开始：`2026-08-02T04:42:53`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`dbn` | 臂：`S1` — 普通CE + SMOTE
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dbn_smote_balacc\merged_2s\run_20260802_044253`

---
## 最终结论

### Task — S1
- Val BalAcc：`0.5268 ± 0.0125`
- Test Spec：`0.4014 ± 0.2032`
- Test Rec：`0.6433 ± 0.2309`
- Test BalAcc：`0.5224 ± 0.0199`
- Test F1：`0.6622 ± 0.1394`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5293`
- Val F1（最优 checkpoint 时，附报）：`0.7214`
- Val loss（最优时）：`0.6788`

**Test（overall）**
- Accuracy：`0.5361`
- Recall：`0.5365`
- Specificity：`0.5352`
- Precision：`0.7403`
- F1：`0.6221`
- Balanced Acc：`0.5358`
- 混淆矩阵：TP=`2708` TN=`1094` FP=`950` FN=`2340`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6414` F1=`0.7280` BalAcc=`0.6039`
- `stieger_only`：Acc=`0.5299` F1=`0.6153` BalAcc=`0.5323`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`55`
- 验证最优轮次（best_epoch）：`37`
- Val 选模分数（Balanced Acc）：`0.5244`
- Val F1（最优 checkpoint 时，附报）：`0.7256`
- Val loss（最优时）：`0.6683`

**Test（overall）**
- Accuracy：`0.5233`
- Recall：`0.5357`
- Specificity：`0.4918`
- Precision：`0.7288`
- F1：`0.6175`
- Balanced Acc：`0.5137`
- 混淆矩阵：TP=`2335` TN=`841` FP=`869` FN=`2024`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4397` F1=`0.4574` BalAcc=`0.4887`
- `stieger_only`：Acc=`0.5292` F1=`0.6267` BalAcc=`0.5143`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.5083`
- Val F1（最优 checkpoint 时，附报）：`0.8066`
- Val loss（最优时）：`0.6644`

**Test（overall）**
- Accuracy：`0.7563`
- Recall：`0.9674`
- Specificity：`0.1360`
- Precision：`0.7669`
- F1：`0.8556`
- Balanced Acc：`0.5517`
- 混淆矩阵：TP=`4600` TN=`220` FP=`1398` FN=`155`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6897` F1=`0.8039` BalAcc=`0.5553`
- `stieger_only`：Acc=`0.7607` F1=`0.8587` BalAcc=`0.5511`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.5474`
- Val F1（最优 checkpoint 时，附报）：`0.5670`
- Val loss（最优时）：`0.6953`

**Test（overall）**
- Accuracy：`0.6434`
- Recall：`0.8460`
- Specificity：`0.1885`
- Precision：`0.7007`
- F1：`0.7665`
- Balanced Acc：`0.5173`
- 混淆矩阵：TP=`4796` TN=`476` FP=`2049` FN=`873`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5272` F1=`0.5797` BalAcc=`0.5519`
- `stieger_only`：Acc=`0.6489` F1=`0.7729` BalAcc=`0.5149`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5246`
- Val F1（最优 checkpoint 时，附报）：`0.7448`
- Val loss（最优时）：`0.6795`

**Test（overall）**
- Accuracy：`0.4259`
- Recall：`0.3310`
- Specificity：`0.6557`
- Precision：`0.6995`
- F1：`0.4494`
- Balanced Acc：`0.4934`
- 混淆矩阵：TP=`1951` TN=`1596` FP=`838` FN=`3943`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3054` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.4284` F1=`0.4555` BalAcc=`0.4930`

- 结束：`2026-08-02T04:47:59`
