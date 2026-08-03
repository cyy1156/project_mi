# 特异度套件（20260802_035628 / conformer_smote_balacc）

- 开始：`2026-08-02T03:56:28`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`conformer` | 臂：`S1` — 普通CE + SMOTE
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\conformer_smote_balacc\merged_2s\run_20260802_035628`

---
## 最终结论

### Task — S1
- Val BalAcc：`0.5648 ± 0.0261`
- Test Spec：`0.2704 ± 0.1465`
- Test Rec：`0.8433 ± 0.0991`
- Test BalAcc：`0.5568 ± 0.0362`
- Test F1：`0.7870 ± 0.0431`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.6037`
- Val F1（最优 checkpoint 时，附报）：`0.7474`
- Val loss（最优时）：`0.7109`

**Test（overall）**
- Accuracy：`0.6567`
- Recall：`0.7924`
- Specificity：`0.3214`
- Precision：`0.7425`
- F1：`0.7667`
- Balanced Acc：`0.5569`
- 混淆矩阵：TP=`4000` TN=`657` FP=`1387` FN=`1048`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6843` F1=`0.7731` BalAcc=`0.6237`
- `stieger_only`：Acc=`0.6550` F1=`0.7663` BalAcc=`0.5524`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.5371`
- Val F1（最优 checkpoint 时，附报）：`0.8025`
- Val loss（最优时）：`0.6828`

**Test（overall）**
- Accuracy：`0.6253`
- Recall：`0.6747`
- Specificity：`0.4994`
- Precision：`0.7746`
- F1：`0.7212`
- Balanced Acc：`0.5871`
- 混淆矩阵：TP=`2941` TN=`854` FP=`856` FN=`1418`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5603` F1=`0.5658` BalAcc=`0.6344`
- `stieger_only`：Acc=`0.6299` F1=`0.7293` BalAcc=`0.5812`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.5680`
- Val F1（最优 checkpoint 时，附报）：`0.7839`
- Val loss（最优时）：`0.6627`

**Test（overall）**
- Accuracy：`0.7568`
- Recall：`0.9123`
- Specificity：`0.2998`
- Precision：`0.7929`
- F1：`0.8484`
- Balanced Acc：`0.6060`
- 混淆矩阵：TP=`4338` TN=`485` FP=`1133` FN=`417`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7103` F1=`0.8215` BalAcc=`0.5626`
- `stieger_only`：Acc=`0.7598` F1=`0.8502` BalAcc=`0.6109`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5803`
- Val F1（最优 checkpoint 时，附报）：`0.7006`
- Val loss（最优时）：`0.8335`

**Test（overall）**
- Accuracy：`0.6619`
- Recall：`0.8869`
- Specificity：`0.1568`
- Precision：`0.7025`
- F1：`0.7840`
- Balanced Acc：`0.5219`
- 混淆矩阵：TP=`5028` TN=`396` FP=`2129` FN=`641`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5924` F1=`0.7126` BalAcc=`0.5079`
- `stieger_only`：Acc=`0.6652` F1=`0.7871` BalAcc=`0.5223`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`36`
- 验证最优轮次（best_epoch）：`18`
- Val 选模分数（Balanced Acc）：`0.5351`
- Val F1（最优 checkpoint 时，附报）：`0.8253`
- Val loss（最优时）：`0.6057`

**Test（overall）**
- Accuracy：`0.6943`
- Recall：`0.9503`
- Specificity：`0.0744`
- Precision：`0.7131`
- F1：`0.8148`
- Balanced Acc：`0.5123`
- 混淆矩阵：TP=`5601` TN=`181` FP=`2253` FN=`293`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5868` F1=`0.6761` BalAcc=`0.5652`
- `stieger_only`：Acc=`0.6965` F1=`0.8170` BalAcc=`0.5110`

- 结束：`2026-08-02T04:21:54`
