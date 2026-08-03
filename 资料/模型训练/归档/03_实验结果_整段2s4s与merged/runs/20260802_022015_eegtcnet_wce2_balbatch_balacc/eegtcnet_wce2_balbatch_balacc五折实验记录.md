# 特异度套件（20260802_022015 / eegtcnet_wce2_balbatch_balacc）

- 开始：`2026-08-02T02:20:15`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`eegtcnet` | 臂：`B2` — w0=2 + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegtcnet_wce2_balbatch_balacc\merged_2s\run_20260802_022015`

---
## 最终结论

### Task — B2
- Val BalAcc：`0.5149 ± 0.0234`
- Test Spec：`0.8134 ± 0.2507`
- Test Rec：`0.2008 ± 0.2626`
- Test BalAcc：`0.5071 ± 0.0105`
- Test F1：`0.2315 ± 0.2901`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.7221`

**Test（overall）**
- Accuracy：`0.2879`
- Recall：`0.0000`
- Specificity：`0.9990`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.4995`
- 混淆矩阵：TP=`0` TN=`2042` FP=`2` FN=`5048`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3258` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2857` F1=`0.0000` BalAcc=`0.4995`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.7991`

**Test（overall）**
- Accuracy：`0.2818`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`1710` FP=`0` FN=`4359`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3241` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2788` F1=`0.0000` BalAcc=`0.5000`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5004`
- Val F1（最优 checkpoint 时，附报）：`0.0016`
- Val loss（最优时）：`0.7418`

**Test（overall）**
- Accuracy：`0.2544`
- Recall：`0.0008`
- Specificity：`0.9994`
- Precision：`0.8000`
- F1：`0.0017`
- Balanced Acc：`0.5001`
- 混淆矩阵：TP=`4` TN=`1617` FP=`1` FN=`4751`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3256` F1=`0.0000` BalAcc=`0.4961`
- `stieger_only`：Acc=`0.2497` F1=`0.0018` BalAcc=`0.5004`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`85`
- 验证最优轮次（best_epoch）：`67`
- Val 选模分数（Balanced Acc）：`0.5606`
- Val F1（最优 checkpoint 时，附报）：`0.3955`
- Val loss（最优时）：`0.7407`

**Test（overall）**
- Accuracy：`0.4608`
- Recall：`0.3547`
- Specificity：`0.6990`
- Precision：`0.7257`
- F1：`0.4765`
- Balanced Acc：`0.5269`
- 混淆矩阵：TP=`2011` TN=`1765` FP=`760` FN=`3658`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4266` F1=`0.3739` BalAcc=`0.5215`
- `stieger_only`：Acc=`0.4624` F1=`0.4808` BalAcc=`0.5270`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5137`
- Val F1（最优 checkpoint 时，附报）：`0.6625`
- Val loss（最优时）：`0.6920`

**Test（overall）**
- Accuracy：`0.5670`
- Recall：`0.6486`
- Specificity：`0.3694`
- Precision：`0.7135`
- F1：`0.6795`
- Balanced Acc：`0.5090`
- 混淆矩阵：TP=`3823` TN=`899` FP=`1535` FN=`2071`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6467` F1=`0.7306` BalAcc=`0.6193`
- `stieger_only`：Acc=`0.5654` F1=`0.6785` BalAcc=`0.5067`

- 结束：`2026-08-02T02:34:59`
