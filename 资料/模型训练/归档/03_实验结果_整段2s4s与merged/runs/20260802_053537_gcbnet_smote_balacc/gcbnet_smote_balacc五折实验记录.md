# 特异度套件（20260802_053537 / gcbnet_smote_balacc）

- 开始：`2026-08-02T05:35:37`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`gcbnet` | 臂：`S1` — 普通CE + SMOTE
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\gcbnet_smote_balacc\merged_2s\run_20260802_053537`

---
## 最终结论

### Task — S1
- Val BalAcc：`0.5292 ± 0.0049`
- Test Spec：`0.4587 ± 0.1482`
- Test Rec：`0.6037 ± 0.1428`
- Test BalAcc：`0.5312 ± 0.0188`
- Test F1：`0.6517 ± 0.1009`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5291`
- Val F1（最优 checkpoint 时，附报）：`0.7856`
- Val loss（最优时）：`0.6651`

**Test（overall）**
- Accuracy：`0.5959`
- Recall：`0.6741`
- Specificity：`0.4026`
- Precision：`0.7359`
- F1：`0.7037`
- Balanced Acc：`0.5384`
- 混淆矩阵：TP=`3403` TN=`823` FP=`1221` FN=`1645`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7071` F1=`0.8060` BalAcc=`0.6025`
- `stieger_only`：Acc=`0.5893` F1=`0.6969` BalAcc=`0.5354`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5261`
- Val F1（最优 checkpoint 时，附报）：`0.5966`
- Val loss（最优时）：`0.6941`

**Test（overall）**
- Accuracy：`0.4368`
- Recall：`0.3209`
- Specificity：`0.7322`
- Precision：`0.7534`
- F1：`0.4501`
- Balanced Acc：`0.5266`
- 混淆矩阵：TP=`1399` TN=`1252` FP=`458` FN=`2960`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3618` F1=`0.1533` BalAcc=`0.5117`
- `stieger_only`：Acc=`0.4421` F1=`0.4652` BalAcc=`0.5259`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5261`
- Val F1（最优 checkpoint 时，附报）：`0.5121`
- Val loss（最优时）：`0.6979`

**Test（overall）**
- Accuracy：`0.6002`
- Recall：`0.6376`
- Specificity：`0.4901`
- Precision：`0.7861`
- F1：`0.7041`
- Balanced Acc：`0.5639`
- 混淆矩阵：TP=`3032` TN=`793` FP=`825` FN=`1723`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4744` F1=`0.4562` BalAcc=`0.5508`
- `stieger_only`：Acc=`0.6084` F1=`0.7155` BalAcc=`0.5607`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.5387`
- Val F1（最优 checkpoint 时，附报）：`0.4680`
- Val loss（最优时）：`0.7322`

**Test（overall）**
- Accuracy：`0.5803`
- Recall：`0.6902`
- Specificity：`0.3335`
- Precision：`0.6992`
- F1：`0.6947`
- Balanced Acc：`0.5119`
- 混淆矩阵：TP=`3913` TN=`842` FP=`1683` FN=`1756`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5245` F1=`0.5232` BalAcc=`0.6003`
- `stieger_only`：Acc=`0.5829` F1=`0.7005` BalAcc=`0.5069`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5257`
- Val F1（最优 checkpoint 时，附报）：`0.8195`
- Val loss（最优时）：`0.6597`

**Test（overall）**
- Accuracy：`0.5902`
- Recall：`0.6955`
- Specificity：`0.3353`
- Precision：`0.7170`
- F1：`0.7061`
- Balanced Acc：`0.5154`
- 混淆矩阵：TP=`4099` TN=`816` FP=`1618` FN=`1795`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3114` F1=`0.0171` BalAcc=`0.5043`
- `stieger_only`：Acc=`0.5959` F1=`0.7131` BalAcc=`0.5151`

- 结束：`2026-08-02T05:43:30`
