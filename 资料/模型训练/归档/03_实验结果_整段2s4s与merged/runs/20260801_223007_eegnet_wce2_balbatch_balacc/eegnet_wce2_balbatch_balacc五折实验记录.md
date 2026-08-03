# 特异度套件（20260801_223007 / eegnet_wce2_balbatch_balacc）

- 开始：`2026-08-01T22:30:07`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`eegnet` | 臂：`B2` — w0=2 + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegnet_wce2_balbatch_balacc\merged_2s\run_20260801_223007`

---
## 最终结论

### Task — B2
- Val BalAcc：`0.5570 ± 0.0286`
- Test Spec：`0.7386 ± 0.1639`
- Test Rec：`0.3561 ± 0.1642`
- Test BalAcc：`0.5473 ± 0.0194`
- Test F1：`0.4635 ± 0.1404`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`130`
- 验证最优轮次（best_epoch）：`112`
- Val 选模分数（Balanced Acc）：`0.5488`
- Val F1（最优 checkpoint 时，附报）：`0.2590`
- Val loss（最优时）：`0.7381`

**Test（overall）**
- Accuracy：`0.4047`
- Recall：`0.2338`
- Specificity：`0.8268`
- Precision：`0.7692`
- F1：`0.3586`
- Balanced Acc：`0.5303`
- 混淆矩阵：TP=`1180` TN=`1690` FP=`354` FN=`3868`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3939` F1=`0.2208` BalAcc=`0.5365`
- `stieger_only`：Acc=`0.4053` F1=`0.3653` BalAcc=`0.5292`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`64`
- 验证最优轮次（best_epoch）：`46`
- Val 选模分数（Balanced Acc）：`0.5712`
- Val F1（最优 checkpoint 时，附报）：`0.6641`
- Val loss（最优时）：`0.6798`

**Test（overall）**
- Accuracy：`0.3966`
- Recall：`0.2244`
- Specificity：`0.8357`
- Precision：`0.7768`
- F1：`0.3482`
- Balanced Acc：`0.5300`
- 混淆矩阵：TP=`978` TN=`1429` FP=`281` FN=`3381`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3543` F1=`0.0919` BalAcc=`0.5203`
- `stieger_only`：Acc=`0.3996` F1=`0.3618` BalAcc=`0.5294`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc）：`0.5046`
- Val F1（最优 checkpoint 时，附报）：`0.0985`
- Val loss（最优时）：`0.7299`

**Test（overall）**
- Accuracy：`0.3935`
- Recall：`0.2095`
- Specificity：`0.9345`
- Precision：`0.9038`
- F1：`0.3401`
- Balanced Acc：`0.5720`
- 混淆矩阵：TP=`996` TN=`1512` FP=`106` FN=`3759`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4538` F1=`0.4132` BalAcc=`0.5416`
- `stieger_only`：Acc=`0.3896` F1=`0.3353` BalAcc=`0.5756`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`85`
- 验证最优轮次（best_epoch）：`67`
- Val 选模分数（Balanced Acc）：`0.5779`
- Val F1（最优 checkpoint 时，附报）：`0.5019`
- Val loss（最优时）：`0.7030`

**Test（overall）**
- Accuracy：`0.5502`
- Recall：`0.5756`
- Specificity：`0.4931`
- Precision：`0.7182`
- F1：`0.6391`
- Balanced Acc：`0.5343`
- 混淆矩阵：TP=`3263` TN=`1245` FP=`1280` FN=`2406`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4212` F1=`0.4259` BalAcc=`0.4780`
- `stieger_only`：Acc=`0.5562` F1=`0.6471` BalAcc=`0.5367`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`95`
- 验证最优轮次（best_epoch）：`77`
- Val 选模分数（Balanced Acc）：`0.5824`
- Val F1（最优 checkpoint 时，附报）：`0.5589`
- Val loss（最优时）：`0.6884`

**Test（overall）**
- Accuracy：`0.5563`
- Recall：`0.5372`
- Specificity：`0.6027`
- Precision：`0.7660`
- F1：`0.6315`
- Balanced Acc：`0.5699`
- 混淆矩阵：TP=`3166` TN=`1467` FP=`967` FN=`2728`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4491` F1=`0.3784` BalAcc=`0.5815`
- `stieger_only`：Acc=`0.5585` F1=`0.6353` BalAcc=`0.5695`

- 结束：`2026-08-01T23:29:58`
