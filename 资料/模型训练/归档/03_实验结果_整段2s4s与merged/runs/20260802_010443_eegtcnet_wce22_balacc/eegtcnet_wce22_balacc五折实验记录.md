# 特异度套件（20260802_010443 / eegtcnet_wce22_balacc）

- 开始：`2026-08-02T01:04:43`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`eegtcnet` | 臂：`A22` — 加权CE w0=2.2 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegtcnet_wce22_balacc\merged_2s\run_20260802_010443`

---
## 最终结论

### Task — A22
- Val BalAcc：`0.5810 ± 0.0317`
- Test Spec：`0.3536 ± 0.1758`
- Test Rec：`0.7286 ± 0.1444`
- Test BalAcc：`0.5411 ± 0.0165`
- Test F1：`0.7256 ± 0.0684`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`87`
- 验证最优轮次（best_epoch）：`69`
- Val 选模分数（Balanced Acc）：`0.6272`
- Val F1（最优 checkpoint 时，附报）：`0.6071`
- Val loss（最优时）：`0.6656`

**Test（overall）**
- Accuracy：`0.5823`
- Recall：`0.6147`
- Specificity：`0.5024`
- Precision：`0.7532`
- F1：`0.6769`
- Balanced Acc：`0.5586`
- 混淆矩阵：TP=`3103` TN=`1027` FP=`1017` FN=`1945`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6540` F1=`0.7243` BalAcc=`0.6433`
- `stieger_only`：Acc=`0.5781` F1=`0.6742` BalAcc=`0.5532`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`81`
- 验证最优轮次（best_epoch）：`63`
- Val 选模分数（Balanced Acc）：`0.5662`
- Val F1（最优 checkpoint 时，附报）：`0.7738`
- Val loss（最优时）：`0.6808`

**Test（overall）**
- Accuracy：`0.5406`
- Recall：`0.5109`
- Specificity：`0.6164`
- Precision：`0.7725`
- F1：`0.6150`
- Balanced Acc：`0.5636`
- 混淆矩阵：TP=`2227` TN=`1054` FP=`656` FN=`2132`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3995` F1=`0.2265` BalAcc=`0.5457`
- `stieger_only`：Acc=`0.5505` F1=`0.6323` BalAcc=`0.5621`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5314`
- Val F1（最优 checkpoint 时，附报）：`0.7010`
- Val loss（最优时）：`0.6903`

**Test（overall）**
- Accuracy：`0.6465`
- Recall：`0.7699`
- Specificity：`0.2837`
- Precision：`0.7595`
- F1：`0.7647`
- Balanced Acc：`0.5268`
- 混淆矩阵：TP=`3661` TN=`459` FP=`1159` FN=`1094`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6897` F1=`0.8039` BalAcc=`0.5553`
- `stieger_only`：Acc=`0.6437` F1=`0.7620` BalAcc=`0.5268`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`59`
- 验证最优轮次（best_epoch）：`41`
- Val 选模分数（Balanced Acc）：`0.5943`
- Val F1（最优 checkpoint 时，附报）：`0.7680`
- Val loss（最优时）：`0.6571`

**Test（overall）**
- Accuracy：`0.6612`
- Recall：`0.8714`
- Specificity：`0.1893`
- Precision：`0.7070`
- F1：`0.7807`
- Balanced Acc：`0.5304`
- 混淆矩阵：TP=`4940` TN=`478` FP=`2047` FN=`729`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6196` F1=`0.7388` BalAcc=`0.5236`
- `stieger_only`：Acc=`0.6632` F1=`0.7825` BalAcc=`0.5306`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`39`
- 验证最优轮次（best_epoch）：`21`
- Val 选模分数（Balanced Acc）：`0.5861`
- Val F1（最优 checkpoint 时，附报）：`0.8142`
- Val loss（最优时）：`0.6569`

**Test（overall）**
- Accuracy：`0.6716`
- Recall：`0.8761`
- Specificity：`0.1763`
- Precision：`0.7203`
- F1：`0.7906`
- Balanced Acc：`0.5262`
- 混淆矩阵：TP=`5164` TN=`429` FP=`2005` FN=`730`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3293` F1=`0.1765` BalAcc=`0.4733`
- `stieger_only`：Acc=`0.6786` F1=`0.7971` BalAcc=`0.5268`

- 结束：`2026-08-02T01:29:29`
