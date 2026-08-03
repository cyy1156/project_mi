# 特异度套件（20260802_055446 / dgcnn_wce2_balacc）

- 开始：`2026-08-02T05:54:46`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`dgcnn` | 臂：`A20` — 加权CE w0=2.0 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dgcnn_wce2_balacc\merged_2s\run_20260802_055446`

---
## 最终结论

### Task — A20
- Val BalAcc：`0.5459 ± 0.0179`
- Test Spec：`0.3457 ± 0.1839`
- Test Rec：`0.7297 ± 0.1758`
- Test BalAcc：`0.5377 ± 0.0221`
- Test F1：`0.7215 ± 0.0916`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`55`
- 验证最优轮次（best_epoch）：`37`
- Val 选模分数（Balanced Acc）：`0.5813`
- Val F1（最优 checkpoint 时，附报）：`0.7762`
- Val loss（最优时）：`0.6772`

**Test（overall）**
- Accuracy：`0.5495`
- Recall：`0.5596`
- Specificity：`0.5245`
- Precision：`0.7440`
- F1：`0.6388`
- Balanced Acc：`0.5420`
- 混淆矩阵：TP=`2825` TN=`1072` FP=`972` FN=`2223`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6944` F1=`0.7866` BalAcc=`0.6192`
- `stieger_only`：Acc=`0.5409` F1=`0.6287` BalAcc=`0.5384`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val 选模分数（Balanced Acc）：`0.5336`
- Val F1（最优 checkpoint 时，附报）：`0.6989`
- Val loss（最优时）：`0.6896`

**Test（overall）**
- Accuracy：`0.5169`
- Recall：`0.4857`
- Specificity：`0.5965`
- Precision：`0.7542`
- F1：`0.5908`
- Balanced Acc：`0.5411`
- 混淆矩阵：TP=`2117` TN=`1020` FP=`690` FN=`2242`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4045` F1=`0.2752` BalAcc=`0.5333`
- `stieger_only`：Acc=`0.5248` F1=`0.6059` BalAcc=`0.5392`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val 选模分数（Balanced Acc）：`0.5391`
- Val F1（最优 checkpoint 时，附报）：`0.7621`
- Val loss（最优时）：`0.6848`

**Test（overall）**
- Accuracy：`0.7251`
- Recall：`0.8803`
- Specificity：`0.2689`
- Precision：`0.7797`
- F1：`0.8269`
- Balanced Acc：`0.5746`
- 混淆矩阵：TP=`4186` TN=`435` FP=`1183` FN=`569`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6821` F1=`0.7817` BalAcc=`0.5955`
- `stieger_only`：Acc=`0.7279` F1=`0.8296` BalAcc=`0.5723`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.5400`
- Val F1（最优 checkpoint 时，附报）：`0.6480`
- Val loss（最优时）：`0.6900`

**Test（overall）**
- Accuracy：`0.6765`
- Recall：`0.9271`
- Specificity：`0.1137`
- Precision：`0.7014`
- F1：`0.7986`
- Balanced Acc：`0.5204`
- 混淆矩阵：TP=`5256` TN=`287` FP=`2238` FN=`413`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6332` F1=`0.7109` BalAcc=`0.6148`
- `stieger_only`：Acc=`0.6785` F1=`0.8018` BalAcc=`0.5153`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5354`
- Val F1（最优 checkpoint 时，附报）：`0.8197`
- Val loss（最优时）：`0.6722`

**Test（overall）**
- Accuracy：`0.6290`
- Recall：`0.7957`
- Specificity：`0.2251`
- Precision：`0.7132`
- F1：`0.7522`
- Balanced Acc：`0.5104`
- 混淆矩阵：TP=`4690` TN=`548` FP=`1886` FN=`1204`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3293` F1=`0.0667` BalAcc=`0.5172`
- `stieger_only`：Acc=`0.6351` F1=`0.7589` BalAcc=`0.5098`

- 结束：`2026-08-02T06:02:25`
