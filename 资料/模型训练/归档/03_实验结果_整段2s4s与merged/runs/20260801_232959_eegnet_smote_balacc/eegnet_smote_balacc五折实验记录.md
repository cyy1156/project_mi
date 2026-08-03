# 特异度套件（20260801_232959 / eegnet_smote_balacc）

- 开始：`2026-08-01T23:29:59`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`eegnet` | 臂：`S1` — 普通CE + SMOTE
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegnet_smote_balacc\merged_2s\run_20260801_232959`

---
## 最终结论

### Task — S1
- Val BalAcc：`0.5765 ± 0.0241`
- Test Spec：`0.4238 ± 0.1889`
- Test Rec：`0.7094 ± 0.1856`
- Test BalAcc：`0.5666 ± 0.0376`
- Test F1：`0.7151 ± 0.1205`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5821`
- Val F1（最优 checkpoint 时，附报）：`0.6827`
- Val loss（最优时）：`0.6700`

**Test（overall）**
- Accuracy：`0.6526`
- Recall：`0.7740`
- Specificity：`0.3527`
- Precision：`0.7470`
- F1：`0.7603`
- Balanced Acc：`0.5634`
- 混淆矩阵：TP=`3907` TN=`721` FP=`1323` FN=`1141`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6490` F1=`0.7459` BalAcc=`0.5874`
- `stieger_only`：Acc=`0.6528` F1=`0.7611` BalAcc=`0.5617`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc）：`0.5645`
- Val F1（最优 checkpoint 时，附报）：`0.7519`
- Val loss（最优时）：`0.6466`

**Test（overall）**
- Accuracy：`0.4615`
- Recall：`0.3427`
- Specificity：`0.7643`
- Precision：`0.7876`
- F1：`0.4776`
- Balanced Acc：`0.5535`
- 混淆矩阵：TP=`1494` TN=`1307` FP=`403` FN=`2865`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3643` F1=`0.1246` BalAcc=`0.5257`
- `stieger_only`：Acc=`0.4683` F1=`0.4947` BalAcc=`0.5536`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val 选模分数（Balanced Acc）：`0.5365`
- Val F1（最优 checkpoint 时，附报）：`0.7001`
- Val loss（最优时）：`0.6857`

**Test（overall）**
- Accuracy：`0.7221`
- Recall：`0.8078`
- Specificity：`0.4703`
- Precision：`0.8176`
- F1：`0.8127`
- Balanced Acc：`0.6391`
- 混淆矩阵：TP=`3841` TN=`761` FP=`857` FN=`914`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7282` F1=`0.8274` BalAcc=`0.6019`
- `stieger_only`：Acc=`0.7217` F1=`0.8116` BalAcc=`0.6445`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.5934`
- Val F1（最优 checkpoint 时，附报）：`0.6509`
- Val loss（最优时）：`0.6875`

**Test（overall）**
- Accuracy：`0.6546`
- Recall：`0.8508`
- Specificity：`0.2143`
- Precision：`0.7085`
- F1：`0.7732`
- Balanced Acc：`0.5325`
- 混淆矩阵：TP=`4823` TN=`541` FP=`1984` FN=`846`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5109` F1=`0.5909` BalAcc=`0.5047`
- `stieger_only`：Acc=`0.6614` F1=`0.7798` BalAcc=`0.5333`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`47`
- 验证最优轮次（best_epoch）：`29`
- Val 选模分数（Balanced Acc）：`0.6057`
- Val F1（最优 checkpoint 时，附报）：`0.7538`
- Val loss（最优时）：`0.6325`

**Test（overall）**
- Accuracy：`0.6389`
- Recall：`0.7718`
- Specificity：`0.3172`
- Precision：`0.7324`
- F1：`0.7516`
- Balanced Acc：`0.5445`
- 混淆矩阵：TP=`4549` TN=`772` FP=`1662` FN=`1345`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3593` F1=`0.1440` BalAcc=`0.5388`
- `stieger_only`：Acc=`0.6447` F1=`0.7579` BalAcc=`0.5441`

- 结束：`2026-08-02T00:03:38`
