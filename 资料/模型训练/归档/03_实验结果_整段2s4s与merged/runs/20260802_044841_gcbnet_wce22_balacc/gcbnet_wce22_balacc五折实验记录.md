# 特异度套件（20260802_044841 / gcbnet_wce22_balacc）

- 开始：`2026-08-02T04:48:41`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`gcbnet` | 臂：`A22` — 加权CE w0=2.2 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\gcbnet_wce22_balacc\merged_2s\run_20260802_044841`

---
## 最终结论

### Task — A22
- Val BalAcc：`0.5511 ± 0.0184`
- Test Spec：`0.4725 ± 0.2699`
- Test Rec：`0.5985 ± 0.2866`
- Test BalAcc：`0.5355 ± 0.0296`
- Test F1：`0.6196 ± 0.1936`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val 选模分数（Balanced Acc）：`0.5800`
- Val F1（最优 checkpoint 时，附报）：`0.6017`
- Val loss（最优时）：`0.6815`

**Test（overall）**
- Accuracy：`0.4474`
- Recall：`0.3269`
- Specificity：`0.7451`
- Precision：`0.7600`
- F1：`0.4571`
- Balanced Acc：`0.5360`
- 混淆矩阵：TP=`1650` TN=`1523` FP=`521` FN=`3398`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5909` F1=`0.6463` BalAcc=`0.6105`
- `stieger_only`：Acc=`0.4389` F1=`0.4443` BalAcc=`0.5323`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`61`
- 验证最优轮次（best_epoch）：`43`
- Val 选模分数（Balanced Acc）：`0.5268`
- Val F1（最优 checkpoint 时，附报）：`0.5255`
- Val loss（最优时）：`0.7030`

**Test（overall）**
- Accuracy：`0.3826`
- Recall：`0.2140`
- Specificity：`0.8123`
- Precision：`0.7440`
- F1：`0.3324`
- Balanced Acc：`0.5132`
- 混淆矩阵：TP=`933` TN=`1389` FP=`321` FN=`3426`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4196` F1=`0.3104` BalAcc=`0.5424`
- `stieger_only`：Acc=`0.3800` F1=`0.3338` BalAcc=`0.5106`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`13`
- Val 选模分数（Balanced Acc）：`0.5379`
- Val F1（最优 checkpoint 时，附报）：`0.7117`
- Val loss（最优时）：`0.6882`

**Test（overall）**
- Accuracy：`0.7194`
- Recall：`0.8503`
- Specificity：`0.3350`
- Precision：`0.7898`
- F1：`0.8189`
- Balanced Acc：`0.5926`
- 混淆矩阵：TP=`4043` TN=`542` FP=`1076` FN=`712`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6513` F1=`0.7344` BalAcc=`0.6166`
- `stieger_only`：Acc=`0.7239` F1=`0.8235` BalAcc=`0.5887`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5499`
- Val F1（最优 checkpoint 时，附报）：`0.6650`
- Val loss（最优时）：`0.6867`

**Test（overall）**
- Accuracy：`0.6839`
- Recall：`0.9490`
- Specificity：`0.0887`
- Precision：`0.7004`
- F1：`0.8060`
- Balanced Acc：`0.5189`
- 混淆矩阵：TP=`5380` TN=`224` FP=`2301` FN=`289`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6168` F1=`0.6773` BalAcc=`0.6291`
- `stieger_only`：Acc=`0.6871` F1=`0.8103` BalAcc=`0.5128`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`60`
- 验证最优轮次（best_epoch）：`42`
- Val 选模分数（Balanced Acc）：`0.5609`
- Val F1（最优 checkpoint 时，附报）：`0.7606`
- Val loss（最优时）：`0.6787`

**Test（overall）**
- Accuracy：`0.5730`
- Recall：`0.6522`
- Specificity：`0.3813`
- Precision：`0.7185`
- F1：`0.6837`
- Balanced Acc：`0.5167`
- 混淆矩阵：TP=`3844` TN=`928` FP=`1506` FN=`2050`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4731` F1=`0.4430` BalAcc=`0.5822`
- `stieger_only`：Acc=`0.5751` F1=`0.6872` BalAcc=`0.5151`

- 结束：`2026-08-02T04:59:33`
