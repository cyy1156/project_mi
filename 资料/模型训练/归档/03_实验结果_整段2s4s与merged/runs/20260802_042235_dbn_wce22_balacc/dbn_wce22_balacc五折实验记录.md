# 特异度套件（20260802_042235 / dbn_wce22_balacc）

- 开始：`2026-08-02T04:22:35`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`dbn` | 臂：`A22` — 加权CE w0=2.2 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dbn_wce22_balacc\merged_2s\run_20260802_042235`

---
## 最终结论

### Task — A22
- Val BalAcc：`0.5308 ± 0.0104`
- Test Spec：`0.3606 ± 0.1723`
- Test Rec：`0.6990 ± 0.1834`
- Test BalAcc：`0.5298 ± 0.0150`
- Test F1：`0.7022 ± 0.1057`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5412`
- Val F1（最优 checkpoint 时，附报）：`0.7970`
- Val loss（最优时）：`0.6856`

**Test（overall）**
- Accuracy：`0.6173`
- Recall：`0.7347`
- Specificity：`0.3273`
- Precision：`0.7295`
- F1：`0.7321`
- Balanced Acc：`0.5310`
- 混淆矩阵：TP=`3709` TN=`669` FP=`1375` FN=`1339`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6970` F1=`0.7966` BalAcc=`0.5990`
- `stieger_only`：Acc=`0.6126` F1=`0.7281` BalAcc=`0.5273`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`51`
- 验证最优轮次（best_epoch）：`33`
- Val 选模分数（Balanced Acc）：`0.5211`
- Val F1（最优 checkpoint 时，附报）：`0.6685`
- Val loss（最优时）：`0.6961`

**Test（overall）**
- Accuracy：`0.4755`
- Recall：`0.4136`
- Specificity：`0.6333`
- Precision：`0.7420`
- F1：`0.5312`
- Balanced Acc：`0.5235`
- 混淆矩阵：TP=`1803` TN=`1083` FP=`627` FN=`2556`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3819` F1=`0.2454` BalAcc=`0.5085`
- `stieger_only`：Acc=`0.4821` F1=`0.5456` BalAcc=`0.5226`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5232`
- Val F1（最优 checkpoint 时，附报）：`0.8089`
- Val loss（最优时）：`0.6924`

**Test（overall）**
- Accuracy：`0.7271`
- Recall：`0.9016`
- Specificity：`0.2145`
- Precision：`0.7713`
- F1：`0.8314`
- Balanced Acc：`0.5580`
- 混淆矩阵：TP=`4287` TN=`347` FP=`1271` FN=`468`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6897` F1=`0.8100` BalAcc=`0.5353`
- `stieger_only`：Acc=`0.7296` F1=`0.8328` BalAcc=`0.5611`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc）：`0.5455`
- Val F1（最优 checkpoint 时，附报）：`0.5403`
- Val loss（最优时）：`0.6907`

**Test（overall）**
- Accuracy：`0.6511`
- Recall：`0.8696`
- Specificity：`0.1604`
- Precision：`0.6993`
- F1：`0.7752`
- Balanced Acc：`0.5150`
- 混淆矩阵：TP=`4930` TN=`405` FP=`2120` FN=`739`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5245` F1=`0.5592` BalAcc=`0.5674`
- `stieger_only`：Acc=`0.6570` F1=`0.7822` BalAcc=`0.5117`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val 选模分数（Balanced Acc）：`0.5228`
- Val F1（最优 checkpoint 时，附报）：`0.8115`
- Val loss（最优时）：`0.6851`

**Test（overall）**
- Accuracy：`0.5439`
- Recall：`0.5755`
- Specificity：`0.4675`
- Precision：`0.7235`
- F1：`0.6411`
- Balanced Acc：`0.5215`
- 混淆矩阵：TP=`3392` TN=`1138` FP=`1296` FN=`2502`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3114` F1=`0.0171` BalAcc=`0.5043`
- `stieger_only`：Acc=`0.5487` F1=`0.6481` BalAcc=`0.5215`

- 结束：`2026-08-02T04:26:22`
