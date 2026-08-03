# 特异度套件（20260802_050014 / gcbnet_wce2_balacc）

- 开始：`2026-08-02T05:00:14`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`gcbnet` | 臂：`A20` — 加权CE w0=2.0 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\gcbnet_wce2_balacc\merged_2s\run_20260802_050014`

---
## 最终结论

### Task — A20
- Val BalAcc：`0.5484 ± 0.0205`
- Test Spec：`0.4599 ± 0.2368`
- Test Rec：`0.6128 ± 0.2561`
- Test BalAcc：`0.5363 ± 0.0332`
- Test F1：`0.6355 ± 0.1828`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`49`
- 验证最优轮次（best_epoch）：`31`
- Val 选模分数（Balanced Acc）：`0.5841`
- Val F1（最优 checkpoint 时，附报）：`0.6914`
- Val loss（最优时）：`0.6763`

**Test（overall）**
- Accuracy：`0.4973`
- Recall：`0.4427`
- Specificity：`0.6321`
- Precision：`0.7482`
- F1：`0.5563`
- Balanced Acc：`0.5374`
- 混淆矩阵：TP=`2235` TN=`1292` FP=`752` FN=`2813`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6364` F1=`0.7188` BalAcc=`0.6081`
- `stieger_only`：Acc=`0.4891` F1=`0.5453` BalAcc=`0.5341`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`69`
- 验证最优轮次（best_epoch）：`51`
- Val 选模分数（Balanced Acc）：`0.5247`
- Val F1（最优 checkpoint 时，附报）：`0.5106`
- Val loss（最优时）：`0.7055`

**Test（overall）**
- Accuracy：`0.3763`
- Recall：`0.2007`
- Specificity：`0.8240`
- Precision：`0.7440`
- F1：`0.3162`
- Balanced Acc：`0.5124`
- 混淆矩阵：TP=`875` TN=`1409` FP=`301` FN=`3484`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4146` F1=`0.3003` BalAcc=`0.5387`
- `stieger_only`：Acc=`0.3737` F1=`0.3172` BalAcc=`0.5101`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val 选模分数（Balanced Acc）：`0.5332`
- Val F1（最优 checkpoint 时，附报）：`0.7178`
- Val loss（最优时）：`0.6874`

**Test（overall）**
- Accuracy：`0.7188`
- Recall：`0.8412`
- Specificity：`0.3591`
- Precision：`0.7941`
- F1：`0.8170`
- Balanced Acc：`0.6002`
- 混淆矩阵：TP=`4000` TN=`581` FP=`1037` FN=`755`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6615` F1=`0.7462` BalAcc=`0.6202`
- `stieger_only`：Acc=`0.7225` F1=`0.8210` BalAcc=`0.5970`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc）：`0.5462`
- Val F1（最优 checkpoint 时，附报）：`0.6060`
- Val loss（最优时）：`0.6941`

**Test（overall）**
- Accuracy：`0.6550`
- Recall：`0.8751`
- Specificity：`0.1608`
- Precision：`0.7007`
- F1：`0.7783`
- Balanced Acc：`0.5180`
- 混淆矩阵：TP=`4961` TN=`406` FP=`2119` FN=`708`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6168` F1=`0.6846` BalAcc=`0.6182`
- `stieger_only`：Acc=`0.6568` F1=`0.7817` BalAcc=`0.5125`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`55`
- 验证最优轮次（best_epoch）：`37`
- Val 选模分数（Balanced Acc）：`0.5539`
- Val F1（最优 checkpoint 时，附报）：`0.7935`
- Val loss（最优时）：`0.6714`

**Test（overall）**
- Accuracy：`0.5928`
- Recall：`0.7041`
- Specificity：`0.3233`
- Precision：`0.7159`
- F1：`0.7099`
- Balanced Acc：`0.5137`
- 混淆矩阵：TP=`4150` TN=`787` FP=`1647` FN=`1744`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4012` F1=`0.2754` BalAcc=`0.5525`
- `stieger_only`：Acc=`0.5967` F1=`0.7151` BalAcc=`0.5125`

- 结束：`2026-08-02T05:13:30`
