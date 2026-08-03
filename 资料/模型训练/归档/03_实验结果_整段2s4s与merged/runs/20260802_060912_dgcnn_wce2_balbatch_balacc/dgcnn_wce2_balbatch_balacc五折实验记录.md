# 特异度套件（20260802_060912 / dgcnn_wce2_balbatch_balacc）

- 开始：`2026-08-02T06:09:12`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`bandpower_cube` shape=`(36056, 8, 2)`
- backbone：`dgcnn` | 臂：`B2` — w0=2 + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dgcnn_wce2_balbatch_balacc\merged_2s\run_20260802_060912`

---
## 最终结论

### Task — B2
- Val BalAcc：`0.5010 ± 0.0005`
- Test Spec：`0.9885 ± 0.0171`
- Test Rec：`0.0102 ± 0.0167`
- Test BalAcc：`0.4993 ± 0.0004`
- Test F1：`0.0194 ± 0.0314`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.5005`
- Val F1（最优 checkpoint 时，附报）：`0.0052`
- Val loss（最优时）：`0.7758`

**Test（overall）**
- Accuracy：`0.2882`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`2044` FP=`0` FN=`5048`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3258` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2860` F1=`0.0000` BalAcc=`0.5000`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.5006`
- Val F1（最优 checkpoint 时，附报）：`0.0167`
- Val loss（最优时）：`0.7492`

**Test（overall）**
- Accuracy：`0.2818`
- Recall：`0.0007`
- Specificity：`0.9982`
- Precision：`0.5000`
- F1：`0.0014`
- Balanced Acc：`0.4995`
- 混淆矩阵：TP=`3` TN=`1707` FP=`3` FN=`4356`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3241` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2788` F1=`0.0015` BalAcc=`0.4994`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5007`
- Val F1（最优 checkpoint 时，附报）：`0.0042`
- Val loss（最优时）：`0.7430`

**Test（overall）**
- Accuracy：`0.2556`
- Recall：`0.0044`
- Specificity：`0.9938`
- Precision：`0.6774`
- F1：`0.0088`
- Balanced Acc：`0.4991`
- 混淆矩阵：TP=`21` TN=`1608` FP=`10` FN=`4734`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3282` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2509` F1=`0.0093` BalAcc=`0.4990`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5019`
- Val F1（最优 checkpoint 时，附报）：`0.0148`
- Val loss（最优时）：`0.8005`

**Test（overall）**
- Accuracy：`0.3243`
- Recall：`0.0436`
- Specificity：`0.9545`
- Precision：`0.6823`
- F1：`0.0819`
- Balanced Acc：`0.4990`
- 混淆矩阵：TP=`247` TN=`2410` FP=`115` FN=`5422`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3207` F1=`0.0079` BalAcc=`0.4936`
- `stieger_only`：Acc=`0.3244` F1=`0.0851` BalAcc=`0.4992`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.5014`
- Val F1（最优 checkpoint 时，附报）：`0.0128`
- Val loss（最优时）：`0.7419`

**Test（overall）**
- Accuracy：`0.2927`
- Recall：`0.0024`
- Specificity：`0.9959`
- Precision：`0.5833`
- F1：`0.0047`
- Balanced Acc：`0.4991`
- 混淆矩阵：TP=`14` TN=`2424` FP=`10` FN=`5880`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3054` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2925` F1=`0.0048` BalAcc=`0.4991`

- 结束：`2026-08-02T06:14:37`
