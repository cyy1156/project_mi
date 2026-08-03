# 特异度套件（20260802_025007 / conformer_wce22_balacc）

- 开始：`2026-08-02T02:50:07`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`conformer` | 臂：`A22` — 加权CE w0=2.2 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\conformer_wce22_balacc\merged_2s\run_20260802_025007`

---
## 最终结论

### Task — A22
- Val BalAcc：`0.5881 ± 0.0250`
- Test Spec：`0.4002 ± 0.1921`
- Test Rec：`0.7208 ± 0.1890`
- Test BalAcc：`0.5605 ± 0.0354`
- Test F1：`0.7208 ± 0.1118`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.6275`
- Val F1（最优 checkpoint 时，附报）：`0.7121`
- Val loss（最优时）：`0.6472`

**Test（overall）**
- Accuracy：`0.6036`
- Recall：`0.6472`
- Specificity：`0.4961`
- Precision：`0.7603`
- F1：`0.6992`
- Balanced Acc：`0.5716`
- 混淆矩阵：TP=`3267` TN=`1014` FP=`1030` FN=`1781`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6338` F1=`0.7047` BalAcc=`0.6263`
- `stieger_only`：Acc=`0.6019` F1=`0.6989` BalAcc=`0.5680`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.5520`
- Val F1（最优 checkpoint 时，附报）：`0.7685`
- Val loss（最优时）：`0.7633`

**Test（overall）**
- Accuracy：`0.4825`
- Recall：`0.3923`
- Specificity：`0.7123`
- Precision：`0.7766`
- F1：`0.5213`
- Balanced Acc：`0.5523`
- 混淆矩阵：TP=`1710` TN=`1218` FP=`492` FN=`2649`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3995` F1=`0.2164` BalAcc=`0.5497`
- `stieger_only`：Acc=`0.4883` F1=`0.5361` BalAcc=`0.5504`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.5809`
- Val F1（最优 checkpoint 时，附报）：`0.7337`
- Val loss（最优时）：`0.6722`

**Test（overall）**
- Accuracy：`0.7552`
- Recall：`0.8915`
- Specificity：`0.3548`
- Precision：`0.8024`
- F1：`0.8446`
- Balanced Acc：`0.6231`
- 混淆矩阵：TP=`4239` TN=`574` FP=`1044` FN=`516`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7385` F1=`0.8371` BalAcc=`0.6016`
- `stieger_only`：Acc=`0.7563` F1=`0.8451` BalAcc=`0.6265`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.6004`
- Val F1（最优 checkpoint 时，附报）：`0.6618`
- Val loss（最优时）：`0.6895`

**Test（overall）**
- Accuracy：`0.6209`
- Recall：`0.7684`
- Specificity：`0.2899`
- Precision：`0.7084`
- F1：`0.7372`
- Balanced Acc：`0.5291`
- 混淆矩阵：TP=`4356` TN=`732` FP=`1793` FN=`1313`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5408` F1=`0.6079` BalAcc=`0.5488`
- `stieger_only`：Acc=`0.6247` F1=`0.7421` BalAcc=`0.5277`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5799`
- Val F1（最优 checkpoint 时，附报）：`0.7966`
- Val loss（最优时）：`0.6681`

**Test（overall）**
- Accuracy：`0.6836`
- Recall：`0.9048`
- Specificity：`0.1479`
- Precision：`0.7200`
- F1：`0.8019`
- Balanced Acc：`0.5264`
- 混淆矩阵：TP=`5333` TN=`360` FP=`2074` FN=`561`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4910` F1=`0.4970` BalAcc=`0.5732`
- `stieger_only`：Acc=`0.6875` F1=`0.8058` BalAcc=`0.5250`

- 结束：`2026-08-02T03:05:46`
