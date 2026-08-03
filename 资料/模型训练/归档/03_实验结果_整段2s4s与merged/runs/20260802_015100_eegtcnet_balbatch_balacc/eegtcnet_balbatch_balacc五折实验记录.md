# 特异度套件（20260802_015100 / eegtcnet_balbatch_balacc）

- 开始：`2026-08-02T01:51:00`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`eegtcnet` | 臂：`B1` — 普通CE + batch balance
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegtcnet_balbatch_balacc\merged_2s\run_20260802_015100`

---
## 最终结论

### Task — B1
- Val BalAcc：`0.5960 ± 0.0351`
- Test Spec：`0.4438 ± 0.1802`
- Test Rec：`0.6825 ± 0.1540`
- Test BalAcc：`0.5632 ± 0.0244`
- Test F1：`0.7061 ± 0.0777`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`72`
- 验证最优轮次（best_epoch）：`54`
- Val 选模分数（Balanced Acc）：`0.6378`
- Val F1（最优 checkpoint 时，附报）：`0.6404`
- Val loss（最优时）：`0.6930`

**Test（overall）**
- Accuracy：`0.5750`
- Recall：`0.5935`
- Specificity：`0.5294`
- Precision：`0.7569`
- F1：`0.6653`
- Balanced Acc：`0.5614`
- 混淆矩阵：TP=`2996` TN=`1082` FP=`962` FN=`2052`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6515` F1=`0.7206` BalAcc=`0.6434`
- `stieger_only`：Acc=`0.5705` F1=`0.6621` BalAcc=`0.5563`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`113`
- 验证最优轮次（best_epoch）：`95`
- Val 选模分数（Balanced Acc）：`0.5753`
- Val F1（最优 checkpoint 时，附报）：`0.7717`
- Val loss（最优时）：`0.6398`

**Test（overall）**
- Accuracy：`0.5162`
- Recall：`0.4591`
- Specificity：`0.6620`
- Precision：`0.7759`
- F1：`0.5768`
- Balanced Acc：`0.5605`
- 混淆矩阵：TP=`2001` TN=`1132` FP=`578` FN=`2358`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4347` F1=`0.3034` BalAcc=`0.5717`
- `stieger_only`：Acc=`0.5220` F1=`0.5902` BalAcc=`0.5574`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5383`
- Val F1（最优 checkpoint 时，附报）：`0.5940`
- Val loss（最优时）：`0.6938`

**Test（overall）**
- Accuracy：`0.6297`
- Recall：`0.6526`
- Specificity：`0.5624`
- Precision：`0.8142`
- F1：`0.7245`
- Balanced Acc：`0.6075`
- 混淆矩阵：TP=`3103` TN=`910` FP=`708` FN=`1652`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6949` F1=`0.7938` BalAcc=`0.6011`
- `stieger_only`：Acc=`0.6254` F1=`0.7195` BalAcc=`0.6111`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`39`
- 验证最优轮次（best_epoch）：`21`
- Val 选模分数（Balanced Acc）：`0.6133`
- Val F1（最优 checkpoint 时，附报）：`0.7699`
- Val loss（最优时）：`0.6554`

**Test（overall）**
- Accuracy：`0.6654`
- Recall：`0.8781`
- Specificity：`0.1877`
- Precision：`0.7082`
- F1：`0.7841`
- Balanced Acc：`0.5329`
- 混淆矩阵：TP=`4978` TN=`474` FP=`2051` FN=`691`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5109` F1=`0.6371` BalAcc=`0.4433`
- `stieger_only`：Acc=`0.6726` F1=`0.7900` BalAcc=`0.5369`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`89`
- 验证最优轮次（best_epoch）：`71`
- Val 选模分数（Balanced Acc）：`0.6152`
- Val F1（最优 checkpoint 时，附报）：`0.7943`
- Val loss（最优时）：`0.6222`

**Test（overall）**
- Accuracy：`0.6681`
- Recall：`0.8295`
- Specificity：`0.2773`
- Precision：`0.7354`
- F1：`0.7796`
- Balanced Acc：`0.5534`
- 混淆矩阵：TP=`4889` TN=`675` FP=`1759` FN=`1005`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4611` F1=`0.4304` BalAcc=`0.5681`
- `stieger_only`：Acc=`0.6723` F1=`0.7841` BalAcc=`0.5527`

- 结束：`2026-08-02T02:20:14`
