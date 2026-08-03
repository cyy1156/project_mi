# 特异度套件（20260802_000339 / deep_wce22_balacc）

- 开始：`2026-08-02T00:03:39`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s` feat=`raw_temporal` shape=`(36056, 8, 500)`
- backbone：`deep` | 臂：`A22` — 加权CE w0=2.2 + BalAcc
- 仅 Task；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\deep_wce22_balacc\merged_2s\run_20260802_000339`

---
## 最终结论

### Task — A22
- Val BalAcc：`0.5141 ± 0.0279`
- Test Spec：`0.9702 ± 0.0565`
- Test Rec：`0.0358 ± 0.0689`
- Test BalAcc：`0.5030 ± 0.0062`
- Test F1：`0.0586 ± 0.1118`
- 过关：`False`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5005`
- Val F1（最优 checkpoint 时，附报）：`0.0097`
- Val loss（最优时）：`0.7598`

**Test（overall）**
- Accuracy：`0.2903`
- Recall：`0.0055`
- Specificity：`0.9936`
- Precision：`0.6829`
- F1：`0.0110`
- Balanced Acc：`0.4996`
- 混淆矩阵：TP=`28` TN=`2031` FP=`13` FN=`5020`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3232` F1=`0.0000` BalAcc=`0.4961`
- `stieger_only`：Acc=`0.2884` F1=`0.0116` BalAcc=`0.4998`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`139`
- 验证最优轮次（best_epoch）：`121`
- Val 选模分数（Balanced Acc）：`0.5699`
- Val F1（最优 checkpoint 时，附报）：`0.6242`
- Val loss（最优时）：`0.6950`

**Test（overall）**
- Accuracy：`0.3661`
- Recall：`0.1734`
- Specificity：`0.8573`
- Precision：`0.7560`
- F1：`0.2821`
- Balanced Acc：`0.5154`
- 混淆矩阵：TP=`756` TN=`1466` FP=`244` FN=`3603`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3643` F1=`0.1246` BalAcc=`0.5257`
- `stieger_only`：Acc=`0.3662` F1=`0.2911` BalAcc=`0.5137`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.8706`

**Test（overall）**
- Accuracy：`0.2539`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`1618` FP=`0` FN=`4755`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3282` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2490` F1=`0.0000` BalAcc=`0.5000`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.9551`

**Test（overall）**
- Accuracy：`0.3082`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`2525` FP=`0` FN=`5669`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3234` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.3074` F1=`0.0000` BalAcc=`0.5000`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.8957`

**Test（overall）**
- Accuracy：`0.2923`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`2434` FP=`0` FN=`5894`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3054` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2920` F1=`0.0000` BalAcc=`0.5000`

- 结束：`2026-08-02T00:16:22`
