# 被试独立五折实验记录（20260731_121423 / dbn）

- 开始：`2026-07-31T12:14:23`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`dbn`（单脚本；无 registry）
- 输入：bandpower 立方体 `(36056, 8, 2)`（非时域 500）
- 结构：DBN(hidden 300/400)；监督 forward，无 RBM 预训练；drop_prob 忽略
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dbn\merged_2s\run_20260731_121423`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8299 ± 0.0125`
- Test F1：`0.8338 ± 0.0120`
- Test Acc：`0.7151 ± 0.0178`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.8450`
- Val loss（最优时）：`0.5814`

**Test（overall）**
- Accuracy：`0.7118`
- Recall：`1.0000`
- Specificity：`0.0000`
- Precision：`0.7118`
- F1：`0.8316`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`5048` TN=`0` FP=`2044` FN=`0`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6742` F1=`0.8054` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.7140` F1=`0.8331` BalAcc=`0.5000`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.8095`
- Val loss（最优时）：`0.6364`

**Test（overall）**
- Accuracy：`0.7182`
- Recall：`1.0000`
- Specificity：`0.0000`
- Precision：`0.7182`
- F1：`0.8360`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`4359` TN=`0` FP=`1710` FN=`0`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6759` F1=`0.8066` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.7212` F1=`0.8380` BalAcc=`0.5000`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.8261`
- Val loss（最优时）：`0.6189`

**Test（overall）**
- Accuracy：`0.7461`
- Recall：`1.0000`
- Specificity：`0.0000`
- Precision：`0.7461`
- F1：`0.8546`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`4755` TN=`0` FP=`1618` FN=`0`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6718` F1=`0.8037` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.7510` F1=`0.8578` BalAcc=`0.5000`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.8284`
- Val loss（最优时）：`0.6046`

**Test（overall）**
- Accuracy：`0.6918`
- Recall：`1.0000`
- Specificity：`0.0000`
- Precision：`0.6918`
- F1：`0.8179`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`5669` TN=`0` FP=`2525` FN=`0`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6766` F1=`0.8071` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.6926` F1=`0.8184` BalAcc=`0.5000`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.8406`
- Val loss（最优时）：`0.5884`

**Test（overall）**
- Accuracy：`0.7077`
- Recall：`1.0000`
- Specificity：`0.0000`
- Precision：`0.7077`
- F1：`0.8289`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`5894` TN=`0` FP=`2434` FN=`0`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6946` F1=`0.8198` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.7080` F1=`0.8290` BalAcc=`0.5000`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.3355 ± 0.0102`
- Test F1-macro：`0.3168 ± 0.0422`
- Test Acc：`0.3536 ± 0.0204`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val F1-macro（最优）：`0.3312`
- Val loss（最优时）：`1.0949`

**Test（overall）**
- Accuracy：`0.3497`
- F1-macro：`0.3153`
- Recall-macro：`0.3594`
- Recall idle/left/right：`0.4863` / `0.5156` / `0.0763`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    994    915    135
  true1   1006   1292    208
  true2   1113   1235    194
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3838` F1m=`0.3264`
- `stieger_only`：Acc=`0.3477` F1m=`0.3146`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val F1-macro（最优）：`0.3455`
- Val loss（最优时）：`1.1051`

**Test（overall）**
- Accuracy：`0.3602`
- F1-macro：`0.3515`
- Recall-macro：`0.3635`
- Recall idle/left/right：`0.4520` / `0.2083` / `0.4303`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    773    307    630
  true1    758    434    892
  true2    770    526    979
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3241` F1m=`0.1805`
- `stieger_only`：Acc=`0.3627` F1m=`0.3539`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`75`
- 验证最优轮次（best_epoch）：`57`
- Val F1-macro（最优）：`0.3221`
- Val loss（最优时）：`1.1028`

**Test（overall）**
- Accuracy：`0.3890`
- F1-macro：`0.3569`
- Recall-macro：`0.3766`
- Recall idle/left/right：`0.2441` / `0.2018` / `0.6840`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    395    254    969
  true1    213    489   1721
  true2    218    519   1595
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4154` F1m=`0.4152`
- `stieger_only`：Acc=`0.3873` F1m=`0.3508`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val F1-macro（最优）：`0.3492`
- Val loss（最优时）：`1.0926`

**Test（overall）**
- Accuracy：`0.3384`
- F1-macro：`0.2389`
- Recall-macro：`0.3298`
- Recall idle/left/right：`0.0301` / `0.1232` / `0.8359`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     76    400   2049
  true1     36    353   2476
  true2     54    406   2344
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4158` F1m=`0.4163`
- `stieger_only`：Acc=`0.3348` F1m=`0.2216`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val F1-macro（最优）：`0.3295`
- Val loss（最优时）：`1.0953`

**Test（overall）**
- Accuracy：`0.3306`
- F1-macro：`0.3216`
- Recall-macro：`0.3440`
- Recall idle/left/right：`0.5624` / `0.2296` / `0.2400`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1369    598    467
  true1   1555    676    713
  true2   1451    791    708
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3114` F1m=`0.1671`
- `stieger_only`：Acc=`0.3310` F1m=`0.3230`

### 共用超参
```json
{
  "data_tag": "merged_2s",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 18,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5
}
```

- 结束：`2026-07-31T12:20:20`
