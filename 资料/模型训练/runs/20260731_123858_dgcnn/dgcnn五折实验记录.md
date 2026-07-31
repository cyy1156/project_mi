# 被试独立五折实验记录（20260731_123858 / dgcnn）

- 开始：`2026-07-31T12:38:58`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`dgcnn`（单脚本；无 registry）
- 输入：bandpower 立方体 `(36056, 8, 2)`（非时域 500）
- 结构：DGCNN(k=2, layers=[128], dropout=shared drop_prob)
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dgcnn\merged_2s\run_20260731_123858`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8302 ± 0.0128`
- Test F1：`0.8317 ± 0.0133`
- Test Acc：`0.7135 ± 0.0193`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val F1（最优）：`0.8462`
- Val loss（最优时）：`0.5924`

**Test（overall）**
- Accuracy：`0.7033`
- Recall：`0.9693`
- Specificity：`0.0465`
- Precision：`0.7151`
- F1：`0.8230`
- Balanced Acc：`0.5079`
- 混淆矩阵：TP=`4893` TN=`95` FP=`1949` FN=`155`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6768` F1=`0.8061` BalAcc=`0.5059`
- `stieger_only`：Acc=`0.7049` F1=`0.8240` BalAcc=`0.5082`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.8095`
- Val loss（最优时）：`0.6472`

**Test（overall）**
- Accuracy：`0.7194`
- Recall：`1.0000`
- Specificity：`0.0041`
- Precision：`0.7191`
- F1：`0.8366`
- Balanced Acc：`0.5020`
- 混淆矩阵：TP=`4359` TN=`7` FP=`1703` FN=`0`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6910` F1=`0.8139` BalAcc=`0.5233`
- `stieger_only`：Acc=`0.7214` F1=`0.8381` BalAcc=`0.5003`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val F1（最优）：`0.8262`
- Val loss（最优时）：`0.6103`

**Test（overall）**
- Accuracy：`0.7478`
- Recall：`1.0000`
- Specificity：`0.0068`
- Precision：`0.7474`
- F1：`0.8554`
- Balanced Acc：`0.5034`
- 混淆矩阵：TP=`4755` TN=`11` FP=`1607` FN=`0`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6744` F1=`0.8049` BalAcc=`0.5039`
- `stieger_only`：Acc=`0.7526` F1=`0.8586` BalAcc=`0.5034`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.8284`
- Val loss（最优时）：`0.6010`

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

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val F1（最优）：`0.8407`
- Val loss（最优时）：`0.5829`

**Test（overall）**
- Accuracy：`0.7049`
- Recall：`0.9880`
- Specificity：`0.0193`
- Precision：`0.7093`
- F1：`0.8257`
- Balanced Acc：`0.5036`
- 混淆矩阵：TP=`5823` TN=`47` FP=`2387` FN=`71`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6707` F1=`0.7368` BalAcc=`0.6750`
- `stieger_only`：Acc=`0.7056` F1=`0.8271` BalAcc=`0.4997`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.3713 ± 0.0181`
- Test F1-macro：`0.3608 ± 0.0228`
- Test Acc：`0.3758 ± 0.0261`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`13`
- Val F1-macro（最优）：`0.4026`
- Val loss（最优时）：`1.0865`

**Test（overall）**
- Accuracy：`0.3600`
- F1-macro：`0.3576`
- Recall-macro：`0.3684`
- Recall idle/left/right：`0.4892` / `0.3520` / `0.2640`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1000    613    431
  true1   1039    882    585
  true2   1044    827    671
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4242` F1m=`0.4250`
- `stieger_only`：Acc=`0.3562` F1m=`0.3535`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val F1-macro（最优）：`0.3624`
- Val loss（最优时）：`1.0908`

**Test（overall）**
- Accuracy：`0.3793`
- F1-macro：`0.3792`
- Recall-macro：`0.3830`
- Recall idle/left/right：`0.4240` / `0.3805` / `0.3446`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    725    511    474
  true1    753    793    538
  true2    706    785    784
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4598` F1m=`0.4357`
- `stieger_only`：Acc=`0.3737` F1m=`0.3730`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val F1-macro（最优）：`0.3546`
- Val loss（最优时）：`1.0943`

**Test（overall）**
- Accuracy：`0.4251`
- F1-macro：`0.3920`
- Recall-macro：`0.3996`
- Recall idle/left/right：`0.1768` / `0.4358` / `0.5862`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    286    607    725
  true1    120   1056   1247
  true2    157    808   1367
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3795` F1m=`0.3735`
- `stieger_only`：Acc=`0.4280` F1m=`0.3922`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`60`
- 验证最优轮次（best_epoch）：`42`
- Val F1-macro（最优）：`0.3805`
- Val loss（最优时）：`1.0836`

**Test（overall）**
- Accuracy：`0.3612`
- F1-macro：`0.3472`
- Recall-macro：`0.3555`
- Recall idle/left/right：`0.1945` / `0.3937` / `0.4782`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    491    884   1150
  true1    394   1128   1343
  true2    417   1046   1341
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4402` F1m=`0.4417`
- `stieger_only`：Acc=`0.3575` F1m=`0.3415`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val F1-macro（最优）：`0.3564`
- Val loss（最优时）：`1.0832`

**Test（overall）**
- Accuracy：`0.3535`
- F1-macro：`0.3279`
- Recall-macro：`0.3410`
- Recall idle/left/right：`0.1393` / `0.3950` / `0.4888`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    339    916   1179
  true1    405   1163   1376
  true2    340   1168   1442
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4192` F1m=`0.3559`
- `stieger_only`：Acc=`0.3522` F1m=`0.3226`

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

- 结束：`2026-07-31T12:51:45`
