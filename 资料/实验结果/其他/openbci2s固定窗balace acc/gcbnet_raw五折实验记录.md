# 被试独立五折实验记录（20260805_194852 / gcbnet_raw）

- 开始：`2026-08-05T19:48:52`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`gcbnet_raw`（TemporalEncoder + GCBNet）
- 输入：raw `(21600, 8, 500)` → Encoder → 节点特征 D=64（非 bandpower）
- 结构：GCBNetRaw(k=2, layers=[128], in_channels=64, relu_is=1)
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\gcbnet_raw\openbmi_balanced_train_2s\run_20260805_194852`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.6061 ± 0.0436`
- Test F1：`0.5975 ± 0.0393`
- Test Acc：`0.5878 ± 0.0222`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`129`
- 验证最优轮次（best_epoch）：`111`
- Val F1（最优）：`0.6006`
- Val loss（最优时）：`0.6893`

**Test（overall）**
- Accuracy：`0.5736`
- Recall：`0.5368`
- Specificity：`0.6105`
- Precision：`0.5795`
- F1：`0.5573`
- Balanced Acc：`0.5736`
- 混淆矩阵：TP=`1181` TN=`1343` FP=`857` FN=`1019`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`144`
- 验证最优轮次（best_epoch）：`126`
- Val F1（最优）：`0.6381`
- Val loss（最优时）：`0.6996`

**Test（overall）**
- Accuracy：`0.6155`
- Recall：`0.5732`
- Specificity：`0.6577`
- Precision：`0.6261`
- F1：`0.5985`
- Balanced Acc：`0.6155`
- 混淆矩阵：TP=`1261` TN=`1447` FP=`753` FN=`939`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`94`
- 验证最优轮次（best_epoch）：`76`
- Val F1（最优）：`0.6372`
- Val loss（最优时）：`0.7272`

**Test（overall）**
- Accuracy：`0.5843`
- Recall：`0.7723`
- Specificity：`0.3964`
- Precision：`0.5613`
- F1：`0.6501`
- Balanced Acc：`0.5843`
- 混淆矩阵：TP=`1699` TN=`872` FP=`1328` FN=`501`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`80`
- 验证最优轮次（best_epoch）：`62`
- Val F1（最优）：`0.6312`
- Val loss（最优时）：`0.6608`

**Test（overall）**
- Accuracy：`0.6098`
- Recall：`0.6673`
- Specificity：`0.5523`
- Precision：`0.5985`
- F1：`0.6310`
- Balanced Acc：`0.6098`
- 混淆矩阵：TP=`1468` TN=`1215` FP=`985` FN=`732`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val F1（最优）：`0.5234`
- Val loss（最优时）：`0.7579`

**Test（overall）**
- Accuracy：`0.5560`
- Recall：`0.5435`
- Specificity：`0.5685`
- Precision：`0.5574`
- F1：`0.5504`
- Balanced Acc：`0.5560`
- 混淆矩阵：TP=`1087` TN=`1137` FP=`863` FN=`913`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4413 ± 0.0169`
- Test F1-macro：`0.4264 ± 0.0192`
- Test Acc：`0.4889 ± 0.0267`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`103`
- 验证最优轮次（best_epoch）：`85`
- Val F1-macro（最优）：`0.4560`
- Val loss（最优时）：`1.0064`

**Test（overall）**
- Accuracy：`0.4845`
- F1-macro：`0.4484`
- Recall-macro：`0.4480`
- Recall idle/left/right：`0.5941` / `0.3555` / `0.3945`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1307    392    501
  true1    469    391    240
  true2    458    208    434
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`77`
- 验证最优轮次（best_epoch）：`59`
- Val F1-macro（最优）：`0.4333`
- Val loss（最优时）：`1.0010`

**Test（overall）**
- Accuracy：`0.5368`
- F1-macro：`0.4090`
- Recall-macro：`0.4241`
- Recall idle/left/right：`0.8750` / `0.2555` / `0.1418`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1925    143    132
  true1    710    281    109
  true2    781    163    156
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`46`
- 验证最优轮次（best_epoch）：`28`
- Val F1-macro（最优）：`0.4371`
- Val loss（最优时）：`0.9971`

**Test（overall）**
- Accuracy：`0.4627`
- F1-macro：`0.3987`
- Recall-macro：`0.3991`
- Recall idle/left/right：`0.6536` / `0.2945` / `0.2491`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1438    433    329
  true1    542    324    234
  true2    612    214    274
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`80`
- 验证最优轮次（best_epoch）：`62`
- Val F1-macro（最优）：`0.4637`
- Val loss（最优时）：`0.9672`

**Test（overall）**
- Accuracy：`0.4943`
- F1-macro：`0.4420`
- Recall-macro：`0.4415`
- Recall idle/left/right：`0.6527` / `0.3764` / `0.2955`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1436    435    329
  true1    491    414    195
  true2    480    295    325
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`94`
- 验证最优轮次（best_epoch）：`76`
- Val F1-macro（最优）：`0.4162`
- Val loss（最优时）：`1.1545`

**Test（overall）**
- Accuracy：`0.4660`
- F1-macro：`0.4339`
- Recall-macro：`0.4352`
- Recall idle/left/right：`0.5585` / `0.4080` / `0.3390`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1117    429    454
  true1    375    408    217
  true2    435    226    339
```

### 共用超参
```json
{
  "data_tag": "openbmi_balanced_train_2s",
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

- 结束：`2026-08-05T20:23:44`
