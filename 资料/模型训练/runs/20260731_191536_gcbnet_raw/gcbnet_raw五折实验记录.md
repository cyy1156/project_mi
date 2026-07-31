# 被试独立五折实验记录（20260731_191536 / gcbnet_raw）

- 开始：`2026-07-31T19:15:36`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`gcbnet_raw`（TemporalEncoder + GCBNet）
- 输入：raw `(36056, 8, 500)` → Encoder → 节点特征 D=64（非 bandpower）
- 结构：GCBNetRaw(k=2, layers=[128], in_channels=64, relu_is=1)
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\gcbnet_raw\merged_2s\run_20260731_191536`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8300 ± 0.0125`
- Test F1：`0.8336 ± 0.0121`
- Test Acc：`0.7149 ± 0.0179`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.8450`
- Val loss（最优时）：`0.6560`

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
- Val loss（最优时）：`0.6580`

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
- Val loss（最优时）：`0.6543`

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
- Val loss（最优时）：`0.6598`

**Test（overall）**
- Accuracy：`0.6917`
- Recall：`0.9998`
- Specificity：`0.0000`
- Precision：`0.6918`
- F1：`0.8178`
- Balanced Acc：`0.4999`
- 混淆矩阵：TP=`5668` TN=`0` FP=`2525` FN=`1`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6739` F1=`0.8052` BalAcc=`0.4980`
- `stieger_only`：Acc=`0.6926` F1=`0.8184` BalAcc=`0.5000`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.8407`
- Val loss（最优时）：`0.6615`

**Test（overall）**
- Accuracy：`0.7064`
- Recall：`0.9976`
- Specificity：`0.0012`
- Precision：`0.7075`
- F1：`0.8279`
- Balanced Acc：`0.4994`
- 混淆矩阵：TP=`5880` TN=`3` FP=`2431` FN=`14`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6467` F1=`0.7807` BalAcc=`0.4820`
- `stieger_only`：Acc=`0.7076` F1=`0.8288` BalAcc=`0.4997`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4241 ± 0.0183`
- Test F1-macro：`0.3962 ± 0.0442`
- Test Acc：`0.4091 ± 0.0431`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`92`
- 验证最优轮次（best_epoch）：`74`
- Val F1-macro（最优）：`0.4230`
- Val loss（最优时）：`1.1012`

**Test（overall）**
- Accuracy：`0.3852`
- F1-macro：`0.3825`
- Recall-macro：`0.3823`
- Recall idle/left/right：`0.3410` / `0.3871` / `0.4190`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    697    757    590
  true1    675    970    861
  true2    643    834   1065
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4924` F1m=`0.4886`
- `stieger_only`：Acc=`0.3789` F1m=`0.3761`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`100`
- 验证最优轮次（best_epoch）：`82`
- Val F1-macro（最优）：`0.4396`
- Val loss（最优时）：`1.0649`

**Test（overall）**
- Accuracy：`0.3725`
- F1-macro：`0.3605`
- Recall-macro：`0.3953`
- Recall idle/left/right：`0.6889` / `0.2490` / `0.2479`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1178    265    267
  true1   1333    519    232
  true2   1308    403    564
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5477` F1m=`0.5486`
- `stieger_only`：Acc=`0.3603` F1m=`0.3462`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`71`
- 验证最优轮次（best_epoch）：`53`
- Val F1-macro（最优）：`0.3908`
- Val loss（最优时）：`1.1073`

**Test（overall）**
- Accuracy：`0.4905`
- F1-macro：`0.4830`
- Recall-macro：`0.4804`
- Recall idle/left/right：`0.3863` / `0.4494` / `0.6055`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    625    475    518
  true1    289   1089   1045
  true2    250    670   1412
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4872` F1m=`0.4690`
- `stieger_only`：Acc=`0.4907` F1m=`0.4839`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`48`
- 验证最优轮次（best_epoch）：`30`
- Val F1-macro（最优）：`0.4253`
- Val loss（最优时）：`1.0842`

**Test（overall）**
- Accuracy：`0.3824`
- F1-macro：`0.3731`
- Recall-macro：`0.3796`
- Recall idle/left/right：`0.2848` / `0.3173` / `0.5367`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    719    763   1043
  true1    637    909   1319
  true2    599    700   1505
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3832` F1m=`0.3830`
- `stieger_only`：Acc=`0.3823` F1m=`0.3725`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`73`
- 验证最优轮次（best_epoch）：`55`
- Val F1-macro（最优）：`0.4420`
- Val loss（最优时）：`1.0587`

**Test（overall）**
- Accuracy：`0.4151`
- F1-macro：`0.3818`
- Recall-macro：`0.3996`
- Recall idle/left/right：`0.1487` / `0.4891` / `0.5610`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    362   1059   1013
  true1    320   1440   1184
  true2    205   1090   1655
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3952` F1m=`0.3222`
- `stieger_only`：Acc=`0.4155` F1m=`0.3775`

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

- 结束：`2026-07-31T19:57:55`
