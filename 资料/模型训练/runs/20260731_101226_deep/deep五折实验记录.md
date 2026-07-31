# 被试独立五折实验记录（20260731_101226 / deep）

- 开始：`2026-07-31T10:12:26`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`deep`（单脚本；无 registry）
- 结构：Deep4Net（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\deep\merged_2s\run_20260731_101226`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.2639 ± 0.2671`
- Test F1：`0.3134 ± 0.2689`
- Test Acc：`0.3976 ± 0.1188`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.7135`
- Val loss（最优时）：`0.6773`

**Test（overall）**
- Accuracy：`0.5979`
- Recall：`0.7116`
- Specificity：`0.3170`
- Precision：`0.7201`
- F1：`0.7158`
- Balanced Acc：`0.5143`
- 混淆矩阵：TP=`3592` TN=`648` FP=`1396` FN=`1456`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6086` F1=`0.7156` BalAcc=`0.5435`
- `stieger_only`：Acc=`0.5972` F1=`0.7158` BalAcc=`0.5124`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.4041`
- Val loss（最优时）：`0.7142`

**Test（overall）**
- Accuracy：`0.3963`
- Recall：`0.2390`
- Specificity：`0.7971`
- Precision：`0.7502`
- F1：`0.3626`
- Balanced Acc：`0.5181`
- 混淆矩阵：TP=`1042` TN=`1363` FP=`347` FN=`3317`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3995` F1=`0.2413` BalAcc=`0.5396`
- `stieger_only`：Acc=`0.3961` F1=`0.3696` BalAcc=`0.5155`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.0209`
- Val loss（最优时）：`0.8164`

**Test（overall）**
- Accuracy：`0.2642`
- Recall：`0.0164`
- Specificity：`0.9926`
- Precision：`0.8667`
- F1：`0.0322`
- Balanced Acc：`0.5045`
- 混淆矩阵：TP=`78` TN=`1606` FP=`12` FN=`4677`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3513` F1=`0.0664` BalAcc=`0.5172`
- `stieger_only`：Acc=`0.2586` F1=`0.0302` BalAcc=`0.5037`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.1785`
- Val loss（最优时）：`0.7459`

**Test（overall）**
- Accuracy：`0.4373`
- Recall：`0.3404`
- Specificity：`0.6547`
- Precision：`0.6888`
- F1：`0.4557`
- Balanced Acc：`0.4976`
- 混淆矩阵：TP=`1930` TN=`1653` FP=`872` FN=`3739`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3397` F1=`0.1352` BalAcc=`0.4835`
- `stieger_only`：Acc=`0.4419` F1=`0.4667` BalAcc=`0.4978`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.0022`
- Val loss（最优时）：`0.8782`

**Test（overall）**
- Accuracy：`0.2923`
- Recall：`0.0003`
- Specificity：`0.9992`
- Precision：`0.5000`
- F1：`0.0007`
- Balanced Acc：`0.4998`
- 混淆矩阵：TP=`2` TN=`2432` FP=`2` FN=`5892`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3054` F1=`0.0000` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.2920` F1=`0.0007` BalAcc=`0.4998`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4246 ± 0.0266`
- Test F1-macro：`0.4060 ± 0.0584`
- Test Acc：`0.4170 ± 0.0534`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`122`
- 验证最优轮次（best_epoch）：`104`
- Val F1-macro（最优）：`0.4234`
- Val loss（最优时）：`1.0831`

**Test（overall）**
- Accuracy：`0.3684`
- F1-macro：`0.3558`
- Recall-macro：`0.3796`
- Recall idle/left/right：`0.5352` / `0.4254` / `0.1782`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1094    704    246
  true1   1169   1066    271
  true2   1059   1030    453
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5101` F1m=`0.5093`
- `stieger_only`：Acc=`0.3601` F1m=`0.3456`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`123`
- 验证最优轮次（best_epoch）：`105`
- Val F1-macro（最优）：`0.4385`
- Val loss（最优时）：`1.0684`

**Test（overall）**
- Accuracy：`0.3740`
- F1-macro：`0.3502`
- Recall-macro：`0.4004`
- Recall idle/left/right：`0.6889` / `0.3853` / `0.1270`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1178    444     88
  true1   1218    803     63
  true2   1393    593    289
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4573` F1m=`0.4071`
- `stieger_only`：Acc=`0.3682` F1m=`0.3452`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`110`
- 验证最优轮次（best_epoch）：`92`
- Val F1-macro（最优）：`0.3788`
- Val loss（最优时）：`1.1045`

**Test（overall）**
- Accuracy：`0.5115`
- F1-macro：`0.5089`
- Recall-macro：`0.5135`
- Recall idle/left/right：`0.5290` / `0.4940` / `0.5176`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    856    360    402
  true1    646   1197    580
  true2    503    622   1207
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4795` F1m=`0.4615`
- `stieger_only`：Acc=`0.5136` F1m=`0.5110`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`149`
- 验证最优轮次（best_epoch）：`131`
- Val F1-macro（最优）：`0.4223`
- Val loss（最优时）：`1.1423`

**Test（overall）**
- Accuracy：`0.3921`
- F1-macro：`0.3871`
- Recall-macro：`0.3888`
- Recall idle/left/right：`0.2962` / `0.4105` / `0.4597`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    748    926    851
  true1    699   1176    990
  true2    660    855   1289
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4239` F1m=`0.3959`
- `stieger_only`：Acc=`0.3906` F1m=`0.3849`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`107`
- 验证最优轮次（best_epoch）：`89`
- Val F1-macro（最优）：`0.4598`
- Val loss（最优时）：`1.0457`

**Test（overall）**
- Accuracy：`0.4390`
- F1-macro：`0.4281`
- Recall-macro：`0.4307`
- Recall idle/left/right：`0.2954` / `0.5526` / `0.4441`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    719   1108    607
  true1    636   1627    681
  true2    611   1029   1310
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4551` F1m=`0.4384`
- `stieger_only`：Acc=`0.4387` F1m=`0.4264`

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

- 结束：`2026-07-31T10:50:32`
