# 被试独立五折实验记录（20260804_061913 / eegtcnet_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T06:19:13`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`eegtcnet`（原结构）
- 结构：EEGTCNet（braindecode 默认）
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\eegtcnet_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_061913`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5901 ± 0.0691`
- Test BalAcc：`0.5856 ± 0.0284`
- Test Spec：`0.6770 ± 0.1685`
- Test Rec：`0.4943 ± 0.1700`
- Test F1：`0.5807 ± 0.1385`
- Test Acc：`0.5494 ± 0.0724`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`110`
- 验证最优轮次（best_epoch）：`92`
- Val 选模分数（Balanced Acc）：`0.7167`
- Val F1（最优 checkpoint 时，附报）：`0.8098`
- Val loss（最优时）：`0.5372`

**Test（overall）**
- Accuracy：`0.5553`
- Recall：`0.5104`
- Specificity：`0.6508`
- Precision：`0.7569`
- F1：`0.6097`
- Balanced Acc：`0.5806`
- 混淆矩阵：TP=`2862` TN=`1713` FP=`919` FN=`2745`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.5411`
- Val F1（最优 checkpoint 时，附报）：`0.5505`
- Val loss（最优时）：`0.6943`

**Test（overall）**
- Accuracy：`0.5601`
- Recall：`0.4826`
- Specificity：`0.7276`
- Precision：`0.7929`
- F1：`0.6000`
- Balanced Acc：`0.6051`
- 混淆矩阵：TP=`2726` TN=`1902` FP=`712` FN=`2923`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc）：`0.5208`
- Val F1（最优 checkpoint 时，附报）：`0.6782`
- Val loss（最优时）：`0.6950`

**Test（overall）**
- Accuracy：`0.6545`
- Recall：`0.7130`
- Specificity：`0.5306`
- Precision：`0.7629`
- F1：`0.7371`
- Balanced Acc：`0.6218`
- 混淆矩阵：TP=`3923` TN=`1378` FP=`1219` FN=`1579`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val 选模分数（Balanced Acc）：`0.5681`
- Val F1（最优 checkpoint 时，附报）：`0.6621`
- Val loss（最优时）：`0.7261`

**Test（overall）**
- Accuracy：`0.5503`
- Recall：`0.5718`
- Specificity：`0.5035`
- Precision：`0.7146`
- F1：`0.6353`
- Balanced Acc：`0.5377`
- 混淆矩阵：TP=`2990` TN=`1211` FP=`1194` FN=`2239`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`65`
- 验证最优轮次（best_epoch）：`47`
- Val 选模分数（Balanced Acc）：`0.6038`
- Val F1（最优 checkpoint 时，附报）：`0.7245`
- Val loss（最优时）：`0.6615`

**Test（overall）**
- Accuracy：`0.4270`
- Recall：`0.1938`
- Specificity：`0.9722`
- Precision：`0.9421`
- F1：`0.3214`
- Balanced Acc：`0.5830`
- 混淆矩阵：TP=`472` TN=`1013` FP=`29` FN=`1964`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4442 ± 0.0659`
- Val F1-macro：`0.4102 ± 0.0882`
- Test BalAcc：`0.4135 ± 0.0171`
- Test F1-macro：`0.3667 ± 0.0545`
- Test Acc：`0.4086 ± 0.0231`
- Test Precision-macro：`0.4435 ± 0.0156`
- Test Recall-macro：`0.4135 ± 0.0171`
- Test Recall idle/left/right：`0.5216±0.3070` / `0.4334±0.2493` / `0.2856±0.1744`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`70`
- 验证最优轮次（best_epoch）：`52`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5630`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5556`
- Val loss（最优时）：`0.9722`

**Test（overall）**
- Accuracy：`0.4325`
- Balanced Acc：`0.4304`
- F1-macro：`0.4292`
- Precision-macro：`0.4331`
- Recall-macro：`0.4304`
- Recall idle/left/right：`0.3514` / `0.4373` / `0.5025`
- Precision idle/left/right：`0.4349` / `0.4386` / `0.4259`
- F1 idle/left/right：`0.3887` / `0.4379` / `0.4610`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    925    686   1021
  true1    635   1203    913
  true2    567    854   1435
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3749`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3075`
- Val loss（最优时）：`1.0953`

**Test（overall）**
- Accuracy：`0.4052`
- Balanced Acc：`0.4156`
- F1-macro：`0.3374`
- Precision-macro：`0.4684`
- Recall-macro：`0.4156`
- Recall idle/left/right：`0.8393` / `0.3626` / `0.0448`
- Precision idle/left/right：`0.3713` / `0.4838` / `0.5502`
- F1 idle/left/right：`0.5148` / `0.4145` / `0.0828`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2194    368     52
  true1   1756   1028     51
  true2   1959    729    126
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4147`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3671`
- Val loss（最优时）：`1.0919`

**Test（overall）**
- Accuracy：`0.3939`
- Balanced Acc：`0.3910`
- F1-macro：`0.3395`
- Precision-macro：`0.4543`
- Recall-macro：`0.3910`
- Recall idle/left/right：`0.1529` / `0.8223` / `0.1977`
- Precision idle/left/right：`0.5796` / `0.3685` / `0.4148`
- F1 idle/left/right：`0.2419` / `0.5089` / `0.2678`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    397   1795    405
  true1    117   2245    368
  true2    171   2053    548
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`105`
- 验证最优轮次（best_epoch）：`87`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4637`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4621`
- Val loss（最优时）：`1.0608`

**Test（overall）**
- Accuracy：`0.4362`
- Balanced Acc：`0.4335`
- F1-macro：`0.4319`
- Precision-macro：`0.4354`
- Recall-macro：`0.4335`
- Recall idle/left/right：`0.3314` / `0.4990` / `0.4701`
- Precision idle/left/right：`0.4060` / `0.4142` / `0.4861`
- F1 idle/left/right：`0.3649` / `0.4527` / `0.4780`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    797   1034    574
  true1    553   1289    741
  true2    613    789   1244
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`77`
- 验证最优轮次（best_epoch）：`59`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4047`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3586`
- Val loss（最优时）：`1.0672`

**Test（overall）**
- Accuracy：`0.3752`
- Balanced Acc：`0.3973`
- F1-macro：`0.2954`
- Precision-macro：`0.4261`
- Recall-macro：`0.3973`
- Recall idle/left/right：`0.9328` / `0.0458` / `0.2132`
- Precision idle/left/right：`0.3547` / `0.4766` / `0.4469`
- F1 idle/left/right：`0.5140` / `0.0836` / `0.2886`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    972     11     59
  true1    772     51    290
  true2    996     45    282
```

### 共用超参
```json
{
  "data_tag": "bci2a_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 18,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-offline-native",
  "early_stop": "balanced_accuracy",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true
}
```

- 结束：`2026-08-04T07:07:18`
