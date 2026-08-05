# 被试独立五折实验记录（20260804_022251 / shallow_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T02:22:51`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`shallow`（原结构）
- 结构：ShallowFBCSPNet（braindecode 默认）
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\shallow_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_022251`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.6405 ± 0.0445`
- Test BalAcc：`0.6027 ± 0.0347`
- Test Spec：`0.5775 ± 0.1171`
- Test Rec：`0.6279 ± 0.0684`
- Test F1：`0.6873 ± 0.0323`
- Test Acc：`0.6113 ± 0.0247`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.7255`
- Val F1（最优 checkpoint 时，附报）：`0.8067`
- Val loss（最优时）：`0.5384`

**Test（overall）**
- Accuracy：`0.5673`
- Recall：`0.5762`
- Specificity：`0.5483`
- Precision：`0.7310`
- F1：`0.6445`
- Balanced Acc：`0.5622`
- 混淆矩阵：TP=`3231` TN=`1443` FP=`1189` FN=`2376`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`58`
- 验证最优轮次（best_epoch）：`40`
- Val 选模分数（Balanced Acc）：`0.6351`
- Val F1（最优 checkpoint 时，附报）：`0.7691`
- Val loss（最优时）：`0.6985`

**Test（overall）**
- Accuracy：`0.6028`
- Recall：`0.5920`
- Specificity：`0.6262`
- Precision：`0.7739`
- F1：`0.6708`
- Balanced Acc：`0.6091`
- 混淆矩阵：TP=`3344` TN=`1637` FP=`977` FN=`2305`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5959`
- Val F1（最优 checkpoint 时，附报）：`0.7358`
- Val loss（最优时）：`0.6542`

**Test（overall）**
- Accuracy：`0.6302`
- Recall：`0.6627`
- Specificity：`0.5614`
- Precision：`0.7620`
- F1：`0.7089`
- Balanced Acc：`0.6120`
- 混淆矩阵：TP=`3646` TN=`1458` FP=`1139` FN=`1856`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.6202`
- Val F1（最优 checkpoint 时，附报）：`0.7296`
- Val loss（最优时）：`0.7123`

**Test（overall）**
- Accuracy：`0.6356`
- Recall：`0.7460`
- Specificity：`0.3954`
- Precision：`0.7285`
- F1：`0.7372`
- Balanced Acc：`0.5707`
- 混淆矩阵：TP=`3901` TN=`951` FP=`1454` FN=`1328`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.6257`
- Val F1（最优 checkpoint 时，附报）：`0.7699`
- Val loss（最优时）：`0.6488`

**Test（overall）**
- Accuracy：`0.6208`
- Recall：`0.5628`
- Specificity：`0.7562`
- Precision：`0.8437`
- F1：`0.6752`
- Balanced Acc：`0.6595`
- 混淆矩阵：TP=`1371` TN=`788` FP=`254` FN=`1065`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4819 ± 0.0343`
- Val F1-macro：`0.4754 ± 0.0328`
- Test BalAcc：`0.4614 ± 0.0362`
- Test F1-macro：`0.4547 ± 0.0378`
- Test Acc：`0.4604 ± 0.0339`
- Test Precision-macro：`0.4718 ± 0.0443`
- Test Recall-macro：`0.4614 ± 0.0362`
- Test Recall idle/left/right：`0.4980±0.1081` / `0.4339±0.1213` / `0.4524±0.1168`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5490`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5403`
- Val loss（最优时）：`0.9933`

**Test（overall）**
- Accuracy：`0.4310`
- Balanced Acc：`0.4308`
- F1-macro：`0.4298`
- Precision-macro：`0.4336`
- Recall-macro：`0.4308`
- Recall idle/left/right：`0.4529` / `0.3660` / `0.4734`
- Precision idle/left/right：`0.3836` / `0.4508` / `0.4665`
- F1 idle/left/right：`0.4154` / `0.4040` / `0.4699`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1192    557    883
  true1   1081   1007    663
  true2    834    670   1352
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4695`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4676`
- Val loss（最优时）：`1.0602`

**Test（overall）**
- Accuracy：`0.4825`
- Balanced Acc：`0.4850`
- F1-macro：`0.4820`
- Precision-macro：`0.4960`
- Recall-macro：`0.4850`
- Recall idle/left/right：`0.5842` / `0.4554` / `0.4154`
- Precision idle/left/right：`0.4200` / `0.5093` / `0.5588`
- F1 idle/left/right：`0.4886` / `0.4808` / `0.4766`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1527    546    541
  true1   1162   1291    382
  true2    947    698   1169
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4604`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4566`
- Val loss（最优时）：`1.1140`

**Test（overall）**
- Accuracy：`0.4304`
- Balanced Acc：`0.4305`
- F1-macro：`0.4170`
- Precision-macro：`0.4449`
- Recall-macro：`0.4305`
- Recall idle/left/right：`0.3801` / `0.6469` / `0.2644`
- Precision idle/left/right：`0.4676` / `0.3995` / `0.4675`
- F1 idle/left/right：`0.4193` / `0.4940` / `0.3378`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    987   1256    354
  true1    483   1766    481
  true2    641   1398    733
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`36`
- 验证最优轮次（best_epoch）：`18`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4756`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4602`
- Val loss（最优时）：`1.2196`

**Test（overall）**
- Accuracy：`0.4416`
- Balanced Acc：`0.4393`
- F1-macro：`0.4292`
- Precision-macro：`0.4366`
- Recall-macro：`0.4393`
- Recall idle/left/right：`0.4096` / `0.2826` / `0.6259`
- Precision idle/left/right：`0.4405` / `0.4136` / `0.4558`
- F1 idle/left/right：`0.4245` / `0.3358` / `0.5275`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    985    585    835
  true1    711    730   1142
  true2    540    450   1656
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4547`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4522`
- Val loss（最优时）：`1.2020`

**Test（overall）**
- Accuracy：`0.5164`
- Balanced Acc：`0.5216`
- F1-macro：`0.5155`
- Precision-macro：`0.5479`
- Recall-macro：`0.5216`
- Recall idle/left/right：`0.6631` / `0.4187` / `0.4830`
- Precision idle/left/right：`0.4158` / `0.4910` / `0.7370`
- F1 idle/left/right：`0.5111` / `0.4520` / `0.5836`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    691    271     80
  true1    499    466    148
  true2    472    212    639
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

- 结束：`2026-08-04T02:40:41`
