# 被试独立五折实验记录（20260804_074123 / dbn_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T07:41:23`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`dbn`（原结构）
- 结构：DBN + 2s μ/β log bandpower (N,8,2)
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\dbn_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_074123`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5809 ± 0.0843`
- Test BalAcc：`0.5092 ± 0.0071`
- Test Spec：`0.6299 ± 0.4020`
- Test Rec：`0.3886 ± 0.4104`
- Test F1：`0.3667 ± 0.3442`
- Test Acc：`0.4603 ± 0.1556`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.7486`
- Val F1（最优 checkpoint 时，附报）：`0.8253`
- Val loss（最优时）：`0.6813`

**Test（overall）**
- Accuracy：`0.6637`
- Recall：`0.9351`
- Specificity：`0.0855`
- Precision：`0.6854`
- F1：`0.7910`
- Balanced Acc：`0.5103`
- 混淆矩阵：TP=`5243` TN=`225` FP=`2407` FN=`364`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.5373`
- Val F1（最优 checkpoint 时，附报）：`0.7718`
- Val loss（最优时）：`0.6820`

**Test（overall）**
- Accuracy：`0.3213`
- Recall：`0.0112`
- Specificity：`0.9916`
- Precision：`0.7412`
- F1：`0.0220`
- Balanced Acc：`0.5014`
- 混淆矩阵：TP=`63` TN=`2592` FP=`22` FN=`5586`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.5492`
- Val F1（最优 checkpoint 时，附报）：`0.7672`
- Val loss（最优时）：`0.6690`

**Test（overall）**
- Accuracy：`0.6311`
- Recall：`0.8346`
- Specificity：`0.1998`
- Precision：`0.6885`
- F1：`0.7545`
- Balanced Acc：`0.5172`
- 混淆矩阵：TP=`4592` TN=`519` FP=`2078` FN=`910`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.5468`
- Val F1（最优 checkpoint 时，附报）：`0.6073`
- Val loss（最优时）：`0.6920`

**Test（overall）**
- Accuracy：`0.3850`
- Recall：`0.1608`
- Specificity：`0.8723`
- Precision：`0.7326`
- F1：`0.2638`
- Balanced Acc：`0.5166`
- 混淆矩阵：TP=`841` TN=`2098` FP=`307` FN=`4388`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5230`
- Val F1（最优 checkpoint 时，附报）：`0.8005`
- Val loss（最优时）：`0.6806`

**Test（overall）**
- Accuracy：`0.3005`
- Recall：`0.0012`
- Specificity：`1.0000`
- Precision：`1.0000`
- F1：`0.0025`
- Balanced Acc：`0.5006`
- 混淆矩阵：TP=`3` TN=`1042` FP=`0` FN=`2433`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4076 ± 0.0476`
- Val F1-macro：`0.3427 ± 0.0636`
- Test BalAcc：`0.3446 ± 0.0139`
- Test F1-macro：`0.2506 ± 0.0652`
- Test Acc：`0.3348 ± 0.0242`
- Test Precision-macro：`0.3026 ± 0.1329`
- Test Recall-macro：`0.3446 ± 0.0139`
- Test Recall idle/left/right：`0.6184±0.3167` / `0.0742±0.0855` / `0.3412±0.3116`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4955`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4081`
- Val loss（最优时）：`1.0808`

**Test（overall）**
- Accuracy：`0.3579`
- Balanced Acc：`0.3496`
- F1-macro：`0.2700`
- Precision-macro：`0.4419`
- Recall-macro：`0.3496`
- Recall idle/left/right：`0.1873` / `0.0389` / `0.8225`
- Precision idle/left/right：`0.3220` / `0.6446` / `0.3591`
- F1 idle/left/right：`0.2368` / `0.0734` / `0.4999`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    493     18   2121
  true1    572    107   2072
  true2    466     41   2349
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`55`
- 验证最优轮次（best_epoch）：`37`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3765`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3508`
- Val loss（最优时）：`1.0968`

**Test（overall）**
- Accuracy：`0.3606`
- Balanced Acc：`0.3690`
- F1-macro：`0.2936`
- Precision-macro：`0.4288`
- Recall-macro：`0.3690`
- Recall idle/left/right：`0.6752` / `0.0113` / `0.4204`
- Precision idle/left/right：`0.3379` / `0.5517` / `0.3968`
- F1 idle/left/right：`0.4504` / `0.0221` / `0.4083`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1765     12    837
  true1   1842     32    961
  true2   1617     14   1183
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4174`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4123`
- Val loss（最优时）：`1.0853`

**Test（overall）**
- Accuracy：`0.3420`
- Balanced Acc：`0.3411`
- F1-macro：`0.3351`
- Precision-macro：`0.3416`
- Recall-macro：`0.3411`
- Recall idle/left/right：`0.3258` / `0.2344` / `0.4632`
- Precision idle/left/right：`0.3369` / `0.3430` / `0.3450`
- F1 idle/left/right：`0.3312` / `0.2785` / `0.3954`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    846    642   1109
  true1    761    640   1329
  true2    904    584   1284
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3875`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2915`
- Val loss（最优时）：`1.0955`

**Test（overall）**
- Accuracy：`0.3139`
- Balanced Acc：`0.3300`
- F1-macro：`0.2007`
- Precision-macro：`0.2009`
- Recall-macro：`0.3300`
- Recall idle/left/right：`0.9035` / `0.0863` / `0.0000`
- Precision idle/left/right：`0.3171` / `0.2855` / `0.0000`
- F1 idle/left/right：`0.4694` / `0.1326` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2173    232      0
  true1   2360    223      0
  true2   2320    326      0
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3612`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2510`
- Val loss（最优时）：`1.0978`

**Test（overall）**
- Accuracy：`0.2996`
- Balanced Acc：`0.3333`
- F1-macro：`0.1537`
- Precision-macro：`0.0999`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- Precision idle/left/right：`0.2996` / `0.0000` / `0.0000`
- F1 idle/left/right：`0.4611` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1042      0      0
  true1   1113      0      0
  true2   1323      0      0
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

- 结束：`2026-08-04T07:47:42`
