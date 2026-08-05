# 被试独立五折实验记录（20260804_070726 / conformer_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T07:07:26`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`conformer`（原结构）
- 结构：EEGConformer num_layers=2, num_heads=10, att_drop=0.5
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\conformer_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_070726`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.6243 ± 0.0607`
- Test BalAcc：`0.5774 ± 0.0132`
- Test Spec：`0.6122 ± 0.1428`
- Test Rec：`0.5427 ± 0.1315`
- Test F1：`0.6210 ± 0.0816`
- Test Acc：`0.5632 ± 0.0481`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.7421`
- Val F1（最优 checkpoint 时，附报）：`0.8290`
- Val loss（最优时）：`0.5045`

**Test（overall）**
- Accuracy：`0.5560`
- Recall：`0.5479`
- Specificity：`0.5733`
- Precision：`0.7323`
- F1：`0.6268`
- Balanced Acc：`0.5606`
- 混淆矩阵：TP=`3072` TN=`1509` FP=`1123` FN=`2535`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.6225`
- Val F1（最优 checkpoint 时，附报）：`0.6781`
- Val loss（最优时）：`0.7652`

**Test（overall）**
- Accuracy：`0.5285`
- Recall：`0.4433`
- Specificity：`0.7127`
- Precision：`0.7693`
- F1：`0.5624`
- Balanced Acc：`0.5780`
- 混淆矩阵：TP=`2504` TN=`1863` FP=`751` FN=`3145`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5851`
- Val F1（最优 checkpoint 时，附报）：`0.7819`
- Val loss（最优时）：`0.6719`

**Test（overall）**
- Accuracy：`0.6054`
- Recall：`0.6234`
- Specificity：`0.5672`
- Precision：`0.7532`
- F1：`0.6822`
- Balanced Acc：`0.5953`
- 混淆矩阵：TP=`3430` TN=`1473` FP=`1124` FN=`2072`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc）：`0.5919`
- Val F1（最优 checkpoint 时，附报）：`0.6860`
- Val loss（最优时）：`0.8025`

**Test（overall）**
- Accuracy：`0.6285`
- Recall：`0.7365`
- Specificity：`0.3938`
- Precision：`0.7254`
- F1：`0.7309`
- Balanced Acc：`0.5651`
- 混淆矩阵：TP=`3851` TN=`947` FP=`1458` FN=`1378`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.5802`
- Val F1（最优 checkpoint 时，附报）：`0.6549`
- Val loss（最优时）：`0.7623`

**Test（overall）**
- Accuracy：`0.4977`
- Recall：`0.3625`
- Specificity：`0.8138`
- Precision：`0.8199`
- F1：`0.5027`
- Balanced Acc：`0.5881`
- 混淆矩阵：TP=`883` TN=`848` FP=`194` FN=`1553`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4835 ± 0.0453`
- Val F1-macro：`0.4725 ± 0.0436`
- Test BalAcc：`0.4567 ± 0.0254`
- Test F1-macro：`0.4433 ± 0.0229`
- Test Acc：`0.4544 ± 0.0226`
- Test Precision-macro：`0.4749 ± 0.0327`
- Test Recall-macro：`0.4567 ± 0.0254`
- Test Recall idle/left/right：`0.5624±0.1324` / `0.3688±0.1295` / `0.4389±0.1284`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5689`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5528`
- Val loss（最优时）：`0.9797`

**Test（overall）**
- Accuracy：`0.4513`
- Balanced Acc：`0.4502`
- F1-macro：`0.4364`
- Precision-macro：`0.4667`
- Recall-macro：`0.4502`
- Recall idle/left/right：`0.5068` / `0.2457` / `0.5980`
- Precision idle/left/right：`0.4124` / `0.5294` / `0.4583`
- F1 idle/left/right：`0.4547` / `0.3357` / `0.5189`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1334    306    992
  true1   1048    676   1027
  true2    853    295   1708
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4751`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4538`
- Val loss（最优时）：`1.1715`

**Test（overall）**
- Accuracy：`0.4763`
- Balanced Acc：`0.4816`
- F1-macro：`0.4710`
- Precision-macro：`0.5072`
- Recall-macro：`0.4816`
- Recall idle/left/right：`0.6859` / `0.3626` / `0.3962`
- Precision idle/left/right：`0.4053` / `0.5393` / `0.5768`
- F1 idle/left/right：`0.5095` / `0.4337` / `0.4698`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1793    393    428
  true1   1417   1028    390
  true2   1214    485   1115
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4735`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4714`
- Val loss（最优时）：`1.0966`

**Test（overall）**
- Accuracy：`0.4260`
- Balanced Acc：`0.4276`
- F1-macro：`0.4099`
- Precision-macro：`0.4509`
- Recall-macro：`0.4276`
- Recall idle/left/right：`0.4582` / `0.6059` / `0.2186`
- Precision idle/left/right：`0.3991` / `0.4141` / `0.5396`
- F1 idle/left/right：`0.4266` / `0.4920` / `0.3112`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1190   1204    203
  true1    762   1654    314
  true2   1030   1136    606
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4334`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4212`
- Val loss（最优时）：`1.1490`

**Test（overall）**
- Accuracy：`0.4345`
- Balanced Acc：`0.4333`
- F1-macro：`0.4322`
- Precision-macro：`0.4322`
- Recall-macro：`0.4333`
- Recall idle/left/right：`0.4104` / `0.3728` / `0.5166`
- Precision idle/left/right：`0.4314` / `0.3940` / `0.4711`
- F1 idle/left/right：`0.4206` / `0.3831` / `0.4928`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    987    804    614
  true1    699    963    921
  true2    602    677   1367
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`49`
- 验证最优轮次（best_epoch）：`31`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4664`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4633`
- Val loss（最优时）：`1.1811`

**Test（overall）**
- Accuracy：`0.4839`
- Balanced Acc：`0.4908`
- F1-macro：`0.4670`
- Precision-macro：`0.5177`
- Recall-macro：`0.4908`
- Recall idle/left/right：`0.7505` / `0.2570` / `0.4649`
- Precision idle/left/right：`0.4008` / `0.5287` / `0.6237`
- F1 idle/left/right：`0.5226` / `0.3458` / `0.5327`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    782    142    118
  true1    574    286    253
  true2    595    113    615
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

- 结束：`2026-08-04T07:41:12`
