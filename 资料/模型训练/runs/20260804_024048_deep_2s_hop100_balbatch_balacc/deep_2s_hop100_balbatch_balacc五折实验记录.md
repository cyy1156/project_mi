# 被试独立五折实验记录（20260804_024048 / deep_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T02:40:48`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`deep`（原结构）
- 结构：Deep4Net-compat（pool=1/1，对齐1s自动缩核；非满血stride3）
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\deep_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_024048`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.6194 ± 0.0514`
- Test BalAcc：`0.5702 ± 0.0263`
- Test Spec：`0.5389 ± 0.1550`
- Test Rec：`0.6016 ± 0.1255`
- Test F1：`0.6562 ± 0.0671`
- Test Acc：`0.5809 ± 0.0414`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`62`
- 验证最优轮次（best_epoch）：`44`
- Val 选模分数（Balanced Acc）：`0.7102`
- Val F1（最优 checkpoint 时，附报）：`0.8279`
- Val loss（最优时）：`0.5985`

**Test（overall）**
- Accuracy：`0.6419`
- Recall：`0.7731`
- Specificity：`0.3625`
- Precision：`0.7209`
- F1：`0.7461`
- Balanced Acc：`0.5678`
- 混淆矩阵：TP=`4335` TN=`954` FP=`1678` FN=`1272`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`99`
- 验证最优轮次（best_epoch）：`81`
- Val 选模分数（Balanced Acc）：`0.6199`
- Val F1（最优 checkpoint 时，附报）：`0.5650`
- Val loss（最优时）：`1.4757`

**Test（overall）**
- Accuracy：`0.5394`
- Recall：`0.4479`
- Specificity：`0.7372`
- Precision：`0.7864`
- F1：`0.5707`
- Balanced Acc：`0.5925`
- 混淆矩阵：TP=`2530` TN=`1927` FP=`687` FN=`3119`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`103`
- 验证最优轮次（best_epoch）：`85`
- Val 选模分数（Balanced Acc）：`0.6130`
- Val F1（最优 checkpoint 时，附报）：`0.7405`
- Val loss（最优时）：`1.0002`

**Test（overall）**
- Accuracy：`0.6008`
- Recall：`0.7186`
- Specificity：`0.3512`
- Precision：`0.7012`
- F1：`0.7098`
- Balanced Acc：`0.5349`
- 混淆矩阵：TP=`3954` TN=`912` FP=`1685` FN=`1548`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5510`
- Val F1（最优 checkpoint 时，附报）：`0.5931`
- Val loss（最优时）：`0.8790`

**Test（overall）**
- Accuracy：`0.5300`
- Recall：`0.4963`
- Specificity：`0.6033`
- Precision：`0.7312`
- F1：`0.5913`
- Balanced Acc：`0.5498`
- 混淆矩阵：TP=`2595` TN=`1451` FP=`954` FN=`2634`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`58`
- 验证最优轮次（best_epoch）：`40`
- Val 选模分数（Balanced Acc）：`0.6030`
- Val F1（最优 checkpoint 时，附报）：`0.7240`
- Val loss（最优时）：`0.8788`

**Test（overall）**
- Accuracy：`0.5926`
- Recall：`0.5722`
- Specificity：`0.6401`
- Precision：`0.7880`
- F1：`0.6630`
- Balanced Acc：`0.6062`
- 混淆矩阵：TP=`1394` TN=`667` FP=`375` FN=`1042`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4547 ± 0.0406`
- Val F1-macro：`0.4398 ± 0.0364`
- Test BalAcc：`0.4228 ± 0.0229`
- Test F1-macro：`0.4160 ± 0.0253`
- Test Acc：`0.4219 ± 0.0231`
- Test Precision-macro：`0.4321 ± 0.0246`
- Test Recall-macro：`0.4228 ± 0.0229`
- Test Recall idle/left/right：`0.4700±0.1277` / `0.4125±0.1119` / `0.3858±0.0737`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`61`
- 验证最优轮次（best_epoch）：`43`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5316`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5116`
- Val loss（最优时）：`1.2533`

**Test（overall）**
- Accuracy：`0.4393`
- Balanced Acc：`0.4401`
- F1-macro：`0.4393`
- Precision-macro：`0.4453`
- Recall-macro：`0.4401`
- Recall idle/left/right：`0.4905` / `0.4009` / `0.4289`
- Precision idle/left/right：`0.3884` / `0.4765` / `0.4712`
- F1 idle/left/right：`0.4335` / `0.4355` / `0.4490`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1291    543    798
  true1   1071   1103    577
  true2    962    669   1225
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`61`
- 验证最优轮次（best_epoch）：`43`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4567`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4249`
- Val loss（最优时）：`1.3329`

**Test（overall）**
- Accuracy：`0.4284`
- Balanced Acc：`0.4350`
- F1-macro：`0.4154`
- Precision-macro：`0.4607`
- Recall-macro：`0.4350`
- Recall idle/left/right：`0.6890` / `0.2515` / `0.3646`
- Precision idle/left/right：`0.3713` / `0.4648` / `0.5460`
- F1 idle/left/right：`0.4826` / `0.3264` / `0.4372`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1801    378    435
  true1   1704    713    418
  true2   1345    443   1026
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`78`
- 验证最优轮次（best_epoch）：`60`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4395`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4311`
- Val loss（最优时）：`1.4794`

**Test（overall）**
- Accuracy：`0.3941`
- Balanced Acc：`0.3931`
- F1-macro：`0.3812`
- Precision-macro：`0.4039`
- Recall-macro：`0.3931`
- Recall idle/left/right：`0.2938` / `0.6026` / `0.2828`
- Precision idle/left/right：`0.3869` / `0.3753` / `0.4495`
- F1 idle/left/right：`0.3340` / `0.4625` / `0.3472`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    763   1387    447
  true1    572   1645    513
  true2    637   1351    784
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4182`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4177`
- Val loss（最优时）：`1.3519`

**Test（overall）**
- Accuracy：`0.3960`
- Balanced Acc：`0.3975`
- F1-macro：`0.3959`
- Precision-macro：`0.4013`
- Recall-macro：`0.3975`
- Recall idle/left/right：`0.4478` / `0.3926` / `0.3522`
- Precision idle/left/right：`0.3647` / `0.3873` / `0.4518`
- F1 idle/left/right：`0.4020` / `0.3899` / `0.3958`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1077    854    474
  true1    912   1014    657
  true2    964    750    932
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`106`
- 验证最优轮次（best_epoch）：`88`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4273`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4138`
- Val loss（最优时）：`1.5500`

**Test（overall）**
- Accuracy：`0.4517`
- Balanced Acc：`0.4482`
- F1-macro：`0.4481`
- Precision-macro：`0.4492`
- Recall-macro：`0.4482`
- Recall idle/left/right：`0.4290` / `0.4151` / `0.5004`
- Precision idle/left/right：`0.3942` / `0.4129` / `0.5404`
- F1 idle/left/right：`0.4108` / `0.4140` / `0.5196`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    447    347    248
  true1    336    462    315
  true2    351    310    662
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

- 结束：`2026-08-04T06:19:05`
