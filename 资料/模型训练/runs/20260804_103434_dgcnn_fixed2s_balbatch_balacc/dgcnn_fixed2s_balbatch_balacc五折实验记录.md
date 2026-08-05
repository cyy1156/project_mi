# 被试独立五折实验记录（20260804_103434 / dgcnn_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:34:34`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`dgcnn`（原结构）
- 结构：DGCNN(k=2, layers=[128]) + 2s μ/β log bandpower (N,8,2)
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dgcnn_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103434`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5968 ± 0.0895`
- Test BalAcc：`0.5553 ± 0.0122`
- Test Spec：`0.5573 ± 0.3193`
- Test Rec：`0.5533 ± 0.3143`
- Test F1：`0.5752 ± 0.1942`
- Test Acc：`0.5508 ± 0.1118`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.7748`
- Val F1（最优 checkpoint 时，附报）：`0.8190`
- Val loss（最优时）：`0.6122`

**Test（overall）**
- Accuracy：`0.6717`
- Recall：`0.8577`
- Specificity：`0.2868`
- Precision：`0.7134`
- F1：`0.7789`
- Balanced Acc：`0.5722`
- 混淆矩阵：TP=`229` TN=`37` FP=`92` FN=`38`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`69`
- 验证最优轮次（best_epoch）：`51`
- Val 选模分数（Balanced Acc）：`0.5670`
- Val F1（最优 checkpoint 时，附报）：`0.7591`
- Val loss（最优时）：`0.6609`

**Test（overall）**
- Accuracy：`0.4849`
- Recall：`0.3457`
- Specificity：`0.7752`
- Precision：`0.7623`
- F1：`0.4757`
- Balanced Acc：`0.5605`
- 混淆矩阵：TP=`93` TN=`100` FP=`29` FN=`176`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val 选模分数（Balanced Acc）：`0.5539`
- Val F1（最优 checkpoint 时，附报）：`0.8182`
- Val loss（最优时）：`0.6282`

**Test（overall）**
- Accuracy：`0.6949`
- Recall：`0.9924`
- Specificity：`0.0859`
- Precision：`0.6897`
- F1：`0.8138`
- Balanced Acc：`0.5392`
- 混淆矩阵：TP=`260` TN=`11` FP=`117` FN=`2`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5369`
- Val F1（最优 checkpoint 时，附报）：`0.7077`
- Val loss（最优时）：`0.6764`

**Test（overall）**
- Accuracy：`0.4891`
- Recall：`0.3896`
- Specificity：`0.6975`
- Precision：`0.7293`
- F1：`0.5079`
- Balanced Acc：`0.5435`
- 混淆矩阵：TP=`97` TN=`83` FP=`36` FN=`152`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`63`
- 验证最优轮次（best_epoch）：`45`
- Val 选模分数（Balanced Acc）：`0.5516`
- Val F1（最优 checkpoint 时，附报）：`0.7336`
- Val loss（最优时）：`0.6669`

**Test（overall）**
- Accuracy：`0.4132`
- Recall：`0.1810`
- Specificity：`0.9412`
- Precision：`0.8750`
- F1：`0.3000`
- Balanced Acc：`0.5611`
- 混淆矩阵：TP=`21` TN=`48` FP=`3` FN=`95`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4384 ± 0.0501`
- Val F1-macro：`0.4077 ± 0.0663`
- Test BalAcc：`0.3832 ± 0.0383`
- Test F1-macro：`0.3137 ± 0.0949`
- Test Acc：`0.3773 ± 0.0469`
- Test Precision-macro：`0.4023 ± 0.1533`
- Test Recall-macro：`0.3832 ± 0.0383`
- Test Recall idle/left/right：`0.5416±0.3741` / `0.2418±0.1724` / `0.3663±0.2741`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5177`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5150`
- Val loss（最优时）：`1.0289`

**Test（overall）**
- Accuracy：`0.4369`
- Balanced Acc：`0.4320`
- F1-macro：`0.4095`
- Precision-macro：`0.4805`
- Recall-macro：`0.4320`
- Recall idle/left/right：`0.2171` / `0.3511` / `0.7279`
- Precision idle/left/right：`0.6087` / `0.4220` / `0.4108`
- F1 idle/left/right：`0.3200` / `0.3833` / `0.5252`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     28     30     71
  true1     14     46     71
  true2      4     33     99
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4265`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4179`
- Val loss（最优时）：`1.0845`

**Test（overall）**
- Accuracy：`0.3719`
- Balanced Acc：`0.3794`
- F1-macro：`0.2883`
- Precision-macro：`0.4921`
- Recall-macro：`0.3794`
- Recall idle/left/right：`0.9147` / `0.0667` / `0.1567`
- Precision idle/left/right：`0.3401` / `0.5000` / `0.6364`
- F1 idle/left/right：`0.4958` / `0.1176` / `0.2515`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    118      4      7
  true1    121      9      5
  true2    108      5     21
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`48`
- 验证最优轮次（best_epoch）：`30`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4096`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3721`
- Val loss（最优时）：`1.1088`

**Test（overall）**
- Accuracy：`0.3538`
- Balanced Acc：`0.3509`
- F1-macro：`0.2987`
- Precision-macro：`0.5165`
- Recall-macro：`0.3509`
- Recall idle/left/right：`0.0469` / `0.3846` / `0.6212`
- Precision idle/left/right：`0.8571` / `0.3521` / `0.3402`
- F1 idle/left/right：`0.0889` / `0.3676` / `0.4397`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      6     43     79
  true1      0     50     80
  true2      1     49     82
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4668`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4202`
- Val loss（最优时）：`1.0812`

**Test（overall）**
- Accuracy：`0.4185`
- Balanced Acc：`0.4204`
- F1-macro：`0.4153`
- Precision-macro：`0.4198`
- Recall-macro：`0.4204`
- Recall idle/left/right：`0.5294` / `0.4065` / `0.3254`
- Precision idle/left/right：`0.4145` / `0.4132` / `0.4316`
- F1 idle/left/right：`0.4649` / `0.4098` / `0.3710`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     63     37     19
  true1     38     50     35
  true2     51     34     41
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3713`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3131`
- Val loss（最优时）：`1.0969`

**Test（overall）**
- Accuracy：`0.3054`
- Balanced Acc：`0.3333`
- F1-macro：`0.1567`
- Precision-macro：`0.1024`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- Precision idle/left/right：`0.3072` / `0.0000` / `0.0000`
- F1 idle/left/right：`0.4700` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     51      0      0
  true1     53      0      0
  true2     62      1      0
```

### 共用超参
```json
{
  "data_tag": "bci2a_2s",
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
  "protocol": "fixed-2s-cue2to4-bci2a",
  "early_stop": "balanced_accuracy",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true
}
```

- 结束：`2026-08-04T10:35:22`
