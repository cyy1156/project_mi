# 被试独立五折实验记录（20260804_102954 / conformer_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:29:54`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`conformer`（原结构）
- 结构：EEGConformer num_layers=2, num_heads=10, att_drop=0.5
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\conformer_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102954`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.6633 ± 0.0448`
- Test BalAcc：`0.6382 ± 0.0435`
- Test Spec：`0.6001 ± 0.2400`
- Test Rec：`0.6762 ± 0.2649`
- Test F1：`0.6867 ± 0.1953`
- Test Acc：`0.6505 ± 0.1080`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.7496`
- Val F1（最优 checkpoint 时，附报）：`0.8203`
- Val loss（最优时）：`0.5398`

**Test（overall）**
- Accuracy：`0.6869`
- Recall：`0.8052`
- Specificity：`0.4419`
- Precision：`0.7491`
- F1：`0.7762`
- Balanced Acc：`0.6236`
- 混淆矩阵：TP=`215` TN=`57` FP=`72` FN=`52`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.6633`
- Val F1（最优 checkpoint 时，附报）：`0.7631`
- Val loss（最优时）：`0.6850`

**Test（overall）**
- Accuracy：`0.4372`
- Recall：`0.1784`
- Specificity：`0.9767`
- Precision：`0.9412`
- F1：`0.3000`
- Balanced Acc：`0.5776`
- 混淆矩阵：TP=`48` TN=`126` FP=`3` FN=`221`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val 选模分数（Balanced Acc）：`0.6392`
- Val F1（最优 checkpoint 时，附报）：`0.7470`
- Val loss（最优时）：`0.7755`

**Test（overall）**
- Accuracy：`0.7256`
- Recall：`0.9122`
- Specificity：`0.3438`
- Precision：`0.7399`
- F1：`0.8171`
- Balanced Acc：`0.6280`
- 混淆矩阵：TP=`239` TN=`44` FP=`84` FN=`23`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.6374`
- Val F1（最优 checkpoint 时，附报）：`0.7739`
- Val loss（最优时）：`0.6540`

**Test（overall）**
- Accuracy：`0.7201`
- Recall：`0.8474`
- Specificity：`0.4538`
- Precision：`0.7645`
- F1：`0.8038`
- Balanced Acc：`0.6506`
- 混淆矩阵：TP=`211` TN=`54` FP=`65` FN=`38`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`70`
- 验证最优轮次（best_epoch）：`52`
- Val 选模分数（Balanced Acc）：`0.6269`
- Val F1（最优 checkpoint 时，附报）：`0.7899`
- Val loss（最优时）：`0.6793`

**Test（overall）**
- Accuracy：`0.6826`
- Recall：`0.6379`
- Specificity：`0.7843`
- Precision：`0.8706`
- F1：`0.7363`
- Balanced Acc：`0.7111`
- 混淆矩阵：TP=`74` TN=`40` FP=`11` FN=`42`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.5429 ± 0.0435`
- Val F1-macro：`0.5217 ± 0.0439`
- Test BalAcc：`0.5297 ± 0.1004`
- Test F1-macro：`0.5158 ± 0.1160`
- Test Acc：`0.5326 ± 0.1013`
- Test Precision-macro：`0.5386 ± 0.0939`
- Test Recall-macro：`0.5297 ± 0.1004`
- Test Recall idle/left/right：`0.3577±0.1772` / `0.6408±0.0931` / `0.5906±0.1798`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`84`
- 验证最优轮次（best_epoch）：`66`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.6133`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5724`
- Val loss（最优时）：`1.3803`

**Test（overall）**
- Accuracy：`0.5657`
- Balanced Acc：`0.5640`
- F1-macro：`0.5624`
- Precision-macro：`0.5696`
- Recall-macro：`0.5640`
- Recall idle/left/right：`0.5271` / `0.4809` / `0.6838`
- Precision idle/left/right：`0.4928` / `0.6238` / `0.5924`
- F1 idle/left/right：`0.5094` / `0.5431` / `0.6348`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     68     23     38
  true1     42     63     26
  true2     28     15     93
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`57`
- 验证最优轮次（best_epoch）：`39`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5688`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5655`
- Val loss（最优时）：`1.1572`

**Test（overall）**
- Accuracy：`0.6432`
- Balanced Acc：`0.6413`
- F1-macro：`0.6382`
- Precision-macro：`0.6375`
- Recall-macro：`0.6413`
- Recall idle/left/right：`0.4884` / `0.6444` / `0.7910`
- Precision idle/left/right：`0.5575` / `0.6084` / `0.7465`
- F1 idle/left/right：`0.5207` / `0.6259` / `0.7681`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     63     36     30
  true1     42     87      6
  true2      8     20    106
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5291`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5272`
- Val loss（最优时）：`1.1326`

**Test（overall）**
- Accuracy：`0.4179`
- Balanced Acc：`0.4164`
- F1-macro：`0.3892`
- Precision-macro：`0.4295`
- Recall-macro：`0.4164`
- Recall idle/left/right：`0.1484` / `0.6462` / `0.4545`
- Precision idle/left/right：`0.4222` / `0.3784` / `0.4878`
- F1 idle/left/right：`0.2197` / `0.4773` / `0.4706`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     19     78     31
  true1     14     84     32
  true2     12     60     60
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5131`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4804`
- Val loss（最优时）：`1.0266`

**Test（overall）**
- Accuracy：`0.4076`
- Balanced Acc：`0.4054`
- F1-macro：`0.3657`
- Precision-macro：`0.4252`
- Recall-macro：`0.4054`
- Recall idle/left/right：`0.1345` / `0.7724` / `0.3095`
- Precision idle/left/right：`0.3556` / `0.3785` / `0.5417`
- F1 idle/left/right：`0.1951` / `0.5080` / `0.3939`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     16     86     17
  true1     12     95     16
  true2     17     70     39
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`78`
- 验证最优轮次（best_epoch）：`60`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4905`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4633`
- Val loss（最优时）：`1.3199`

**Test（overall）**
- Accuracy：`0.6287`
- Balanced Acc：`0.6216`
- F1-macro：`0.6233`
- Precision-macro：`0.6314`
- Recall-macro：`0.6216`
- Recall idle/left/right：`0.4902` / `0.6604` / `0.7143`
- Precision idle/left/right：`0.4808` / `0.5645` / `0.8491`
- F1 idle/left/right：`0.4854` / `0.6087` / `0.7759`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     25     22      4
  true1     14     35      4
  true2     13      5     45
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

- 结束：`2026-08-04T10:32:35`
