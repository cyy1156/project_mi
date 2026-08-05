# 被试独立五折实验记录（20260804_102108 / eegnet_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:21:08`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`eegnet`（原结构）
- 结构：EEGNet F1=8, D=2, F2=16（默认池化）
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\eegnet_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102108`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.7054 ± 0.0235`
- Test BalAcc：`0.6395 ± 0.0775`
- Test Spec：`0.6621 ± 0.1170`
- Test Rec：`0.6170 ± 0.0995`
- Test F1：`0.6905 ± 0.0814`
- Test Acc：`0.6312 ± 0.0775`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`97`
- 验证最优轮次（best_epoch）：`79`
- Val 选模分数（Balanced Acc）：`0.7520`
- Val F1（最优 checkpoint 时，附报）：`0.7407`
- Val loss（最优时）：`0.5873`

**Test（overall）**
- Accuracy：`0.6414`
- Recall：`0.6442`
- Specificity：`0.6357`
- Precision：`0.7854`
- F1：`0.7078`
- Balanced Acc：`0.6399`
- 混淆矩阵：TP=`172` TN=`82` FP=`47` FN=`95`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`68`
- 验证最优轮次（best_epoch）：`50`
- Val 选模分数（Balanced Acc）：`0.6935`
- Val F1（最优 checkpoint 时，附报）：`0.7489`
- Val loss（最优时）：`0.6612`

**Test（overall）**
- Accuracy：`0.6658`
- Recall：`0.5911`
- Specificity：`0.8217`
- Precision：`0.8736`
- F1：`0.7051`
- Balanced Acc：`0.7064`
- 混淆矩阵：TP=`159` TN=`106` FP=`23` FN=`110`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`48`
- Val 选模分数（Balanced Acc）：`0.6922`
- Val F1（最优 checkpoint 时，附报）：`0.7705`
- Val loss（最优时）：`0.6229`

**Test（overall）**
- Accuracy：`0.6436`
- Recall：`0.7137`
- Specificity：`0.5000`
- Precision：`0.7450`
- F1：`0.7290`
- Balanced Acc：`0.6069`
- 混淆矩阵：TP=`187` TN=`64` FP=`64` FN=`75`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`40`
- 验证最优轮次（best_epoch）：`22`
- Val 选模分数（Balanced Acc）：`0.6910`
- Val F1（最优 checkpoint 时，附报）：`0.7200`
- Val loss（最优时）：`0.6581`

**Test（overall）**
- Accuracy：`0.4864`
- Recall：`0.4378`
- Specificity：`0.5882`
- Precision：`0.6899`
- F1：`0.5356`
- Balanced Acc：`0.5130`
- 混淆矩阵：TP=`109` TN=`70` FP=`49` FN=`140`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc）：`0.6982`
- Val F1（最优 checkpoint 时，附报）：`0.7826`
- Val loss（最优时）：`0.6312`

**Test（overall）**
- Accuracy：`0.7186`
- Recall：`0.6983`
- Specificity：`0.7647`
- Precision：`0.8710`
- F1：`0.7751`
- Balanced Acc：`0.7315`
- 混淆矩阵：TP=`81` TN=`39` FP=`12` FN=`35`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.5060 ± 0.0789`
- Val F1-macro：`0.4976 ± 0.0799`
- Test BalAcc：`0.4383 ± 0.0779`
- Test F1-macro：`0.4134 ± 0.0869`
- Test Acc：`0.4348 ± 0.0752`
- Test Precision-macro：`0.4481 ± 0.0751`
- Test Recall-macro：`0.4383 ± 0.0779`
- Test Recall idle/left/right：`0.4937±0.1590` / `0.5826±0.1873` / `0.2388±0.1065`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`71`
- 验证最优轮次（best_epoch）：`53`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.6102`
- Val F1-macro（最优 checkpoint 时，附报）：`0.6065`
- Val loss（最优时）：`0.9283`

**Test（overall）**
- Accuracy：`0.4520`
- Balanced Acc：`0.4560`
- F1-macro：`0.4300`
- Precision-macro：`0.4719`
- Recall-macro：`0.4560`
- Recall idle/left/right：`0.5736` / `0.6031` / `0.1912`
- Precision idle/left/right：`0.4625` / `0.4225` / `0.5306`
- F1 idle/left/right：`0.5121` / `0.4969` / `0.2811`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     74     39     16
  true1     45     79      7
  true2     41     69     26
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`43`
- 验证最优轮次（best_epoch）：`25`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3937`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3892`
- Val loss（最优时）：`1.0968`

**Test（overall）**
- Accuracy：`0.4322`
- Balanced Acc：`0.4329`
- F1-macro：`0.4238`
- Precision-macro：`0.4236`
- Recall-macro：`0.4329`
- Recall idle/left/right：`0.5116` / `0.5333` / `0.2537`
- Precision idle/left/right：`0.4748` / `0.4417` / `0.3542`
- F1 idle/left/right：`0.4925` / `0.4832` / `0.2957`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     66     32     31
  true1     32     72     31
  true2     41     59     34
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`78`
- 验证最优轮次（best_epoch）：`60`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5800`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5720`
- Val loss（最优时）：`0.9959`

**Test（overall）**
- Accuracy：`0.3897`
- Balanced Acc：`0.3902`
- F1-macro：`0.3245`
- Precision-macro：`0.4418`
- Recall-macro：`0.3902`
- Recall idle/left/right：`0.1953` / `0.8692` / `0.1061`
- Precision idle/left/right：`0.6098` / `0.3657` / `0.3500`
- F1 idle/left/right：`0.2959` / `0.5148` / `0.1628`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     25     88     15
  true1      6    113     11
  true2     10    108     14
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4726`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4590`
- Val loss（最优时）：`1.0869`

**Test（overall）**
- Accuracy：`0.3370`
- Balanced Acc：`0.3399`
- F1-macro：`0.3269`
- Precision-macro：`0.3354`
- Recall-macro：`0.3399`
- Recall idle/left/right：`0.5210` / `0.2846` / `0.2143`
- Precision idle/left/right：`0.3388` / `0.3535` / `0.3140`
- F1 idle/left/right：`0.4106` / `0.3153` / `0.2547`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     62     29     28
  true1     57     35     31
  true2     64     35     27
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4738`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4616`
- Val loss（最优时）：`1.0647`

**Test（overall）**
- Accuracy：`0.5629`
- Balanced Acc：`0.5726`
- F1-macro：`0.5618`
- Precision-macro：`0.5677`
- Recall-macro：`0.5726`
- Recall idle/left/right：`0.6667` / `0.6226` / `0.4286`
- Precision idle/left/right：`0.5484` / `0.5410` / `0.6136`
- F1 idle/left/right：`0.6018` / `0.5789` / `0.5047`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     34     15      2
  true1      5     33     15
  true2     23     13     27
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

- 结束：`2026-08-04T10:25:12`
