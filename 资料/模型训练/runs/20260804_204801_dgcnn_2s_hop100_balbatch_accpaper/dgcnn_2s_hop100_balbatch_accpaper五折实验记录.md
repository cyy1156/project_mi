# 被试独立五折实验记录（20260804_204801 / dgcnn_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T20:48:01`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-offline-native` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`dgcnn` | DGCNN(k=2, layers=[128]) + 2s bandpower
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\dgcnn_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_204801`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6812 ± 0.0400`
- Test Acc_paper：`0.5542 ± 0.1238`
- Test BalAcc_maj：`0.5132 ± 0.0136`
- Test 窗级 BalAcc（附报）：`0.5267 ± 0.0098`

### Task 分折明细

#### Fold 0

- stopped_epoch：`33` | best_epoch：`15`
- Val Acc_paper（早停）：`0.7515`
- Val BalAcc_maj（附报）：`0.6162`

**Test 试次级**
- Acc_paper：`0.6288`
- BalAcc_maj：`0.5404`
- Acc_majority：`0.6288`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5419` | F1：`0.7381` | Acc：`0.6259`

#### Fold 1

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6737`
- Val BalAcc_maj（附报）：`0.5004`

**Test 试次级**
- Acc_paper：`0.5302`
- BalAcc_maj：`0.5052`
- Acc_majority：`0.5302`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.5217` | F1：`0.6359` | Acc：`0.5440`

#### Fold 2

- stopped_epoch：`20` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6842`
- Val BalAcc_maj（附报）：`0.5514`

**Test 试次级**
- Acc_paper：`0.6231`
- BalAcc_maj：`0.5057`
- Acc_majority：`0.6231`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.5137` | F1：`0.7494` | Acc：`0.6253`

#### Fold 3

- stopped_epoch：`39` | best_epoch：`21`
- Val Acc_paper（早停）：`0.6684`
- Val BalAcc_maj（附报）：`0.5009`

**Test 试次级**
- Acc_paper：`0.6658`
- BalAcc_maj：`0.5073`
- Acc_majority：`0.6658`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.5232` | F1：`0.7952` | Acc：`0.6728`

#### Fold 4

- stopped_epoch：`21` | best_epoch：`3`
- Val Acc_paper（早停）：`0.6282`
- Val BalAcc_maj（附报）：`0.5055`

**Test 试次级**
- Acc_paper：`0.3234`
- BalAcc_maj：`0.5074`
- Acc_majority：`0.3234`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.5330` | F1：`0.1937` | Acc：`0.3631`

### Three
- Val Acc_paper：`0.3565 ± 0.0387`
- Test Acc_paper：`0.3616 ± 0.0356`
- Test BalAcc_maj：`0.3731 ± 0.0329`
- Test 窗级 BalAcc（附报）：`0.3674 ± 0.0262`

### Three 分折明细

#### Fold 0

- stopped_epoch：`24` | best_epoch：`6`
- Val Acc_paper（早停）：`0.3939`
- Val BalAcc_maj（附报）：`0.4094`

**Test 试次级**
- Acc_paper：`0.3157`
- BalAcc_maj：`0.3507`
- F1-macro（众数）：`0.3438`
- Rec idle/left/right：`0.4729` / `0.3511` / `0.2279`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.3593` | F1m：`0.3544`

#### Fold 1

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.2947`
- Val BalAcc_maj（附报）：`0.3074`

**Test 试次级**
- Acc_paper：`0.3995`
- BalAcc_maj：`0.4078`
- F1-macro（众数）：`0.4072`
- Rec idle/left/right：`0.4574` / `0.3704` / `0.3955`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.4004` | F1m：`0.3991`

#### Fold 2

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3947`
- Val BalAcc_maj（附报）：`0.3798`

**Test 试次级**
- Acc_paper：`0.3462`
- BalAcc_maj：`0.3473`
- F1-macro（众数）：`0.2643`
- Rec idle/left/right：`0.0625` / `0.1308` / `0.8485`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.3377` | F1m：`0.2703`

#### Fold 3

- stopped_epoch：`42` | best_epoch：`24`
- Val Acc_paper（早停）：`0.3684`
- Val BalAcc_maj（附报）：`0.3609`

**Test 试次级**
- Acc_paper：`0.3397`
- BalAcc_maj：`0.3415`
- F1-macro（众数）：`0.2759`
- Rec idle/left/right：`0.0168` / `0.3252` / `0.6825`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.3434` | F1m：`0.2936`

#### Fold 4

- stopped_epoch：`20` | best_epoch：`2`
- Val Acc_paper（早停）：`0.3308`
- Val BalAcc_maj（附报）：`0.3322`

**Test 试次级**
- Acc_paper：`0.4072`
- BalAcc_maj：`0.4183`
- F1-macro（众数）：`0.3123`
- Rec idle/left/right：`0.9216` / `0.0000` / `0.3333`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.3960` | F1m：`0.3083`

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

- 结束：`2026-08-04T20:57:22`
