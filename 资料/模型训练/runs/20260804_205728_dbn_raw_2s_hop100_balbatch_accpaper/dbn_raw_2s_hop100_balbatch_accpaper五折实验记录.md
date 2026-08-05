# 被试独立五折实验记录（20260804_205728 / dbn_raw_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T20:57:28`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-offline-native` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`dbn_raw` | TemporalEncoder(D=64) + DBN；2s/hop100 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\dbn_raw_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_205728`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.5217 ± 0.1473`
- Test Acc_paper：`0.4025 ± 0.1426`
- Test BalAcc_maj：`0.5050 ± 0.0104`
- Test 窗级 BalAcc（附报）：`0.5092 ± 0.0122`

### Task 分折明细

#### Fold 0

- stopped_epoch：`30` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4061`
- Val BalAcc_maj（附报）：`0.5612`

**Test 试次级**
- Acc_paper：`0.3258`
- BalAcc_maj：`0.5000`
- Acc_majority：`0.3258`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5010` | F1：`0.0039` | Acc：`0.3208`

#### Fold 1

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6947`
- Val BalAcc_maj（附报）：`0.5289`

**Test 试次级**
- Acc_paper：`0.3417`
- BalAcc_maj：`0.4908`
- Acc_majority：`0.3417`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.4965` | F1：`0.1451` | Acc：`0.3441`

#### Fold 2

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3211`
- Val BalAcc_maj（附报）：`0.5000`

**Test 试次级**
- Acc_paper：`0.3282`
- BalAcc_maj：`0.5000`
- Acc_majority：`0.3282`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.5006` | F1：`0.0025` | Acc：`0.3215`

#### Fold 3

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6789`
- Val BalAcc_maj（附报）：`0.5000`

**Test 试次级**
- Acc_paper：`0.6875`
- BalAcc_maj：`0.5168`
- Acc_majority：`0.6875`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.5212` | F1：`0.8188` | Acc：`0.6975`

#### Fold 4

- stopped_epoch：`98` | best_epoch：`80`
- Val Acc_paper（早停）：`0.5077`
- Val BalAcc_maj（附报）：`0.5776`

**Test 试次级**
- Acc_paper：`0.3293`
- BalAcc_maj：`0.5172`
- Acc_majority：`0.3293`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.5266` | F1：`0.1028` | Acc：`0.3373`

### Three
- Val Acc_paper：`0.3888 ± 0.0623`
- Test Acc_paper：`0.3411 ± 0.0183`
- Test BalAcc_maj：`0.3500 ± 0.0117`
- Test 窗级 BalAcc（附报）：`0.3531 ± 0.0126`

### Three 分折明细

#### Fold 0

- stopped_epoch：`66` | best_epoch：`48`
- Val Acc_paper（早停）：`0.4909`
- Val BalAcc_maj（附报）：`0.4984`

**Test 试次级**
- Acc_paper：`0.3561`
- BalAcc_maj：`0.3638`
- F1-macro（众数）：`0.2586`
- Rec idle/left/right：`0.9070` / `0.0153` / `0.1691`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.3685` | F1m：`0.2761`

#### Fold 1

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3316`
- Val BalAcc_maj（附报）：`0.3150`

**Test 试次级**
- Acc_paper：`0.3568`
- BalAcc_maj：`0.3602`
- F1-macro（众数）：`0.3077`
- Rec idle/left/right：`0.5659` / `0.0444` / `0.4701`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.3597` | F1m：`0.3124`

#### Fold 2

- stopped_epoch：`20` | best_epoch：`2`
- Val Acc_paper（早停）：`0.3211`
- Val BalAcc_maj（附报）：`0.3333`

**Test 试次级**
- Acc_paper：`0.3282`
- BalAcc_maj：`0.3333`
- F1-macro（众数）：`0.1647`
- Rec idle/left/right：`1.0000` / `0.0000` / `0.0000`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.3333` | F1m：`0.1619`

#### Fold 3

- stopped_epoch：`38` | best_epoch：`20`
- Val Acc_paper（早停）：`0.4211`
- Val BalAcc_maj（附报）：`0.4126`

**Test 试次级**
- Acc_paper：`0.3533`
- BalAcc_maj：`0.3531`
- F1-macro（众数）：`0.2916`
- Rec idle/left/right：`0.0168` / `0.4472` / `0.5952`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.3597` | F1m：`0.3121`

#### Fold 4

- stopped_epoch：`30` | best_epoch：`12`
- Val Acc_paper（早停）：`0.3795`
- Val BalAcc_maj（附报）：`0.3864`

**Test 试次级**
- Acc_paper：`0.3114`
- BalAcc_maj：`0.3396`
- F1-macro（众数）：`0.1698`
- Rec idle/left/right：`1.0000` / `0.0189` / `0.0000`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.3440` | F1m：`0.1784`

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

- 结束：`2026-08-04T21:18:31`
