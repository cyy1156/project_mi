# 被试独立五折实验记录（20260804_203813 / gcbnet_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T20:38:13`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-offline-native` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`gcbnet` | GCBNet(k=2, layers=[128]) + 2s bandpower
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\gcbnet_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_203813`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6845 ± 0.0466`
- Test Acc_paper：`0.5424 ± 0.1308`
- Test BalAcc_maj：`0.5272 ± 0.0210`
- Test 窗级 BalAcc（附报）：`0.5286 ± 0.0176`

### Task 分折明细

#### Fold 0

- stopped_epoch：`28` | best_epoch：`10`
- Val Acc_paper（早停）：`0.7636`
- Val BalAcc_maj（附报）：`0.6561`

**Test 试次级**
- Acc_paper：`0.5707`
- BalAcc_maj：`0.5194`
- Acc_majority：`0.5707`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5346` | F1：`0.6764` | Acc：`0.5764`

#### Fold 1

- stopped_epoch：`23` | best_epoch：`5`
- Val Acc_paper（早停）：`0.6789`
- Val BalAcc_maj（附报）：`0.5346`

**Test 试次级**
- Acc_paper：`0.4925`
- BalAcc_maj：`0.5277`
- Acc_majority：`0.4925`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.5235` | F1：`0.5257` | Acc：`0.4847`

#### Fold 2

- stopped_epoch：`20` | best_epoch：`2`
- Val Acc_paper（早停）：`0.7000`
- Val BalAcc_maj（附报）：`0.5544`

**Test 试次级**
- Acc_paper：`0.6410`
- BalAcc_maj：`0.5131`
- Acc_majority：`0.6410`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.5213` | F1：`0.7688` | Acc：`0.6455`

#### Fold 3

- stopped_epoch：`20` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6263`
- Val BalAcc_maj（附报）：`0.4699`

**Test 试次级**
- Acc_paper：`0.6902`
- BalAcc_maj：`0.5671`
- Acc_majority：`0.6902`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.5585` | F1：`0.7860` | Acc：`0.6746`

#### Fold 4

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6538`
- Val BalAcc_maj（附报）：`0.5106`

**Test 试次级**
- Acc_paper：`0.3174`
- BalAcc_maj：`0.5086`
- Acc_majority：`0.3174`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.5053` | F1：`0.0486` | Acc：`0.3128`

### Three
- Val Acc_paper：`0.3795 ± 0.0610`
- Test Acc_paper：`0.3353 ± 0.0215`
- Test BalAcc_maj：`0.3449 ± 0.0252`
- Test 窗级 BalAcc（附报）：`0.3551 ± 0.0262`

### Three 分折明细

#### Fold 0

- stopped_epoch：`24` | best_epoch：`6`
- Val Acc_paper（早停）：`0.4909`
- Val BalAcc_maj（附报）：`0.5284`

**Test 试次级**
- Acc_paper：`0.3333`
- BalAcc_maj：`0.3462`
- F1-macro（众数）：`0.3300`
- Rec idle/left/right：`0.2016` / `0.5649` / `0.2721`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.3391` | F1m：`0.3263`

#### Fold 1

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3684`
- Val BalAcc_maj（附报）：`0.3595`

**Test 试次级**
- Acc_paper：`0.3492`
- BalAcc_maj：`0.3613`
- F1-macro（众数）：`0.3385`
- Rec idle/left/right：`0.2868` / `0.1778` / `0.6194`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.3764` | F1m：`0.3609`

#### Fold 2

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3842`
- Val BalAcc_maj（附报）：`0.3760`

**Test 试次级**
- Acc_paper：`0.3385`
- BalAcc_maj：`0.3374`
- F1-macro（众数）：`0.2702`
- Rec idle/left/right：`0.0391` / `0.2308` / `0.7424`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.3503` | F1m：`0.2910`

#### Fold 3

- stopped_epoch：`20` | best_epoch：`2`
- Val Acc_paper（早停）：`0.3105`
- Val BalAcc_maj（附报）：`0.2983`

**Test 试次级**
- Acc_paper：`0.2962`
- BalAcc_maj：`0.3025`
- F1-macro（众数）：`0.2646`
- Rec idle/left/right：`0.0420` / `0.3496` / `0.5159`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.3180` | F1m：`0.2883`

#### Fold 4

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3436`
- Val BalAcc_maj（附报）：`0.3487`

**Test 试次级**
- Acc_paper：`0.3593`
- BalAcc_maj：`0.3772`
- F1-macro（众数）：`0.2571`
- Rec idle/left/right：`0.9412` / `0.0000` / `0.1905`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.3916` | F1m：`0.2782`

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

- 结束：`2026-08-04T20:47:47`
