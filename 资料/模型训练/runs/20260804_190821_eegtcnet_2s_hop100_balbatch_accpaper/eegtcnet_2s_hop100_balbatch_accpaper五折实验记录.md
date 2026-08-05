# 被试独立五折实验记录（20260804_190821 / eegtcnet_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T19:08:21`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-T-only` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`eegtcnet` | EEGTCNet（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\eegtcnet_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_190821`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-T-only', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'bci2a_T_only': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6930 ± 0.0701`
- Test Acc_paper：`0.6087 ± 0.1224`
- Test BalAcc_maj：`0.5993 ± 0.0357`
- Test 窗级 BalAcc（附报）：`0.5791 ± 0.0249`

### Task 分折明细

#### Fold 0

- stopped_epoch：`70` | best_epoch：`52`
- Val Acc_paper（早停）：`0.8000`
- Val BalAcc_maj（附报）：`0.7242`

**Test 试次级**
- Acc_paper：`0.6288`
- BalAcc_maj：`0.6005`
- Acc_majority：`0.6288`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5723` | F1：`0.6720` | Acc：`0.5890`

#### Fold 1

- stopped_epoch：`48` | best_epoch：`30`
- Val Acc_paper（早停）：`0.6000`
- Val BalAcc_maj（附报）：`0.5196`

**Test 试次级**
- Acc_paper：`0.5829`
- BalAcc_maj：`0.6027`
- Acc_majority：`0.5829`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.5927` | F1：`0.6349` | Acc：`0.5740`

#### Fold 2

- stopped_epoch：`30` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6368`
- Val BalAcc_maj（附报）：`0.5295`

**Test 试次级**
- Acc_paper：`0.7333`
- BalAcc_maj：`0.6617`
- Acc_majority：`0.7333`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.6099` | F1：`0.7546` | Acc：`0.6639`

#### Fold 3

- stopped_epoch：`23` | best_epoch：`5`
- Val Acc_paper（早停）：`0.7000`
- Val BalAcc_maj（附报）：`0.5371`

**Test 试次级**
- Acc_paper：`0.7092`
- BalAcc_maj：`0.5767`
- Acc_majority：`0.7092`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.5357` | F1：`0.7530` | Acc：`0.6366`

#### Fold 4

- stopped_epoch：`41` | best_epoch：`23`
- Val Acc_paper（早停）：`0.7282`
- Val BalAcc_maj（附报）：`0.6059`

**Test 试次级**
- Acc_paper：`0.3892`
- BalAcc_maj：`0.5549`
- Acc_majority：`0.3892`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.5848` | F1：`0.3776` | Acc：`0.4465`

### Three
- Val Acc_paper：`0.4272 ± 0.0707`
- Test Acc_paper：`0.3803 ± 0.0240`
- Test BalAcc_maj：`0.4057 ± 0.0283`
- Test 窗级 BalAcc（附报）：`0.4010 ± 0.0214`

### Three 分折明细

#### Fold 0

- stopped_epoch：`70` | best_epoch：`52`
- Val Acc_paper（早停）：`0.5576`
- Val BalAcc_maj（附报）：`0.6033`

**Test 试次级**
- Acc_paper：`0.4091`
- BalAcc_maj：`0.4550`
- F1-macro（众数）：`0.4503`
- Rec idle/left/right：`0.3178` / `0.4809` / `0.5662`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.4304` | F1m：`0.4292`

#### Fold 1

- stopped_epoch：`29` | best_epoch：`11`
- Val Acc_paper（早停）：`0.3421`
- Val BalAcc_maj（附报）：`0.3709`

**Test 试次级**
- Acc_paper：`0.4020`
- BalAcc_maj：`0.4138`
- F1-macro（众数）：`0.3110`
- Rec idle/left/right：`0.9302` / `0.3111` / `0.0000`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.4156` | F1m：`0.3374`

#### Fold 2

- stopped_epoch：`20` | best_epoch：`2`
- Val Acc_paper（早停）：`0.4053`
- Val BalAcc_maj（附报）：`0.4170`

**Test 试次级**
- Acc_paper：`0.3846`
- BalAcc_maj：`0.3863`
- F1-macro（众数）：`0.2965`
- Rec idle/left/right：`0.3281` / `0.8308` / `0.0000`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.3667` | F1m：`0.2895`

#### Fold 3

- stopped_epoch：`46` | best_epoch：`28`
- Val Acc_paper（早停）：`0.4158`
- Val BalAcc_maj（附报）：`0.4457`

**Test 试次级**
- Acc_paper：`0.3587`
- BalAcc_maj：`0.4015`
- F1-macro（众数）：`0.3854`
- Rec idle/left/right：`0.4454` / `0.5528` / `0.2063`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.3954` | F1m：`0.3876`

#### Fold 4

- stopped_epoch：`114` | best_epoch：`96`
- Val Acc_paper（早停）：`0.4154`
- Val BalAcc_maj（附报）：`0.4353`

**Test 试次级**
- Acc_paper：`0.3473`
- BalAcc_maj：`0.3721`
- F1-macro（众数）：`0.2441`
- Rec idle/left/right：`0.9804` / `0.0566` / `0.0794`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.3968` | F1m：`0.2888`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-T-only",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "bci2a_T_only": true
}
```

- 结束：`2026-08-04T19:52:47`
