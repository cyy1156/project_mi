# 被试独立五折实验记录（20260804_211840 / gcbnet_raw_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T21:18:40`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-offline-native` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`gcbnet_raw` | TemporalEncoder(D=64) + GCBNet(k=2)；2s/hop100 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\gcbnet_raw_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_211840`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.5776 ± 0.1272`
- Test Acc_paper：`0.3983 ± 0.0809`
- Test BalAcc_maj：`0.5177 ± 0.0335`
- Test 窗级 BalAcc（附报）：`0.5219 ± 0.0304`

### Task 分折明细

#### Fold 0

- stopped_epoch：`60` | best_epoch：`42`
- Val Acc_paper（早停）：`0.7636`
- Val BalAcc_maj（附报）：`0.6977`

**Test 试次级**
- Acc_paper：`0.5328`
- BalAcc_maj：`0.5754`
- Acc_majority：`0.5328`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5730` | F1：`0.5763` | Acc：`0.5342`

#### Fold 1

- stopped_epoch：`39` | best_epoch：`21`
- Val Acc_paper（早停）：`0.6737`
- Val BalAcc_maj（附报）：`0.6474`

**Test 试次级**
- Acc_paper：`0.4397`
- BalAcc_maj：`0.5169`
- Acc_majority：`0.4397`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.5224` | F1：`0.4366` | Acc：`0.4458`

#### Fold 2

- stopped_epoch：`25` | best_epoch：`7`
- Val Acc_paper（早停）：`0.5632`
- Val BalAcc_maj（附报）：`0.5098`

**Test 试次级**
- Acc_paper：`0.3359`
- BalAcc_maj：`0.4738`
- Acc_majority：`0.3359`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.4835` | F1：`0.1820` | Acc：`0.3483`

#### Fold 3

- stopped_epoch：`20` | best_epoch：`2`
- Val Acc_paper（早停）：`0.4211`
- Val BalAcc_maj（附报）：`0.4829`

**Test 试次级**
- Acc_paper：`0.3777`
- BalAcc_maj：`0.5226`
- Acc_majority：`0.3777`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.5301` | F1：`0.3049` | Acc：`0.4045`

#### Fold 4

- stopped_epoch：`25` | best_epoch：`7`
- Val Acc_paper（早停）：`0.4667`
- Val BalAcc_maj（附报）：`0.5251`

**Test 试次级**
- Acc_paper：`0.3054`
- BalAcc_maj：`0.5000`
- Acc_majority：`0.3054`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.5005` | F1：`0.0098` | Acc：`0.3019`

### Three
- Val Acc_paper：`0.4266 ± 0.0615`
- Test Acc_paper：`0.3475 ± 0.0219`
- Test BalAcc_maj：`0.3734 ± 0.0248`
- Test 窗级 BalAcc（附报）：`0.3779 ± 0.0195`

### Three 分折明细

#### Fold 0

- stopped_epoch：`37` | best_epoch：`19`
- Val Acc_paper（早停）：`0.5455`
- Val BalAcc_maj（附报）：`0.5805`

**Test 试次级**
- Acc_paper：`0.3460`
- BalAcc_maj：`0.3593`
- F1-macro（众数）：`0.3110`
- Rec idle/left/right：`0.7287` / `0.0992` / `0.2500`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.3755` | F1m：`0.3386`

#### Fold 1

- stopped_epoch：`34` | best_epoch：`16`
- Val Acc_paper（早停）：`0.4053`
- Val BalAcc_maj（附报）：`0.4287`

**Test 试次级**
- Acc_paper：`0.3794`
- BalAcc_maj：`0.4100`
- F1-macro（众数）：`0.3986`
- Rec idle/left/right：`0.6202` / `0.2815` / `0.3284`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.4056` | F1m：`0.3939`

#### Fold 2

- stopped_epoch：`31` | best_epoch：`13`
- Val Acc_paper（早停）：`0.4105`
- Val BalAcc_maj（附报）：`0.4085`

**Test 试次级**
- Acc_paper：`0.3154`
- BalAcc_maj：`0.3434`
- F1-macro（众数）：`0.3112`
- Rec idle/left/right：`0.6562` / `0.1846` / `0.1894`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.3478` | F1m：`0.3270`

#### Fold 3

- stopped_epoch：`21` | best_epoch：`3`
- Val Acc_paper（早停）：`0.4053`
- Val BalAcc_maj（附报）：`0.4061`

**Test 试次级**
- Acc_paper：`0.3614`
- BalAcc_maj：`0.3945`
- F1-macro（众数）：`0.3844`
- Rec idle/left/right：`0.4958` / `0.2195` / `0.4683`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.3906` | F1m：`0.3841`

#### Fold 4

- stopped_epoch：`41` | best_epoch：`23`
- Val Acc_paper（早停）：`0.3667`
- Val BalAcc_maj（附报）：`0.3949`

**Test 试次级**
- Acc_paper：`0.3353`
- BalAcc_maj：`0.3598`
- F1-macro（众数）：`0.2088`
- Rec idle/left/right：`1.0000` / `0.0000` / `0.0794`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.3699` | F1m：`0.2301`

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

- 结束：`2026-08-04T21:45:30`
