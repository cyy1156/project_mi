# 被试独立五折实验记录（20260804_203236 / dbn_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T20:32:36`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-offline-native` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`dbn` | DBN + 2s μ/β log bandpower (N,8,2)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\dbn_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_203236`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7093 ± 0.0502`
- Test Acc_paper：`0.6036 ± 0.1197`
- Test BalAcc_maj：`0.5164 ± 0.0137`
- Test 窗级 BalAcc（附报）：`0.5185 ± 0.0175`

### Task 分折明细

#### Fold 0

- stopped_epoch：`33` | best_epoch：`15`
- Val Acc_paper（早停）：`0.8061`
- Val BalAcc_maj（附报）：`0.7494`

**Test 试次级**
- Acc_paper：`0.6490`
- BalAcc_maj：`0.5273`
- Acc_majority：`0.6490`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5157` | F1：`0.7487` | Acc：`0.6254`

#### Fold 1

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6789`
- Val BalAcc_maj（附报）：`0.5000`

**Test 试次级**
- Acc_paper：`0.6759`
- BalAcc_maj：`0.5000`
- Acc_majority：`0.6759`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.5000` | F1：`0.8121` | Acc：`0.6837`

#### Fold 2

- stopped_epoch：`24` | best_epoch：`6`
- Val Acc_paper（早停）：`0.7105`
- Val BalAcc_maj（附报）：`0.5621`

**Test 试次级**
- Acc_paper：`0.6513`
- BalAcc_maj：`0.5227`
- Acc_majority：`0.6513`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.5331` | F1：`0.7813` | Acc：`0.6613`

#### Fold 3

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6789`
- Val BalAcc_maj（附报）：`0.5000`

**Test 试次级**
- Acc_paper：`0.6766`
- BalAcc_maj：`0.5000`
- Acc_majority：`0.6766`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.5000` | F1：`0.8130` | Acc：`0.6850`

#### Fold 4

- stopped_epoch：`30` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6718`
- Val BalAcc_maj（附报）：`0.5040`

**Test 试次级**
- Acc_paper：`0.3653`
- BalAcc_maj：`0.5321`
- Acc_majority：`0.3653`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.5437` | F1：`0.2677` | Acc：`0.3896`

### Three
- Val Acc_paper：`0.4046 ± 0.0619`
- Test Acc_paper：`0.3250 ± 0.0135`
- Test BalAcc_maj：`0.3291 ± 0.0091`
- Test 窗级 BalAcc（附报）：`0.3331 ± 0.0093`

### Three 分折明细

#### Fold 0

- stopped_epoch：`38` | best_epoch：`20`
- Val Acc_paper（早停）：`0.5273`
- Val BalAcc_maj（附报）：`0.5301`

**Test 试次级**
- Acc_paper：`0.3460`
- BalAcc_maj：`0.3410`
- F1-macro（众数）：`0.2478`
- Rec idle/left/right：`0.1473` / `0.0229` / `0.8529`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.3496` | F1m：`0.2700`

#### Fold 1

- stopped_epoch：`25` | best_epoch：`7`
- Val Acc_paper（早停）：`0.3737`
- Val BalAcc_maj（附报）：`0.3625`

**Test 试次级**
- Acc_paper：`0.3241`
- BalAcc_maj：`0.3333`
- F1-macro（众数）：`0.1632`
- Rec idle/left/right：`1.0000` / `0.0000` / `0.0000`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.3332` | F1m：`0.1607`

#### Fold 2

- stopped_epoch：`21` | best_epoch：`3`
- Val Acc_paper（早停）：`0.3684`
- Val BalAcc_maj（附报）：`0.3585`

**Test 试次级**
- Acc_paper：`0.3179`
- BalAcc_maj：`0.3150`
- F1-macro（众数）：`0.2661`
- Rec idle/left/right：`0.0547` / `0.2615` / `0.6288`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.3277` | F1m：`0.2875`

#### Fold 3

- stopped_epoch：`28` | best_epoch：`10`
- Val Acc_paper（早停）：`0.3895`
- Val BalAcc_maj（附报）：`0.3734`

**Test 试次级**
- Acc_paper：`0.3315`
- BalAcc_maj：`0.3229`
- F1-macro（众数）：`0.1806`
- Rec idle/left/right：`0.0000` / `0.0244` / `0.9444`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.3218` | F1m：`0.1951`

#### Fold 4

- stopped_epoch：`23` | best_epoch：`5`
- Val Acc_paper（早停）：`0.3641`
- Val BalAcc_maj（附报）：`0.3601`

**Test 试次级**
- Acc_paper：`0.3054`
- BalAcc_maj：`0.3333`
- F1-macro（众数）：`0.1560`
- Rec idle/left/right：`1.0000` / `0.0000` / `0.0000`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.3333` | F1m：`0.1537`

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

- 结束：`2026-08-04T20:38:00`
