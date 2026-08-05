# 被试独立五折实验记录（20260804_165940 / shallow_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T16:59:40`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-T-only` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\shallow_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_165940`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-T-only', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'bci2a_T_only': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7429 ± 0.0341`
- Test Acc_paper：`0.6576 ± 0.0455`
- Test BalAcc_maj：`0.6381 ± 0.0469`
- Test 窗级 BalAcc（附报）：`0.6047 ± 0.0341`

### Task 分折明细

#### Fold 0

- stopped_epoch：`21` | best_epoch：`3`
- Val Acc_paper（早停）：`0.8000`
- Val BalAcc_maj（附报）：`0.7657`

**Test 试次级**
- Acc_paper：`0.6010`
- BalAcc_maj：`0.5879`
- Acc_majority：`0.6010`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5622` | F1：`0.6445` | Acc：`0.5673`

#### Fold 1

- stopped_epoch：`61` | best_epoch：`43`
- Val Acc_paper（早停）：`0.7474`
- Val BalAcc_maj（附报）：`0.6800`

**Test 试次级**
- Acc_paper：`0.6181`
- BalAcc_maj：`0.6348`
- Acc_majority：`0.6181`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.6100` | F1：`0.6650` | Acc：`0.5995`

#### Fold 2

- stopped_epoch：`52` | best_epoch：`34`
- Val Acc_paper（早停）：`0.7105`
- Val BalAcc_maj（附报）：`0.6054`

**Test 试次级**
- Acc_paper：`0.7282`
- BalAcc_maj：`0.6678`
- Acc_majority：`0.7282`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.6162` | F1：`0.7541` | Acc：`0.6658`

#### Fold 3

- stopped_epoch：`46` | best_epoch：`28`
- Val Acc_paper（早停）：`0.7053`
- Val BalAcc_maj（附报）：`0.6490`

**Test 试次级**
- Acc_paper：`0.6821`
- BalAcc_maj：`0.5896`
- Acc_majority：`0.6821`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.5754` | F1：`0.7499` | Acc：`0.6478`

#### Fold 4

- stopped_epoch：`29` | best_epoch：`11`
- Val Acc_paper（早停）：`0.7513`
- Val BalAcc_maj（附报）：`0.6770`

**Test 试次级**
- Acc_paper：`0.6587`
- BalAcc_maj：`0.7104`
- Acc_majority：`0.6587`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.6595` | F1：`0.6752` | Acc：`0.6208`

### Three
- Val Acc_paper：`0.4921 ± 0.0390`
- Test Acc_paper：`0.4597 ± 0.0480`
- Test BalAcc_maj：`0.4954 ± 0.0448`
- Test 窗级 BalAcc（附报）：`0.4701 ± 0.0393`

### Three 分折明细

#### Fold 0

- stopped_epoch：`24` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5515`
- Val BalAcc_maj（附报）：`0.5984`

**Test 试次级**
- Acc_paper：`0.4444`
- BalAcc_maj：`0.4746`
- F1-macro（众数）：`0.4654`
- Rec idle/left/right：`0.4264` / `0.3282` / `0.6691`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.4592` | F1m：`0.4558`

#### Fold 1

- stopped_epoch：`40` | best_epoch：`22`
- Val Acc_paper（早停）：`0.5000`
- Val BalAcc_maj（附报）：`0.5378`

**Test 试次级**
- Acc_paper：`0.4472`
- BalAcc_maj：`0.4756`
- F1-macro（众数）：`0.4625`
- Rec idle/left/right：`0.7132` / `0.3556` / `0.3582`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.4637` | F1m：`0.4514`

#### Fold 2

- stopped_epoch：`30` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4474`
- Val BalAcc_maj（附报）：`0.4828`

**Test 试次级**
- Acc_paper：`0.4077`
- BalAcc_maj：`0.4597`
- F1-macro（众数）：`0.4404`
- Rec idle/left/right：`0.3984` / `0.7308` / `0.2500`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.4305` | F1m：`0.4170`

#### Fold 3

- stopped_epoch：`31` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5105`
- Val BalAcc_maj（附报）：`0.5276`

**Test 试次级**
- Acc_paper：`0.4484`
- BalAcc_maj：`0.4835`
- F1-macro（众数）：`0.4700`
- Rec idle/left/right：`0.4034` / `0.3171` / `0.7302`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.4517` | F1m：`0.4436`

#### Fold 4

- stopped_epoch：`23` | best_epoch：`5`
- Val Acc_paper（早停）：`0.4513`
- Val BalAcc_maj（附报）：`0.4763`

**Test 试次级**
- Acc_paper：`0.5509`
- BalAcc_maj：`0.5837`
- F1-macro（众数）：`0.5740`
- Rec idle/left/right：`0.7843` / `0.4906` / `0.4762`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.5452` | F1m：`0.5319`

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

- 结束：`2026-08-04T17:12:44`
