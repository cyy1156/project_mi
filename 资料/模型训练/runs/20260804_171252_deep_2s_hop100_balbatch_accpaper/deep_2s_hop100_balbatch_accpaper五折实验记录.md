# 被试独立五折实验记录（20260804_171252 / deep_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T17:12:52`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-T-only` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`deep` | Deep4Net-compat（pool=1/1，对齐1s自动缩核；非满血stride3）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\deep_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_171252`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-T-only', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'bci2a_T_only': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6455 ± 0.1460`
- Test Acc_paper：`0.5799 ± 0.1158`
- Test BalAcc_maj：`0.5705 ± 0.0328`
- Test 窗级 BalAcc（附报）：`0.5555 ± 0.0272`

### Task 分折明细

#### Fold 0

- stopped_epoch：`39` | best_epoch：`21`
- Val Acc_paper（早停）：`0.8303`
- Val BalAcc_maj（附报）：`0.7619`

**Test 试次级**
- Acc_paper：`0.6995`
- BalAcc_maj：`0.6009`
- Acc_majority：`0.6995`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5561` | F1：`0.7234` | Acc：`0.6194`

#### Fold 1

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3842`
- Val BalAcc_maj（附报）：`0.5249`

**Test 试次级**
- Acc_paper：`0.3593`
- BalAcc_maj：`0.5119`
- Acc_majority：`0.3593`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.5261` | F1：`0.3333` | Acc：`0.4118`

#### Fold 2

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6737`
- Val BalAcc_maj（附报）：`0.6257`

**Test 试次级**
- Acc_paper：`0.6256`
- BalAcc_maj：`0.5675`
- Acc_majority：`0.6256`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.5411` | F1：`0.6672` | Acc：`0.5730`

#### Fold 3

- stopped_epoch：`22` | best_epoch：`4`
- Val Acc_paper（早停）：`0.6368`
- Val BalAcc_maj（附报）：`0.5943`

**Test 试次级**
- Acc_paper：`0.6223`
- BalAcc_maj：`0.5695`
- Acc_majority：`0.6223`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.5481` | F1：`0.6732` | Acc：`0.5794`

#### Fold 4

- stopped_epoch：`58` | best_epoch：`40`
- Val Acc_paper（早停）：`0.7026`
- Val BalAcc_maj（附报）：`0.6508`

**Test 试次级**
- Acc_paper：`0.5928`
- BalAcc_maj：`0.6025`
- Acc_majority：`0.5928`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.6062` | F1：`0.6630` | Acc：`0.5926`

### Three
- Val Acc_paper：`0.4312 ± 0.0485`
- Test Acc_paper：`0.3950 ± 0.0635`
- Test BalAcc_maj：`0.4281 ± 0.0580`
- Test 窗级 BalAcc（附报）：`0.4131 ± 0.0345`

### Three 分折明细

#### Fold 0

- stopped_epoch：`24` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5273`
- Val BalAcc_maj（附报）：`0.5472`

**Test 试次级**
- Acc_paper：`0.4015`
- BalAcc_maj：`0.4050`
- F1-macro（众数）：`0.3222`
- Rec idle/left/right：`0.4651` / `0.0000` / `0.7500`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.3888` | F1m：`0.3398`

#### Fold 1

- stopped_epoch：`23` | best_epoch：`5`
- Val Acc_paper（早停）：`0.4000`
- Val BalAcc_maj（附报）：`0.3987`

**Test 试次级**
- Acc_paper：`0.4246`
- BalAcc_maj：`0.4358`
- F1-macro（众数）：`0.3474`
- Rec idle/left/right：`0.8372` / `0.0000` / `0.4701`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.4168` | F1m：`0.3457`

#### Fold 2

- stopped_epoch：`27` | best_epoch：`9`
- Val Acc_paper（早停）：`0.4000`
- Val BalAcc_maj（附报）：`0.4130`

**Test 试次级**
- Acc_paper：`0.2769`
- BalAcc_maj：`0.3393`
- F1-macro（众数）：`0.3204`
- Rec idle/left/right：`0.5156` / `0.1462` / `0.3561`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.3665` | F1m：`0.3589`

#### Fold 3

- stopped_epoch：`58` | best_epoch：`40`
- Val Acc_paper（早停）：`0.4158`
- Val BalAcc_maj（附报）：`0.4510`

**Test 试次级**
- Acc_paper：`0.4049`
- BalAcc_maj：`0.4419`
- F1-macro（众数）：`0.4203`
- Rec idle/left/right：`0.3277` / `0.2439` / `0.7540`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.4253` | F1m：`0.4132`

#### Fold 4

- stopped_epoch：`56` | best_epoch：`38`
- Val Acc_paper（早停）：`0.4128`
- Val BalAcc_maj（附报）：`0.4394`

**Test 试次级**
- Acc_paper：`0.4671`
- BalAcc_maj：`0.5184`
- F1-macro（众数）：`0.5166`
- Rec idle/left/right：`0.4118` / `0.6038` / `0.5397`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.4680` | F1m：`0.4655`

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

- 结束：`2026-08-04T19:08:07`
