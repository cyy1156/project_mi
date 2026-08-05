# 被试独立五折实验记录（20260804_160649 / eegnet_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T16:06:49`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-T-only` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`eegnet` | EEGNet F1=8, D=2, F2=16（默认池化）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\eegnet_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_160649`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-T-only', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'bci2a_T_only': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6965 ± 0.0475`
- Test Acc_paper：`0.5868 ± 0.0794`
- Test BalAcc_maj：`0.6019 ± 0.0608`
- Test 窗级 BalAcc（附报）：`0.5785 ± 0.0381`

### Task 分折明细

#### Fold 0

- stopped_epoch：`32` | best_epoch：`14`
- Val Acc_paper（早停）：`0.7394`
- Val BalAcc_maj（附报）：`0.7319`

**Test 试次级**
- Acc_paper：`0.6035`
- BalAcc_maj：`0.5838`
- Acc_majority：`0.6035`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5675` | F1：`0.6628` | Acc：`0.5810`

#### Fold 1

- stopped_epoch：`21` | best_epoch：`3`
- Val Acc_paper（早停）：`0.6684`
- Val BalAcc_maj（附报）：`0.5916`

**Test 试次级**
- Acc_paper：`0.6457`
- BalAcc_maj：`0.6855`
- Acc_majority：`0.6457`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.6167` | F1：`0.6428` | Acc：`0.5889`

#### Fold 2

- stopped_epoch：`22` | best_epoch：`4`
- Val Acc_paper（早停）：`0.6368`
- Val BalAcc_maj（附报）：`0.5727`

**Test 试次级**
- Acc_paper：`0.6897`
- BalAcc_maj：`0.6252`
- Acc_majority：`0.6897`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.5847` | F1：`0.7282` | Acc：`0.6339`

#### Fold 3

- stopped_epoch：`20` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6737`
- Val BalAcc_maj（附报）：`0.6430`

**Test 试次级**
- Acc_paper：`0.5217`
- BalAcc_maj：`0.4996`
- Acc_majority：`0.5217`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.5115` | F1：`0.5985` | Acc：`0.5168`

#### Fold 4

- stopped_epoch：`40` | best_epoch：`22`
- Val Acc_paper（早停）：`0.7641`
- Val BalAcc_maj（附报）：`0.6866`

**Test 试次级**
- Acc_paper：`0.4731`
- BalAcc_maj：`0.6152`
- Acc_majority：`0.4731`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.6123` | F1：`0.4479` | Acc：`0.4862`

### Three
- Val Acc_paper：`0.4630 ± 0.0530`
- Test Acc_paper：`0.4201 ± 0.0701`
- Test BalAcc_maj：`0.4521 ± 0.0643`
- Test 窗级 BalAcc（附报）：`0.4421 ± 0.0530`

### Three 分折明细

#### Fold 0

- stopped_epoch：`35` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5636`
- Val BalAcc_maj（附报）：`0.6048`

**Test 试次级**
- Acc_paper：`0.3914`
- BalAcc_maj：`0.4547`
- F1-macro（众数）：`0.4514`
- Rec idle/left/right：`0.3333` / `0.4351` / `0.5956`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.4319` | F1m：`0.4302`

#### Fold 1

- stopped_epoch：`43` | best_epoch：`25`
- Val Acc_paper（早停）：`0.4105`
- Val BalAcc_maj（附报）：`0.4326`

**Test 试次级**
- Acc_paper：`0.4447`
- BalAcc_maj：`0.4646`
- F1-macro（众数）：`0.4378`
- Rec idle/left/right：`0.8140` / `0.2963` / `0.2836`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.4606` | F1m：`0.4402`

#### Fold 2

- stopped_epoch：`27` | best_epoch：`9`
- Val Acc_paper（早停）：`0.4316`
- Val BalAcc_maj（附报）：`0.4707`

**Test 试次级**
- Acc_paper：`0.3718`
- BalAcc_maj：`0.4084`
- F1-macro（众数）：`0.3479`
- Rec idle/left/right：`0.2344` / `0.8846` / `0.1061`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.4002` | F1m：`0.3619`

#### Fold 3

- stopped_epoch：`22` | best_epoch：`4`
- Val Acc_paper（早停）：`0.4526`
- Val BalAcc_maj（附报）：`0.4583`

**Test 试次级**
- Acc_paper：`0.3478`
- BalAcc_maj：`0.3712`
- F1-macro（众数）：`0.3391`
- Rec idle/left/right：`0.4286` / `0.1057` / `0.5794`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.3836` | F1m：`0.3633`

#### Fold 4

- stopped_epoch：`89` | best_epoch：`71`
- Val Acc_paper（早停）：`0.4564`
- Val BalAcc_maj（附报）：`0.4710`

**Test 试次级**
- Acc_paper：`0.5449`
- BalAcc_maj：`0.5618`
- F1-macro（众数）：`0.5166`
- Rec idle/left/right：`0.9412` / `0.1887` / `0.5556`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.5340` | F1m：`0.4919`

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

- 结束：`2026-08-04T16:59:29`
