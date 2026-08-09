# 被试独立五折实验记录（20260806_172218 / eegnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T17:22:18`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`eegnet` | EEGNet F1=8, D=2, F2=16（默认池化）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\eegnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_172218`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6886 ± 0.0165`
- Test Acc_paper：`0.6869 ± 0.0334`
- Test BalAcc_maj：`0.6597 ± 0.0275`
- Test 窗级 BalAcc（附报）：`0.6373 ± 0.0214`

### Task 分折明细

#### Fold 0

- stopped_epoch：`84` | best_epoch：`64`
- Val Acc_paper（早停）：`0.7004`
- Val BalAcc_maj（附报）：`0.6675`

**Test 试次级**
- Acc_paper：`0.6967`
- BalAcc_maj：`0.6634`
- Acc_majority：`0.6967`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6359` | F1：`0.7446` | Acc：`0.6668`

#### Fold 1

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.7078`
- Val BalAcc_maj（附报）：`0.6817`

**Test 试次级**
- Acc_paper：`0.6748`
- BalAcc_maj：`0.6584`
- Acc_majority：`0.6748`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6392` | F1：`0.7236` | Acc：`0.6532`

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.6856`
- Val BalAcc_maj（附报）：`0.6589`

**Test 试次级**
- Acc_paper：`0.6415`
- BalAcc_maj：`0.6214`
- Acc_majority：`0.6415`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6037` | F1：`0.7015` | Acc：`0.6236`

#### Fold 3

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.6896`
- Val BalAcc_maj（附报）：`0.6875`

**Test 试次级**
- Acc_paper：`0.7433`
- BalAcc_maj：`0.7064`
- Acc_majority：`0.7433`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6712` | F1：`0.7753` | Acc：`0.7034`

#### Fold 4

- stopped_epoch：`114` | best_epoch：`94`
- Val Acc_paper（早停）：`0.6596`
- Val BalAcc_maj（附报）：`0.5844`

**Test 试次级**
- Acc_paper：`0.6780`
- BalAcc_maj：`0.6488`
- Acc_majority：`0.6780`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6364` | F1：`0.7426` | Acc：`0.6655`

### Three
- Val Acc_paper：`0.5107 ± 0.0358`
- Test Acc_paper：`0.5322 ± 0.0291`
- Test BalAcc_maj：`0.5468 ± 0.0290`
- Test 窗级 BalAcc（附报）：`0.5255 ± 0.0218`

### Three 分折明细

#### Fold 0

- stopped_epoch：`96` | best_epoch：`76`
- Val Acc_paper（早停）：`0.5133`
- Val BalAcc_maj（附报）：`0.5226`

**Test 试次级**
- Acc_paper：`0.5091`
- BalAcc_maj：`0.5230`
- F1-macro（众数）：`0.5231`
- Rec idle/left/right：`0.5318` / `0.5136` / `0.5236`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5069` | F1m：`0.5069`

#### Fold 1

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5307`
- Val BalAcc_maj（附报）：`0.5448`

**Test 试次级**
- Acc_paper：`0.5679`
- BalAcc_maj：`0.5788`
- F1-macro（众数）：`0.5779`
- Rec idle/left/right：`0.4855` / `0.6536` / `0.5973`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5509` | F1m：`0.5492`

#### Fold 2

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.4937`
- Val BalAcc_maj（附报）：`0.5107`

**Test 试次级**
- Acc_paper：`0.4915`
- BalAcc_maj：`0.5061`
- F1-macro（众数）：`0.4972`
- Rec idle/left/right：`0.4209` / `0.7091` / `0.3882`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4951` | F1m：`0.4878`

#### Fold 3

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5611`
- Val BalAcc_maj（附报）：`0.5781`

**Test 试次级**
- Acc_paper：`0.5600`
- BalAcc_maj：`0.5779`
- F1-macro（众数）：`0.5767`
- Rec idle/left/right：`0.4918` / `0.5645` / `0.6773`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5468` | F1m：`0.5447`

#### Fold 4

- stopped_epoch：`84` | best_epoch：`64`
- Val Acc_paper（早停）：`0.4544`
- Val BalAcc_maj（附报）：`0.4759`

**Test 试次级**
- Acc_paper：`0.5323`
- BalAcc_maj：`0.5483`
- F1-macro（众数）：`0.5473`
- Rec idle/left/right：`0.5580` / `0.6150` / `0.4720`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5277` | F1m：`0.5267`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 20,
  "batch_train": 128,
  "batch_eval": 256,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "prefetch_factor": 2,
  "non_blocking": true,
  "torch_num_threads": 6,
  "cudnn_benchmark": true,
  "deterministic": false,
  "use_amp": true
}
```

- 结束：`2026-08-06T20:49:51`
