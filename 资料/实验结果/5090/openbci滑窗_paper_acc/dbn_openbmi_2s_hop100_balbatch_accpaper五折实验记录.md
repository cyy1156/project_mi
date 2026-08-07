# 被试独立五折实验记录（20260806_190248 / dbn_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T19:02:48`
- device：`cuda`
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dbn` | DBN + 2s μ/β log bandpower (N,8,2)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\baseline_openbmi_2s_hop100_accpaper\dbn_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_190248`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 1}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6297 ± 0.0278`
- Test Acc_paper：`0.6210 ± 0.0437`
- Test BalAcc_maj：`0.5438 ± 0.0122`
- Test 窗级 BalAcc（附报）：`0.5412 ± 0.0113`

### Task 分折明细

#### Fold 0

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5893`
- Val BalAcc_maj（附报）：`0.5344`

**Test 试次级**
- Acc_paper：`0.6558`
- BalAcc_maj：`0.5355`
- Acc_majority：`0.6558`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5349` | F1：`0.7690` | Acc：`0.6488`

#### Fold 1

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.6419`
- Val BalAcc_maj（附报）：`0.5619`

**Test 试次级**
- Acc_paper：`0.6609`
- BalAcc_maj：`0.5545`
- Acc_majority：`0.6609`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5518` | F1：`0.7635` | Acc：`0.6502`

#### Fold 2

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.6044`
- Val BalAcc_maj（附报）：`0.5433`

**Test 试次级**
- Acc_paper：`0.5421`
- BalAcc_maj：`0.5498`
- Acc_majority：`0.5421`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5417` | F1：`0.6002` | Acc：`0.5354`

#### Fold 3

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6578`
- Val BalAcc_maj（附报）：`0.5581`

**Test 试次级**
- Acc_paper：`0.6394`
- BalAcc_maj：`0.5550`
- Acc_majority：`0.6394`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5541` | F1：`0.7458` | Acc：`0.6362`

#### Fold 4

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.6552`
- Val BalAcc_maj（附报）：`0.5225`

**Test 试次级**
- Acc_paper：`0.6070`
- BalAcc_maj：`0.5240`
- Acc_majority：`0.6070`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5234` | F1：`0.7221` | Acc：`0.6053`

### Three
- Val Acc_paper：`0.3813 ± 0.0117`
- Test Acc_paper：`0.3809 ± 0.0155`
- Test BalAcc_maj：`0.3944 ± 0.0181`
- Test 窗级 BalAcc（附报）：`0.3893 ± 0.0151`

### Three 分折明细

#### Fold 0

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.3767`
- Val BalAcc_maj（附报）：`0.3911`

**Test 试次级**
- Acc_paper：`0.3718`
- BalAcc_maj：`0.3861`
- F1-macro（众数）：`0.3762`
- Rec idle/left/right：`0.2609` / `0.3345` / `0.5627`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3813` | F1m：`0.3721`

#### Fold 1

- stopped_epoch：`61` | best_epoch：`41`
- Val Acc_paper（早停）：`0.3970`
- Val BalAcc_maj（附报）：`0.4130`

**Test 试次级**
- Acc_paper：`0.4000`
- BalAcc_maj：`0.4182`
- F1-macro（众数）：`0.4170`
- Rec idle/left/right：`0.4282` / `0.4709` / `0.3555`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4042` | F1m：`0.4033`

#### Fold 2

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.3659`
- Val BalAcc_maj（附报）：`0.3770`

**Test 试次级**
- Acc_paper：`0.3585`
- BalAcc_maj：`0.3664`
- F1-macro（众数）：`0.3057`
- Rec idle/left/right：`0.7791` / `0.2427` / `0.0773`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3641` | F1m：`0.3142`

#### Fold 3

- stopped_epoch：`50` | best_epoch：`30`
- Val Acc_paper（早停）：`0.3926`
- Val BalAcc_maj（附报）：`0.4074`

**Test 试次级**
- Acc_paper：`0.3964`
- BalAcc_maj：`0.4091`
- F1-macro（众数）：`0.3957`
- Rec idle/left/right：`0.5873` / `0.4073` / `0.2327`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4036` | F1m：`0.3943`

#### Fold 4

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.3741`
- Val BalAcc_maj（附报）：`0.3896`

**Test 试次级**
- Acc_paper：`0.3780`
- BalAcc_maj：`0.3923`
- F1-macro（众数）：`0.3835`
- Rec idle/left/right：`0.5380` / `0.3890` / `0.2500`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.3934` | F1m：`0.3869`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "non_blocking": true,
  "use_amp": true,
  "cudnn_benchmark": false,
  "gpu_memory_fraction": 1
}
```

- 结束：`2026-08-06T20:44:53`
