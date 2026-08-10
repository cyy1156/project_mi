# 被试独立五折实验记录（20260810_172757 / gcbnet_raw_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T17:27:57`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`gcbnet_raw` | GCBNet（raw 版）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\gcbnet_raw_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_172757`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6684 ± 0.0261`
- Test Acc_paper：`0.6652 ± 0.0121`
- Test BalAcc_maj：`0.5664 ± 0.0285`
- Test 窗级 BalAcc（附报）：`0.5664 ± 0.0285`

### Task 分折明细

#### Fold 0

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.6800`
- Val BalAcc_maj（附报）：`0.5750`

**Test 试次级**
- Acc_paper：`0.6479`
- BalAcc_maj：`0.5520`
- Acc_majority：`0.6479`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5520` | F1：`0.7607` | Acc：`0.6479`

#### Fold 1

- stopped_epoch：`54` | best_epoch：`34`
- Val Acc_paper（早停）：`0.6915`
- Val BalAcc_maj（附报）：`0.5861`

**Test 试次级**
- Acc_paper：`0.6776`
- BalAcc_maj：`0.6177`
- Acc_majority：`0.6776`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6177` | F1：`0.7673` | Acc：`0.6776`

#### Fold 2

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.6315`
- Val BalAcc_maj（附报）：`0.5358`

**Test 试次级**
- Acc_paper：`0.6591`
- BalAcc_maj：`0.5334`
- Acc_majority：`0.6591`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5334` | F1：`0.7807` | Acc：`0.6591`

#### Fold 3

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6956`
- Val BalAcc_maj（附报）：`0.5911`

**Test 试次级**
- Acc_paper：`0.6609`
- BalAcc_maj：`0.5564`
- Acc_majority：`0.6609`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5564` | F1：`0.7738` | Acc：`0.6609`

#### Fold 4

- stopped_epoch：`109` | best_epoch：`89`
- Val Acc_paper（早停）：`0.6433`
- Val BalAcc_maj（附报）：`0.5342`

**Test 试次级**
- Acc_paper：`0.6803`
- BalAcc_maj：`0.5725`
- Acc_majority：`0.6803`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5725` | F1：`0.7889` | Acc：`0.6803`

### Three
- Val Acc_paper：`0.4611 ± 0.0385`
- Test Acc_paper：`0.4734 ± 0.0245`
- Test BalAcc_maj：`0.4734 ± 0.0245`
- Test 窗级 BalAcc（附报）：`0.4734 ± 0.0245`

### Three 分折明细

#### Fold 0

- stopped_epoch：`105` | best_epoch：`85`
- Val Acc_paper（早停）：`0.4815`
- Val BalAcc_maj（附报）：`0.4815`

**Test 试次级**
- Acc_paper：`0.4570`
- BalAcc_maj：`0.4570`
- F1-macro（众数）：`0.4544`
- Rec idle/left/right：`0.4600` / `0.3664` / `0.5445`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4570` | F1m：`0.4544`

#### Fold 1

- stopped_epoch：`45` | best_epoch：`25`
- Val Acc_paper（早停）：`0.4811`
- Val BalAcc_maj（附报）：`0.4811`

**Test 试次级**
- Acc_paper：`0.5079`
- BalAcc_maj：`0.5079`
- F1-macro（众数）：`0.5005`
- Rec idle/left/right：`0.3364` / `0.5900` / `0.5973`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5079` | F1m：`0.5005`

#### Fold 2

- stopped_epoch：`75` | best_epoch：`55`
- Val Acc_paper（早停）：`0.4385`
- Val BalAcc_maj（附报）：`0.4385`

**Test 试次级**
- Acc_paper：`0.4464`
- BalAcc_maj：`0.4464`
- F1-macro（众数）：`0.4116`
- Rec idle/left/right：`0.1500` / `0.4845` / `0.7045`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4464` | F1m：`0.4116`

#### Fold 3

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.5067`
- Val BalAcc_maj（附报）：`0.5067`

**Test 试次级**
- Acc_paper：`0.4976`
- BalAcc_maj：`0.4976`
- F1-macro（众数）：`0.4888`
- Rec idle/left/right：`0.3264` / `0.6355` / `0.5309`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4976` | F1m：`0.4888`

#### Fold 4

- stopped_epoch：`74` | best_epoch：`54`
- Val Acc_paper（早停）：`0.3978`
- Val BalAcc_maj（附报）：`0.3978`

**Test 试次级**
- Acc_paper：`0.4580`
- BalAcc_maj：`0.4580`
- F1-macro（众数）：`0.4432`
- Rec idle/left/right：`0.2440` / `0.5260` / `0.6040`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4580` | F1m：`0.4432`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_fixed_cue2to4_noz",
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
  "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 0,
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

- 结束：`（见 run.log）`