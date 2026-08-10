# 被试独立五折实验记录（20260810_163132 / conformer_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T16:31:32`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`conformer` | EEGConformer num_layers=2, num_heads=10, att_drop=0.5
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\conformer_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_163132`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6592 ± 0.0312`
- Test Acc_paper：`0.6636 ± 0.0145`
- Test BalAcc_maj：`0.5811 ± 0.0240`
- Test 窗级 BalAcc（附报）：`0.5811 ± 0.0240`

### Task 分折明细

#### Fold 0

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.6756`
- Val BalAcc_maj（附报）：`0.5664`

**Test 试次级**
- Acc_paper：`0.6494`
- BalAcc_maj：`0.5677`
- Acc_majority：`0.6494`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5677` | F1：`0.7555` | Acc：`0.6494`

#### Fold 1

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.6881`
- Val BalAcc_maj（附报）：`0.5933`

**Test 试次级**
- Acc_paper：`0.6915`
- BalAcc_maj：`0.6241`
- Acc_majority：`0.6915`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6241` | F1：`0.7813` | Acc：`0.6915`

#### Fold 2

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.6185`
- Val BalAcc_maj（附报）：`0.5567`

**Test 试次级**
- Acc_paper：`0.6582`
- BalAcc_maj：`0.5582`
- Acc_majority：`0.6582`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5582` | F1：`0.7700` | Acc：`0.6582`

#### Fold 3

- stopped_epoch：`64` | best_epoch：`44`
- Val Acc_paper（早停）：`0.6893`
- Val BalAcc_maj（附报）：`0.6142`

**Test 试次级**
- Acc_paper：`0.6609`
- BalAcc_maj：`0.5902`
- Acc_majority：`0.6609`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5902` | F1：`0.7593` | Acc：`0.6609`

#### Fold 4

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.6244`
- Val BalAcc_maj（附报）：`0.5206`

**Test 试次级**
- Acc_paper：`0.6580`
- BalAcc_maj：`0.5655`
- Acc_majority：`0.6580`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5655` | F1：`0.7667` | Acc：`0.6580`

### Three
- Val Acc_paper：`0.4702 ± 0.0457`
- Test Acc_paper：`0.4837 ± 0.0257`
- Test BalAcc_maj：`0.4837 ± 0.0257`
- Test 窗级 BalAcc（附报）：`0.4837 ± 0.0257`

### Three 分折明细

#### Fold 0

- stopped_epoch：`86` | best_epoch：`66`
- Val Acc_paper（早停）：`0.4833`
- Val BalAcc_maj（附报）：`0.4833`

**Test 试次级**
- Acc_paper：`0.4718`
- BalAcc_maj：`0.4718`
- F1-macro（众数）：`0.4695`
- Rec idle/left/right：`0.4355` / `0.4082` / `0.5718`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4718` | F1m：`0.4695`

#### Fold 1

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.4881`
- Val BalAcc_maj（附报）：`0.4881`

**Test 试次级**
- Acc_paper：`0.4942`
- BalAcc_maj：`0.4942`
- F1-macro（众数）：`0.4861`
- Rec idle/left/right：`0.3609` / `0.4373` / `0.6845`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4942` | F1m：`0.4861`

#### Fold 2

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.4544`
- Val BalAcc_maj（附报）：`0.4544`

**Test 试次级**
- Acc_paper：`0.4536`
- BalAcc_maj：`0.4536`
- F1-macro（众数）：`0.4347`
- Rec idle/left/right：`0.2418` / `0.4127` / `0.7064`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4536` | F1m：`0.4347`

#### Fold 3

- stopped_epoch：`45` | best_epoch：`25`
- Val Acc_paper（早停）：`0.5319`
- Val BalAcc_maj（附报）：`0.5319`

**Test 试次级**
- Acc_paper：`0.5282`
- BalAcc_maj：`0.5282`
- F1-macro（众数）：`0.5241`
- Rec idle/left/right：`0.4345` / `0.6609` / `0.4891`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5282` | F1m：`0.5241`

#### Fold 4

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.3933`
- Val BalAcc_maj（附报）：`0.3933`

**Test 试次级**
- Acc_paper：`0.4707`
- BalAcc_maj：`0.4707`
- F1-macro（众数）：`0.4549`
- Rec idle/left/right：`0.2470` / `0.5510` / `0.6140`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4707` | F1m：`0.4549`

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