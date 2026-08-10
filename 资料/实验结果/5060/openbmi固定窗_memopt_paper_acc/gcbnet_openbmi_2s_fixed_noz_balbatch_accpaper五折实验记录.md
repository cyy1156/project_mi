# 被试独立五折实验记录（20260810_170810 / gcbnet_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T17:08:10`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`gcbnet` | GCBNet（z-score 版）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\gcbnet_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_170810`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6018 ± 0.0386`
- Test Acc_paper：`0.6159 ± 0.0224`
- Test BalAcc_maj：`0.5884 ± 0.0169`
- Test 窗级 BalAcc（附报）：`0.5884 ± 0.0169`

### Task 分折明细

#### Fold 0

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.6044`
- Val BalAcc_maj（附报）：`0.5811`

**Test 试次级**
- Acc_paper：`0.5770`
- BalAcc_maj：`0.5782`
- Acc_majority：`0.5770`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5782` | F1：`0.6442` | Acc：`0.5770`

#### Fold 1

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.6474`
- Val BalAcc_maj（附报）：`0.6061`

**Test 试次级**
- Acc_paper：`0.6245`
- BalAcc_maj：`0.6095`
- Acc_majority：`0.6245`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6095` | F1：`0.6992` | Acc：`0.6245`

#### Fold 2

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.5352`
- Val BalAcc_maj（附报）：`0.5500`

**Test 试次级**
- Acc_paper：`0.6145`
- BalAcc_maj：`0.5616`
- Acc_majority：`0.6145`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5616` | F1：`0.7136` | Acc：`0.6145`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.6304`
- Val BalAcc_maj（附报）：`0.6072`

**Test 试次级**
- Acc_paper：`0.6176`
- BalAcc_maj：`0.6002`
- Acc_majority：`0.6176`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6002` | F1：`0.6946` | Acc：`0.6176`

#### Fold 4

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5915`
- Val BalAcc_maj（附报）：`0.5122`

**Test 试次级**
- Acc_paper：`0.6460`
- BalAcc_maj：`0.5922`
- Acc_majority：`0.6460`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5922` | F1：`0.7395` | Acc：`0.6460`

### Three
- Val Acc_paper：`0.4525 ± 0.0371`
- Test Acc_paper：`0.4551 ± 0.0208`
- Test BalAcc_maj：`0.4551 ± 0.0208`
- Test 窗级 BalAcc（附报）：`0.4551 ± 0.0208`

### Three 分折明细

#### Fold 0

- stopped_epoch：`77` | best_epoch：`57`
- Val Acc_paper（早停）：`0.4670`
- Val BalAcc_maj（附报）：`0.4670`

**Test 试次级**
- Acc_paper：`0.4270`
- BalAcc_maj：`0.4270`
- F1-macro（众数）：`0.4265`
- Rec idle/left/right：`0.3927` / `0.4627` / `0.4255`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4270` | F1m：`0.4265`

#### Fold 1

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.4711`
- Val BalAcc_maj（附报）：`0.4711`

**Test 试次级**
- Acc_paper：`0.4867`
- BalAcc_maj：`0.4867`
- F1-macro（众数）：`0.4858`
- Rec idle/left/right：`0.4282` / `0.4909` / `0.5409`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4867` | F1m：`0.4858`

#### Fold 2

- stopped_epoch：`66` | best_epoch：`46`
- Val Acc_paper（早停）：`0.4315`
- Val BalAcc_maj（附报）：`0.4315`

**Test 试次级**
- Acc_paper：`0.4385`
- BalAcc_maj：`0.4385`
- F1-macro（众数）：`0.4271`
- Rec idle/left/right：`0.2518` / `0.4964` / `0.5673`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4385` | F1m：`0.4271`

#### Fold 3

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.5004`
- Val BalAcc_maj（附报）：`0.5004`

**Test 试次级**
- Acc_paper：`0.4585`
- BalAcc_maj：`0.4585`
- F1-macro（众数）：`0.4569`
- Rec idle/left/right：`0.5073` / `0.4827` / `0.3855`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4585` | F1m：`0.4569`

#### Fold 4

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.3926`
- Val BalAcc_maj（附报）：`0.3926`

**Test 试次级**
- Acc_paper：`0.4647`
- BalAcc_maj：`0.4647`
- F1-macro（众数）：`0.4602`
- Rec idle/left/right：`0.3440` / `0.4800` / `0.5700`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4647` | F1m：`0.4602`

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