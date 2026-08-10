# 被试独立五折实验记录（20260810_161228 / deep_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T16:12:28`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`deep` | Deep4Net（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\deep_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_161228`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6310 ± 0.0266`
- Test Acc_paper：`0.6445 ± 0.0256`
- Test BalAcc_maj：`0.5556 ± 0.0223`
- Test 窗级 BalAcc（附报）：`0.5556 ± 0.0223`

### Task 分折明细

#### Fold 0

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5993`
- Val BalAcc_maj（附报）：`0.5425`

**Test 试次级**
- Acc_paper：`0.6027`
- BalAcc_maj：`0.5448`
- Acc_majority：`0.6027`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5448` | F1：`0.7069` | Acc：`0.6027`

#### Fold 1

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.6604`
- Val BalAcc_maj（附报）：`0.5567`

**Test 试次级**
- Acc_paper：`0.6803`
- BalAcc_maj：`0.5900`
- Acc_majority：`0.6803`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5900` | F1：`0.7822` | Acc：`0.6803`

#### Fold 2

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6078`
- Val BalAcc_maj（附报）：`0.5294`

**Test 试次级**
- Acc_paper：`0.6391`
- BalAcc_maj：`0.5350`
- Acc_majority：`0.6391`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5350` | F1：`0.7579` | Acc：`0.6391`

#### Fold 3

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.6637`
- Val BalAcc_maj（附报）：`0.5881`

**Test 试次级**
- Acc_paper：`0.6415`
- BalAcc_maj：`0.5736`
- Acc_majority：`0.6415`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5736` | F1：`0.7430` | Acc：`0.6415`

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6237`
- Val BalAcc_maj（附报）：`0.4942`

**Test 试次级**
- Acc_paper：`0.6587`
- BalAcc_maj：`0.5347`
- Acc_majority：`0.6587`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5347` | F1：`0.7798` | Acc：`0.6587`

### Three
- Val Acc_paper：`0.4257 ± 0.0236`
- Test Acc_paper：`0.4334 ± 0.0191`
- Test BalAcc_maj：`0.4334 ± 0.0191`
- Test 窗级 BalAcc（附报）：`0.4334 ± 0.0191`

### Three 分折明细

#### Fold 0

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.4344`
- Val BalAcc_maj（附报）：`0.4344`

**Test 试次级**
- Acc_paper：`0.4100`
- BalAcc_maj：`0.4100`
- F1-macro（众数）：`0.3996`
- Rec idle/left/right：`0.3873` / `0.2618` / `0.5809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4100` | F1m：`0.3996`

#### Fold 1

- stopped_epoch：`81` | best_epoch：`61`
- Val Acc_paper（早停）：`0.4489`
- Val BalAcc_maj（附报）：`0.4489`

**Test 试次级**
- Acc_paper：`0.4552`
- BalAcc_maj：`0.4552`
- F1-macro（众数）：`0.4513`
- Rec idle/left/right：`0.3745` / `0.4018` / `0.5891`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4552` | F1m：`0.4513`

#### Fold 2

- stopped_epoch：`54` | best_epoch：`34`
- Val Acc_paper（早停）：`0.4122`
- Val BalAcc_maj（附报）：`0.4122`

**Test 试次级**
- Acc_paper：`0.4233`
- BalAcc_maj：`0.4233`
- F1-macro（众数）：`0.4011`
- Rec idle/left/right：`0.1918` / `0.4273` / `0.6509`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4233` | F1m：`0.4011`

#### Fold 3

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.4467`
- Val BalAcc_maj（附报）：`0.4467`

**Test 试次级**
- Acc_paper：`0.4215`
- BalAcc_maj：`0.4215`
- F1-macro（众数）：`0.4150`
- Rec idle/left/right：`0.4209` / `0.5491` / `0.2945`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4215` | F1m：`0.4150`

#### Fold 4

- stopped_epoch：`75` | best_epoch：`55`
- Val Acc_paper（早停）：`0.3863`
- Val BalAcc_maj（附报）：`0.3863`

**Test 试次级**
- Acc_paper：`0.4570`
- BalAcc_maj：`0.4570`
- F1-macro（众数）：`0.4506`
- Rec idle/left/right：`0.3210` / `0.4630` / `0.5870`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4570` | F1m：`0.4506`

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