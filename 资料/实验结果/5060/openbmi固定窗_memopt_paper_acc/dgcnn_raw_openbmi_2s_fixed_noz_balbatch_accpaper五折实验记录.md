# 被试独立五折实验记录（20260810_173725 / dgcnn_raw_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T17:37:25`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dgcnn_raw` | DGCNN（raw 版）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\dgcnn_raw_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_173725`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6714 ± 0.0277`
- Test Acc_paper：`0.6655 ± 0.0090`
- Test BalAcc_maj：`0.5425 ± 0.0215`
- Test 窗级 BalAcc（附报）：`0.5425 ± 0.0215`

### Task 分折明细

#### Fold 0

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.6919`
- Val BalAcc_maj（附报）：`0.5644`

**Test 试次级**
- Acc_paper：`0.6536`
- BalAcc_maj：`0.5336`
- Acc_majority：`0.6536`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5336` | F1：`0.7748` | Acc：`0.6536`

#### Fold 1

- stopped_epoch：`106` | best_epoch：`86`
- Val Acc_paper（早停）：`0.6856`
- Val BalAcc_maj（附报）：`0.5647`

**Test 试次级**
- Acc_paper：`0.6694`
- BalAcc_maj：`0.5748`
- Acc_majority：`0.6694`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5748` | F1：`0.7759` | Acc：`0.6694`

#### Fold 2

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.6470`
- Val BalAcc_maj（附报）：`0.5292`

**Test 试次级**
- Acc_paper：`0.6582`
- BalAcc_maj：`0.5141`
- Acc_majority：`0.6582`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5141` | F1：`0.7868` | Acc：`0.6582`

#### Fold 3

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.7022`
- Val BalAcc_maj（附报）：`0.5756`

**Test 试次级**
- Acc_paper：`0.6670`
- BalAcc_maj：`0.5316`
- Acc_majority：`0.6670`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5316` | F1：`0.7897` | Acc：`0.6670`

#### Fold 4

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.6304`
- Val BalAcc_maj（附报）：`0.5039`

**Test 试次级**
- Acc_paper：`0.6793`
- BalAcc_maj：`0.5585`
- Acc_majority：`0.6793`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5585` | F1：`0.7929` | Acc：`0.6793`

### Three
- Val Acc_paper：`0.4584 ± 0.0331`
- Test Acc_paper：`0.4630 ± 0.0256`
- Test BalAcc_maj：`0.4630 ± 0.0256`
- Test 窗级 BalAcc（附报）：`0.4630 ± 0.0256`

### Three 分折明细

#### Fold 0

- stopped_epoch：`69` | best_epoch：`49`
- Val Acc_paper（早停）：`0.4767`
- Val BalAcc_maj（附报）：`0.4767`

**Test 试次级**
- Acc_paper：`0.4364`
- BalAcc_maj：`0.4364`
- F1-macro（众数）：`0.4253`
- Rec idle/left/right：`0.2691` / `0.4345` / `0.6055`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4364` | F1m：`0.4253`

#### Fold 1

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.4837`
- Val BalAcc_maj（附报）：`0.4837`

**Test 试次级**
- Acc_paper：`0.4979`
- BalAcc_maj：`0.4979`
- F1-macro（众数）：`0.4871`
- Rec idle/left/right：`0.3064` / `0.5336` / `0.6536`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4979` | F1m：`0.4871`

#### Fold 2

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.4367`
- Val BalAcc_maj（附报）：`0.4367`

**Test 试次级**
- Acc_paper：`0.4391`
- BalAcc_maj：`0.4391`
- F1-macro（众数）：`0.3990`
- Rec idle/left/right：`0.1173` / `0.5700` / `0.6300`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4391` | F1m：`0.3990`

#### Fold 3

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.4911`
- Val BalAcc_maj（附报）：`0.4911`

**Test 试次级**
- Acc_paper：`0.4891`
- BalAcc_maj：`0.4891`
- F1-macro（众数）：`0.4794`
- Rec idle/left/right：`0.3100` / `0.6309` / `0.5264`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4891` | F1m：`0.4794`

#### Fold 4

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.4041`
- Val BalAcc_maj（附报）：`0.4041`

**Test 试次级**
- Acc_paper：`0.4527`
- BalAcc_maj：`0.4527`
- F1-macro（众数）：`0.4374`
- Rec idle/left/right：`0.2420` / `0.4860` / `0.6300`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4527` | F1m：`0.4374`

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