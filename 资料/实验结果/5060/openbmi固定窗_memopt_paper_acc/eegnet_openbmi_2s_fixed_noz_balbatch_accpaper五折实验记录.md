# 被试独立五折实验记录（20260810_164412 / eegnet_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T16:44:12`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`eegnet` | EEGNet v4
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\eegnet_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_164412`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6450 ± 0.0236`
- Test Acc_paper：`0.6310 ± 0.0171`
- Test BalAcc_maj：`0.5630 ± 0.0436`
- Test 窗级 BalAcc（附报）：`0.5630 ± 0.0436`

### Task 分折明细

#### Fold 0

- stopped_epoch：`114` | best_epoch：`94`
- Val Acc_paper（早停）：`0.6752`
- Val BalAcc_maj（附报）：`0.5919`

**Test 试次级**
- Acc_paper：`0.6479`
- BalAcc_maj：`0.5920`
- Acc_majority：`0.6479`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5920` | F1：`0.7420` | Acc：`0.6479`

#### Fold 1

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6681`
- Val BalAcc_maj（附报）：`0.6019`

**Test 试次级**
- Acc_paper：`0.6509`
- BalAcc_maj：`0.5927`
- Acc_majority：`0.6509`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5927` | F1：`0.7456` | Acc：`0.6509`

#### Fold 2

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6167`
- Val BalAcc_maj（附报）：`0.5250`

**Test 试次级**
- Acc_paper：`0.6073`
- BalAcc_maj：`0.5125`
- Acc_majority：`0.6073`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5125` | F1：`0.7301` | Acc：`0.6073`

#### Fold 3

- stopped_epoch：`66` | best_epoch：`46`
- Val Acc_paper（早停）：`0.6433`
- Val BalAcc_maj（附报）：`0.6042`

**Test 试次级**
- Acc_paper：`0.6327`
- BalAcc_maj：`0.6098`
- Acc_majority：`0.6327`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6098` | F1：`0.7113` | Acc：`0.6327`

#### Fold 4

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6219`
- Val BalAcc_maj（附报）：`0.5258`

**Test 试次级**
- Acc_paper：`0.6163`
- BalAcc_maj：`0.5078`
- Acc_majority：`0.6163`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5078` | F1：`0.7434` | Acc：`0.6163`

### Three
- Val Acc_paper：`0.4760 ± 0.0435`
- Test Acc_paper：`0.4817 ± 0.0197`
- Test BalAcc_maj：`0.4817 ± 0.0197`
- Test 窗级 BalAcc（附报）：`0.4817 ± 0.0197`

### Three 分折明细

#### Fold 0

- stopped_epoch：`112` | best_epoch：`92`
- Val Acc_paper（早停）：`0.4874`
- Val BalAcc_maj（附报）：`0.4874`

**Test 试次级**
- Acc_paper：`0.4667`
- BalAcc_maj：`0.4667`
- F1-macro（众数）：`0.4571`
- Rec idle/left/right：`0.2964` / `0.4936` / `0.6100`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4667` | F1m：`0.4571`

#### Fold 1

- stopped_epoch：`66` | best_epoch：`46`
- Val Acc_paper（早停）：`0.5219`
- Val BalAcc_maj（附报）：`0.5219`

**Test 试次级**
- Acc_paper：`0.5039`
- BalAcc_maj：`0.5039`
- F1-macro（众数）：`0.4950`
- Rec idle/left/right：`0.3191` / `0.5909` / `0.6018`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5039` | F1m：`0.4950`

#### Fold 2

- stopped_epoch：`78` | best_epoch：`58`
- Val Acc_paper（早停）：`0.4496`
- Val BalAcc_maj（附报）：`0.4496`

**Test 试次级**
- Acc_paper：`0.4555`
- BalAcc_maj：`0.4555`
- F1-macro（众数）：`0.4406`
- Rec idle/left/right：`0.2491` / `0.4845` / `0.6327`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4555` | F1m：`0.4406`

#### Fold 3

- stopped_epoch：`92` | best_epoch：`72`
- Val Acc_paper（早停）：`0.5156`
- Val BalAcc_maj（附报）：`0.5156`

**Test 试次级**
- Acc_paper：`0.5045`
- BalAcc_maj：`0.5045`
- F1-macro（众数）：`0.4973`
- Rec idle/left/right：`0.3973` / `0.6855` / `0.4309`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5045` | F1m：`0.4973`

#### Fold 4

- stopped_epoch：`73` | best_epoch：`53`
- Val Acc_paper（早停）：`0.4056`
- Val BalAcc_maj（附报）：`0.4056`

**Test 试次级**
- Acc_paper：`0.4777`
- BalAcc_maj：`0.4777`
- F1-macro（众数）：`0.4679`
- Rec idle/left/right：`0.2980` / `0.5980` / `0.5370`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4777` | F1m：`0.4679`

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