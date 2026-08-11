# 被试独立五折实验记录（20260810_171222 / dgcnn_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T17:12:22`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dgcnn` | DGCNN（z-score 版）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\dgcnn_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_171222`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6212 ± 0.0475`
- Test Acc_paper：`0.6380 ± 0.0160`
- Test BalAcc_maj：`0.5899 ± 0.0173`
- Test 窗级 BalAcc（附报）：`0.5899 ± 0.0173`

### Task 分折明细

#### Fold 0

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.6470`
- Val BalAcc_maj（附报）：`0.5892`

**Test 试次级**
- Acc_paper：`0.6209`
- BalAcc_maj：`0.5730`
- Acc_majority：`0.6209`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5730` | F1：`0.7160` | Acc：`0.6209`

#### Fold 1

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.6441`
- Val BalAcc_maj（附报）：`0.5922`

**Test 试次级**
- Acc_paper：`0.6600`
- BalAcc_maj：`0.6232`
- Acc_majority：`0.6600`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6232` | F1：`0.7421` | Acc：`0.6600`

#### Fold 2

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.5426`
- Val BalAcc_maj（附报）：`0.5542`

**Test 试次级**
- Acc_paper：`0.6182`
- BalAcc_maj：`0.5866`
- Acc_majority：`0.6182`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5866` | F1：`0.7041` | Acc：`0.6182`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.6778`
- Val BalAcc_maj（附报）：`0.6103`

**Test 试次级**
- Acc_paper：`0.6448`
- BalAcc_maj：`0.5832`
- Acc_majority：`0.6448`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5832` | F1：`0.7425` | Acc：`0.6448`

#### Fold 4

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5944`
- Val BalAcc_maj（附报）：`0.5131`

**Test 试次级**
- Acc_paper：`0.6460`
- BalAcc_maj：`0.5837`
- Acc_majority：`0.6460`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5837` | F1：`0.7437` | Acc：`0.6460`

### Three
- Val Acc_paper：`0.4517 ± 0.0379`
- Test Acc_paper：`0.4545 ± 0.0244`
- Test BalAcc_maj：`0.4545 ± 0.0244`
- Test 窗级 BalAcc（附报）：`0.4545 ± 0.0244`

### Three 分折明细

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4559`
- Val BalAcc_maj（附报）：`0.4559`

**Test 试次级**
- Acc_paper：`0.4236`
- BalAcc_maj：`0.4236`
- F1-macro（众数）：`0.4237`
- Rec idle/left/right：`0.4300` / `0.4118` / `0.4291`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4236` | F1m：`0.4237`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.4796`
- Val BalAcc_maj（附报）：`0.4796`

**Test 试次级**
- Acc_paper：`0.4912`
- BalAcc_maj：`0.4912`
- F1-macro（众数）：`0.4903`
- Rec idle/left/right：`0.4309` / `0.4882` / `0.5545`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4912` | F1m：`0.4903`

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.4204`
- Val BalAcc_maj（附报）：`0.4204`

**Test 试次级**
- Acc_paper：`0.4330`
- BalAcc_maj：`0.4330`
- F1-macro（众数）：`0.4295`
- Rec idle/left/right：`0.3236` / `0.4836` / `0.4918`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4330` | F1m：`0.4295`

#### Fold 3

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.5033`
- Val BalAcc_maj（附报）：`0.5033`

**Test 试次级**
- Acc_paper：`0.4694`
- BalAcc_maj：`0.4694`
- F1-macro（众数）：`0.4645`
- Rec idle/left/right：`0.4927` / `0.5709` / `0.3445`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4694` | F1m：`0.4645`

#### Fold 4

- stopped_epoch：`93` | best_epoch：`73`
- Val Acc_paper（早停）：`0.3993`
- Val BalAcc_maj（附报）：`0.3993`

**Test 试次级**
- Acc_paper：`0.4550`
- BalAcc_maj：`0.4550`
- F1-macro（众数）：`0.4456`
- Rec idle/left/right：`0.2800` / `0.5090` / `0.5760`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4550` | F1m：`0.4456`

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