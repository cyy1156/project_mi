# 被试独立五折实验记录（20260810_165209 / eegtcnet_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T16:52:09`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`eegtcnet` | EEGTCNet
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\eegtcnet_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_165209`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6658 ± 0.0234`
- Test Acc_paper：`0.6597 ± 0.0063`
- Test BalAcc_maj：`0.5536 ± 0.0417`
- Test 窗级 BalAcc（附报）：`0.5536 ± 0.0417`

### Task 分折明细

#### Fold 0

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.6796`
- Val BalAcc_maj（附报）：`0.5594`

**Test 试次级**
- Acc_paper：`0.6582`
- BalAcc_maj：`0.5520`
- Acc_majority：`0.6582`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5520` | F1：`0.7725` | Acc：`0.6582`

#### Fold 1

- stopped_epoch：`134` | best_epoch：`114`
- Val Acc_paper（早停）：`0.6711`
- Val BalAcc_maj（附报）：`0.6247`

**Test 试次级**
- Acc_paper：`0.6491`
- BalAcc_maj：`0.6250`
- Acc_majority：`0.6491`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6250` | F1：`0.7260` | Acc：`0.6491`

#### Fold 2

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.6244`
- Val BalAcc_maj（附报）：`0.5361`

**Test 试次级**
- Acc_paper：`0.6679`
- BalAcc_maj：`0.5652`
- Acc_majority：`0.6679`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5652` | F1：`0.7780` | Acc：`0.6679`

#### Fold 3

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.6937`
- Val BalAcc_maj（附报）：`0.5711`

**Test 试次级**
- Acc_paper：`0.6594`
- BalAcc_maj：`0.5214`
- Acc_majority：`0.6594`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5214` | F1：`0.7855` | Acc：`0.6594`

#### Fold 4

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6600`
- Val BalAcc_maj（附报）：`0.4989`

**Test 试次级**
- Acc_paper：`0.6640`
- BalAcc_maj：`0.5042`
- Acc_majority：`0.6640`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5042` | F1：`0.7960` | Acc：`0.6640`

### Three
- Val Acc_paper：`0.4244 ± 0.0654`
- Test Acc_paper：`0.4257 ± 0.0649`
- Test BalAcc_maj：`0.4257 ± 0.0649`
- Test 窗级 BalAcc（附报）：`0.4257 ± 0.0649`

### Three 分折明细

#### Fold 0

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.3604`
- Val BalAcc_maj（附报）：`0.3604`

**Test 试次级**
- Acc_paper：`0.3567`
- BalAcc_maj：`0.3567`
- F1-macro（众数）：`0.3298`
- Rec idle/left/right：`0.6436` / `0.2500` / `0.1764`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3567` | F1m：`0.3298`

#### Fold 1

- stopped_epoch：`166` | best_epoch：`146`
- Val Acc_paper（早停）：`0.4933`
- Val BalAcc_maj（附报）：`0.4933`

**Test 试次级**
- Acc_paper：`0.4927`
- BalAcc_maj：`0.4927`
- F1-macro（众数）：`0.4862`
- Rec idle/left/right：`0.4664` / `0.3591` / `0.6527`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4927` | F1m：`0.4862`

#### Fold 2

- stopped_epoch：`175` | best_epoch：`155`
- Val Acc_paper（早停）：`0.4470`
- Val BalAcc_maj（附报）：`0.4470`

**Test 试次级**
- Acc_paper：`0.4497`
- BalAcc_maj：`0.4497`
- F1-macro（众数）：`0.4312`
- Rec idle/left/right：`0.2236` / `0.4736` / `0.6518`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4497` | F1m：`0.4312`

#### Fold 3

- stopped_epoch：`259` | best_epoch：`239`
- Val Acc_paper（早停）：`0.4870`
- Val BalAcc_maj（附报）：`0.4870`

**Test 试次级**
- Acc_paper：`0.4888`
- BalAcc_maj：`0.4888`
- F1-macro（众数）：`0.4724`
- Rec idle/left/right：`0.4382` / `0.7345` / `0.2936`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4888` | F1m：`0.4724`

#### Fold 4

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.3344`
- Val BalAcc_maj（附报）：`0.3344`

**Test 试次级**
- Acc_paper：`0.3407`
- BalAcc_maj：`0.3407`
- F1-macro（众数）：`0.2318`
- Rec idle/left/right：`0.0220` / `0.8930` / `0.1070`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.3407` | F1m：`0.2318`

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