# 被试独立五折实验记录（20260810_160439 / shallow_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T16:04:39`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\shallow_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_160439`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6228 ± 0.0344`
- Test Acc_paper：`0.6361 ± 0.0163`
- Test BalAcc_maj：`0.5892 ± 0.0234`
- Test 窗级 BalAcc（附报）：`0.5892 ± 0.0234`

### Task 分折明细

#### Fold 0

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6293`
- Val BalAcc_maj（附报）：`0.5744`

**Test 试次级**
- Acc_paper：`0.6127`
- BalAcc_maj：`0.5727`
- Acc_majority：`0.6127`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5727` | F1：`0.7046` | Acc：`0.6127`

#### Fold 1

- stopped_epoch：`50` | best_epoch：`30`
- Val Acc_paper（早停）：`0.6689`
- Val BalAcc_maj（附报）：`0.5961`

**Test 试次级**
- Acc_paper：`0.6594`
- BalAcc_maj：`0.6127`
- Acc_majority：`0.6594`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6127` | F1：`0.7466` | Acc：`0.6594`

#### Fold 2

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.5685`
- Val BalAcc_maj（附报）：`0.5553`

**Test 试次级**
- Acc_paper：`0.6294`
- BalAcc_maj：`0.5711`
- Acc_majority：`0.6294`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5711` | F1：`0.7285` | Acc：`0.6294`

#### Fold 3

- stopped_epoch：`81` | best_epoch：`61`
- Val Acc_paper（早停）：`0.6437`
- Val BalAcc_maj（附报）：`0.6200`

**Test 试次级**
- Acc_paper：`0.6488`
- BalAcc_maj：`0.6223`
- Acc_majority：`0.6488`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6223` | F1：`0.7271` | Acc：`0.6488`

#### Fold 4

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.6037`
- Val BalAcc_maj（附报）：`0.5208`

**Test 试次级**
- Acc_paper：`0.6300`
- BalAcc_maj：`0.5673`
- Acc_majority：`0.6300`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5673` | F1：`0.7314` | Acc：`0.6300`

### Three
- Val Acc_paper：`0.4761 ± 0.0467`
- Test Acc_paper：`0.4874 ± 0.0256`
- Test BalAcc_maj：`0.4874 ± 0.0256`
- Test 窗级 BalAcc（附报）：`0.4874 ± 0.0256`

### Three 分折明细

#### Fold 0

- stopped_epoch：`157` | best_epoch：`137`
- Val Acc_paper（早停）：`0.4989`
- Val BalAcc_maj（附报）：`0.4989`

**Test 试次级**
- Acc_paper：`0.4767`
- BalAcc_maj：`0.4767`
- F1-macro（众数）：`0.4742`
- Rec idle/left/right：`0.5155` / `0.3809` / `0.5336`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4767` | F1m：`0.4742`

#### Fold 1

- stopped_epoch：`95` | best_epoch：`75`
- Val Acc_paper（早停）：`0.5107`
- Val BalAcc_maj（附报）：`0.5107`

**Test 试次级**
- Acc_paper：`0.5130`
- BalAcc_maj：`0.5130`
- F1-macro（众数）：`0.5127`
- Rec idle/left/right：`0.5109` / `0.4627` / `0.5655`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5130` | F1m：`0.5127`

#### Fold 2

- stopped_epoch：`77` | best_epoch：`57`
- Val Acc_paper（早停）：`0.4456`
- Val BalAcc_maj（附报）：`0.4456`

**Test 试次级**
- Acc_paper：`0.4594`
- BalAcc_maj：`0.4594`
- F1-macro（众数）：`0.4572`
- Rec idle/left/right：`0.3755` / `0.4664` / `0.5364`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4594` | F1m：`0.4572`

#### Fold 3

- stopped_epoch：`64` | best_epoch：`44`
- Val Acc_paper（早停）：`0.5256`
- Val BalAcc_maj（附报）：`0.5256`

**Test 试次级**
- Acc_paper：`0.5224`
- BalAcc_maj：`0.5224`
- F1-macro（众数）：`0.5202`
- Rec idle/left/right：`0.4627` / `0.6200` / `0.4845`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5224` | F1m：`0.5202`

#### Fold 4

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.4000`
- Val BalAcc_maj（附报）：`0.4000`

**Test 试次级**
- Acc_paper：`0.4653`
- BalAcc_maj：`0.4653`
- F1-macro（众数）：`0.4614`
- Rec idle/left/right：`0.3600` / `0.4570` / `0.5790`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4653` | F1m：`0.4614`

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