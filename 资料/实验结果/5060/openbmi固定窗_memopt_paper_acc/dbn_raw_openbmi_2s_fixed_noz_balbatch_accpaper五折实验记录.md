# 被试独立五折实验记录（20260810_171927 / dbn_raw_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T17:19:27`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dbn_raw` | DBN（raw 版）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\dbn_raw_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_171927`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6760 ± 0.0210`
- Test Acc_paper：`0.6764 ± 0.0149`
- Test BalAcc_maj：`0.5437 ± 0.0226`
- Test 窗级 BalAcc（附报）：`0.5437 ± 0.0226`

### Task 分折明细

#### Fold 0

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6956`
- Val BalAcc_maj（附报）：`0.5533`

**Test 试次级**
- Acc_paper：`0.6579`
- BalAcc_maj：`0.5348`
- Acc_majority：`0.6579`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5348` | F1：`0.7789` | Acc：`0.6579`

#### Fold 1

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.6967`
- Val BalAcc_maj（附报）：`0.5642`

**Test 试次级**
- Acc_paper：`0.6773`
- BalAcc_maj：`0.5673`
- Acc_majority：`0.6773`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5673` | F1：`0.7876` | Acc：`0.6773`

#### Fold 2

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.6448`
- Val BalAcc_maj（附报）：`0.5225`

**Test 试次级**
- Acc_paper：`0.6624`
- BalAcc_maj：`0.5123`
- Acc_majority：`0.6624`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5123` | F1：`0.7918` | Acc：`0.6624`

#### Fold 3

- stopped_epoch：`87` | best_epoch：`67`
- Val Acc_paper（早停）：`0.6852`
- Val BalAcc_maj（附报）：`0.5739`

**Test 试次级**
- Acc_paper：`0.6982`
- BalAcc_maj：`0.5718`
- Acc_majority：`0.6982`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5718` | F1：`0.8077` | Acc：`0.6982`

#### Fold 4

- stopped_epoch：`159` | best_epoch：`139`
- Val Acc_paper（早停）：`0.6578`
- Val BalAcc_maj（附报）：`0.4969`

**Test 试次级**
- Acc_paper：`0.6863`
- BalAcc_maj：`0.5323`
- Acc_majority：`0.6863`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5323` | F1：`0.8087` | Acc：`0.6863`

### Three
- Val Acc_paper：`0.4653 ± 0.0342`
- Test Acc_paper：`0.4721 ± 0.0144`
- Test BalAcc_maj：`0.4721 ± 0.0144`
- Test 窗级 BalAcc（附报）：`0.4721 ± 0.0144`

### Three 分折明细

#### Fold 0

- stopped_epoch：`72` | best_epoch：`52`
- Val Acc_paper（早停）：`0.4837`
- Val BalAcc_maj（附报）：`0.4837`

**Test 试次级**
- Acc_paper：`0.4627`
- BalAcc_maj：`0.4627`
- F1-macro（众数）：`0.4600`
- Rec idle/left/right：`0.3664` / `0.5255` / `0.4964`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4627` | F1m：`0.4600`

#### Fold 1

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.4933`
- Val BalAcc_maj（附报）：`0.4933`

**Test 试次级**
- Acc_paper：`0.4936`
- BalAcc_maj：`0.4936`
- F1-macro（众数）：`0.4878`
- Rec idle/left/right：`0.3409` / `0.5855` / `0.5545`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4936` | F1m：`0.4878`

#### Fold 2

- stopped_epoch：`65` | best_epoch：`45`
- Val Acc_paper（早停）：`0.4559`
- Val BalAcc_maj（附报）：`0.4559`

**Test 试次级**
- Acc_paper：`0.4512`
- BalAcc_maj：`0.4512`
- F1-macro（众数）：`0.4331`
- Rec idle/left/right：`0.2191` / `0.5473` / `0.5873`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4512` | F1m：`0.4331`

#### Fold 3

- stopped_epoch：`45` | best_epoch：`25`
- Val Acc_paper（早停）：`0.4911`
- Val BalAcc_maj（附报）：`0.4911`

**Test 试次级**
- Acc_paper：`0.4794`
- BalAcc_maj：`0.4794`
- F1-macro（众数）：`0.4523`
- Rec idle/left/right：`0.2118` / `0.7482` / `0.4782`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4794` | F1m：`0.4523`

#### Fold 4

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.4022`
- Val BalAcc_maj（附报）：`0.4022`

**Test 试次级**
- Acc_paper：`0.4737`
- BalAcc_maj：`0.4737`
- F1-macro（众数）：`0.4430`
- Rec idle/left/right：`0.1740` / `0.6050` / `0.6420`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4737` | F1m：`0.4430`

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