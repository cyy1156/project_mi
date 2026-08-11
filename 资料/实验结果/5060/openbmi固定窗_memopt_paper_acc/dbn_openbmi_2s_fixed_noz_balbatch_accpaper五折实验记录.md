# 被试独立五折实验记录（20260810_171528 / dbn_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-10T17:15:28`
- device：`cuda`（NVIDIA RTX 5060 Laptop）
- data：`openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / 固定窗 cue2to4 / 无 z-score**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s fixed cue2-4s nozscore openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dbn` | DBN（z-score 版）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper_memopt\dbn_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260810_171528`
- shared hp：`{"data_tag": "openbmi_2s_fixed_cue2to4_noz", "n_folds": 5, "val_ratio": 0.2, "seed": 42, "max_epochs": 300, "patience": 20, "batch_train": 128, "batch_eval": 256, "lr": 0.0001, "weight_decay": 0.0001, "drop_prob": 0.5, "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train", "early_stop": "acc_paper", "train_sampler": "balanced_invfreq", "n_times_expected": 500, "no_rap": true, "no_balbatch": false, "openbmi_only": true, "num_workers": 0, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 2, "non_blocking": true, "torch_num_threads": 6, "cudnn_benchmark": true, "deterministic": false, "use_amp": true}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6606 ± 0.0355`
- Test Acc_paper：`0.6607 ± 0.0301`
- Test BalAcc_maj：`0.5290 ± 0.0267`
- Test 窗级 BalAcc（附报）：`0.5290 ± 0.0267`

### Task 分折明细

#### Fold 0

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6681`
- Val BalAcc_maj（附报）：`0.5083`

**Test 试次级**
- Acc_paper：`0.6764`
- BalAcc_maj：`0.5236`
- Acc_majority：`0.6764`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5236` | F1：`0.8018` | Acc：`0.6764`

#### Fold 1

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6670`
- Val BalAcc_maj（附报）：`0.5008`

**Test 试次级**
- Acc_paper：`0.6667`
- BalAcc_maj：`0.5000`
- Acc_majority：`0.6667`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5000` | F1：`0.8000` | Acc：`0.6667`

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5963`
- Val BalAcc_maj（附报）：`0.5211`

**Test 试次级**
- Acc_paper：`0.6027`
- BalAcc_maj：`0.5516`
- Acc_majority：`0.6027`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5516` | F1：`0.7029` | Acc：`0.6027`

#### Fold 3

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.7059`
- Val BalAcc_maj（附报）：`0.5842`

**Test 试次级**
- Acc_paper：`0.6897`
- BalAcc_maj：`0.5673`
- Acc_majority：`0.6897`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5673` | F1：`0.8006` | Acc：`0.6897`

#### Fold 4

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6656`
- Val BalAcc_maj（附报）：`0.4994`

**Test 试次级**
- Acc_paper：`0.6680`
- BalAcc_maj：`0.5022`
- Acc_majority：`0.6680`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5022` | F1：`0.8006` | Acc：`0.6680`

### Three
- Val Acc_paper：`0.4485 ± 0.0402`
- Test Acc_paper：`0.4557 ± 0.0243`
- Test BalAcc_maj：`0.4557 ± 0.0243`
- Test 窗级 BalAcc（附报）：`0.4557 ± 0.0243`

### Three 分折明细

#### Fold 0

- stopped_epoch：`84` | best_epoch：`64`
- Val Acc_paper（早停）：`0.4578`
- Val BalAcc_maj（附报）：`0.4578`

**Test 试次级**
- Acc_paper：`0.4303`
- BalAcc_maj：`0.4303`
- F1-macro（众数）：`0.4261`
- Rec idle/left/right：`0.3145` / `0.4936` / `0.4827`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4303` | F1m：`0.4261`

#### Fold 1

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.4641`
- Val BalAcc_maj（附报）：`0.4641`

**Test 试次级**
- Acc_paper：`0.4755`
- BalAcc_maj：`0.4755`
- F1-macro（众数）：`0.4682`
- Rec idle/left/right：`0.5855` / `0.3209` / `0.5200`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4755` | F1m：`0.4682`

#### Fold 2

- stopped_epoch：`92` | best_epoch：`72`
- Val Acc_paper（早停）：`0.4167`
- Val BalAcc_maj（附报）：`0.4167`

**Test 试次级**
- Acc_paper：`0.4452`
- BalAcc_maj：`0.4452`
- F1-macro（众数）：`0.4385`
- Rec idle/left/right：`0.3036` / `0.4609` / `0.5709`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4452` | F1m：`0.4385`

#### Fold 3

- stopped_epoch：`117` | best_epoch：`97`
- Val Acc_paper（早停）：`0.5100`
- Val BalAcc_maj（附报）：`0.5100`

**Test 试次级**
- Acc_paper：`0.4927`
- BalAcc_maj：`0.4927`
- F1-macro（众数）：`0.4911`
- Rec idle/left/right：`0.4255` / `0.5782` / `0.4745`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4927` | F1m：`0.4911`

#### Fold 4

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.3941`
- Val BalAcc_maj（附报）：`0.3941`

**Test 试次级**
- Acc_paper：`0.4350`
- BalAcc_maj：`0.4350`
- F1-macro（众数）：`0.4310`
- Rec idle/left/right：`0.5560` / `0.3510` / `0.3980`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4350` | F1m：`0.4310`

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