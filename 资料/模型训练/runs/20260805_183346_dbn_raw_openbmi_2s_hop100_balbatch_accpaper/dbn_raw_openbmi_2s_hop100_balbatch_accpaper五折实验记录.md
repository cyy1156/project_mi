# 被试独立五折实验记录（20260805_183346 / dbn_raw_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-05T18:33:46`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dbn_raw` | TemporalEncoder(D=64) + DBN；OpenBMI 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_openbmi_2s_hop100_accpaper\dbn_raw_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260805_183346`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7118 ± 0.0448`
- Test Acc_paper：`0.6919 ± 0.0458`
- Test BalAcc_maj：`0.6395 ± 0.0175`
- Test 窗级 BalAcc（附报）：`0.6275 ± 0.0138`

### Task 分折明细

#### Fold 0

- stopped_epoch：`113` | best_epoch：`93`
- Val Acc_paper（早停）：`0.7363`
- Val BalAcc_maj（附报）：`0.6581`

**Test 试次级**
- Acc_paper：`0.7003`
- BalAcc_maj：`0.6075`
- Acc_majority：`0.7003`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6038` | F1：`0.7865` | Acc：`0.6890`

#### Fold 1

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.7304`
- Val BalAcc_maj（附报）：`0.6247`

**Test 试次级**
- Acc_paper：`0.7415`
- BalAcc_maj：`0.6536`
- Acc_majority：`0.7415`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6412` | F1：`0.8112` | Acc：`0.7240`

#### Fold 2

- stopped_epoch：`62` | best_epoch：`42`
- Val Acc_paper（早停）：`0.6663`
- Val BalAcc_maj（附报）：`0.6722`

**Test 试次级**
- Acc_paper：`0.6082`
- BalAcc_maj：`0.6364`
- Acc_majority：`0.6082`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6207` | F1：`0.6466` | Acc：`0.5978`

#### Fold 3

- stopped_epoch：`111` | best_epoch：`91`
- Val Acc_paper（早停）：`0.7722`
- Val BalAcc_maj（附报）：`0.7125`

**Test 试次级**
- Acc_paper：`0.7224`
- BalAcc_maj：`0.6434`
- Acc_majority：`0.7224`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6333` | F1：`0.7943` | Acc：`0.7060`

#### Fold 4

- stopped_epoch：`103` | best_epoch：`83`
- Val Acc_paper（早停）：`0.6537`
- Val BalAcc_maj（附报）：`0.6050`

**Test 试次级**
- Acc_paper：`0.6870`
- BalAcc_maj：`0.6565`
- Acc_majority：`0.6870`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6387` | F1：`0.7428` | Acc：`0.6666`

### Three
- Val Acc_paper：`0.4875 ± 0.0359`
- Test Acc_paper：`0.4883 ± 0.0359`
- Test BalAcc_maj：`0.5040 ± 0.0354`
- Test 窗级 BalAcc（附报）：`0.4867 ± 0.0291`

### Three 分折明细

#### Fold 0

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.4881`
- Val BalAcc_maj（附报）：`0.5037`

**Test 试次级**
- Acc_paper：`0.4785`
- BalAcc_maj：`0.4967`
- F1-macro（众数）：`0.4906`
- Rec idle/left/right：`0.3682` / `0.6664` / `0.4555`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4795` | F1m：`0.4741`

#### Fold 1

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.4907`
- Val BalAcc_maj（附报）：`0.5048`

**Test 试次级**
- Acc_paper：`0.5479`
- BalAcc_maj：`0.5615`
- F1-macro（众数）：`0.5586`
- Rec idle/left/right：`0.5200` / `0.7173` / `0.4473`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5378` | F1m：`0.5354`

#### Fold 2

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.4744`
- Val BalAcc_maj（附报）：`0.4807`

**Test 试次级**
- Acc_paper：`0.4406`
- BalAcc_maj：`0.4585`
- F1-macro（众数）：`0.4491`
- Rec idle/left/right：`0.4773` / `0.6100` / `0.2882`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4513` | F1m：`0.4446`

#### Fold 3

- stopped_epoch：`68` | best_epoch：`48`
- Val Acc_paper（早停）：`0.5478`
- Val BalAcc_maj（附报）：`0.5552`

**Test 试次级**
- Acc_paper：`0.4709`
- BalAcc_maj：`0.4815`
- F1-macro（众数）：`0.4764`
- Rec idle/left/right：`0.3318` / `0.6009` / `0.5118`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4706` | F1m：`0.4659`

#### Fold 4

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.4363`
- Val BalAcc_maj（附报）：`0.4507`

**Test 试次级**
- Acc_paper：`0.5037`
- BalAcc_maj：`0.5217`
- F1-macro（众数）：`0.5194`
- Rec idle/left/right：`0.5320` / `0.4270` / `0.6060`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4943` | F1m：`0.4925`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100",
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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
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

- 结束：`2026-08-05T22:35:09`
