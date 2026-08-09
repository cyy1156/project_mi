# 被试独立五折实验记录（20260808_202912 / shallow_a1_lat_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-08T20:29:12`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-mi-feat subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_a1_lat` | ShallowFBCSPNet A1 laterality +2ch (C3-C4, CP3-CP4)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_mi_feat_openbmi_accpaper\shallow_a1_lat_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260808_202912`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-mi-feat subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6879 ± 0.0293`
- Test Acc_paper：`0.6978 ± 0.0306`
- Test BalAcc_maj：`0.6752 ± 0.0187`
- Test 窗级 BalAcc（附报）：`0.6499 ± 0.0161`

### Task 分折明细

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6852`
- Val BalAcc_maj（附报）：`0.6714`

**Test 试次级**
- Acc_paper：`0.7006`
- BalAcc_maj：`0.6841`
- Acc_majority：`0.7006`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6557` | F1：`0.7402` | Acc：`0.6713`

#### Fold 1

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.7289`
- Val BalAcc_maj（附报）：`0.6742`

**Test 试次级**
- Acc_paper：`0.7300`
- BalAcc_maj：`0.6975`
- Acc_majority：`0.7300`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6680` | F1：`0.7692` | Acc：`0.6974`

#### Fold 2

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.6559`
- Val BalAcc_maj（附报）：`0.6489`

**Test 试次级**
- Acc_paper：`0.6612`
- BalAcc_maj：`0.6525`
- Acc_majority：`0.6612`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6270` | F1：`0.7045` | Acc：`0.6353`

#### Fold 3

- stopped_epoch：`45` | best_epoch：`25`
- Val Acc_paper（早停）：`0.7126`
- Val BalAcc_maj（附报）：`0.6883`

**Test 试次级**
- Acc_paper：`0.7324`
- BalAcc_maj：`0.6884`
- Acc_majority：`0.7324`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6638` | F1：`0.7796` | Acc：`0.7042`

#### Fold 4

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6567`
- Val BalAcc_maj（附报）：`0.6058`

**Test 试次级**
- Acc_paper：`0.6647`
- BalAcc_maj：`0.6532`
- Acc_majority：`0.6647`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6352` | F1：`0.7178` | Acc：`0.6476`

### Three
- Val Acc_paper：`0.5213 ± 0.0311`
- Test Acc_paper：`0.5411 ± 0.0246`
- Test BalAcc_maj：`0.5568 ± 0.0263`
- Test 窗级 BalAcc（附报）：`0.5310 ± 0.0206`

### Three 分折明细

#### Fold 0

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.5252`
- Val BalAcc_maj（附报）：`0.5463`

**Test 试次级**
- Acc_paper：`0.5303`
- BalAcc_maj：`0.5485`
- F1-macro（众数）：`0.5478`
- Rec idle/left/right：`0.6073` / `0.5218` / `0.5164`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5218` | F1m：`0.5212`

#### Fold 1

- stopped_epoch：`63` | best_epoch：`43`
- Val Acc_paper（早停）：`0.5430`
- Val BalAcc_maj（附报）：`0.5604`

**Test 试次级**
- Acc_paper：`0.5694`
- BalAcc_maj：`0.5870`
- F1-macro（众数）：`0.5872`
- Rec idle/left/right：`0.6136` / `0.5764` / `0.5709`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5552` | F1m：`0.5553`

#### Fold 2

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.5074`
- Val BalAcc_maj（附报）：`0.5241`

**Test 试次级**
- Acc_paper：`0.5121`
- BalAcc_maj：`0.5248`
- F1-macro（众数）：`0.5216`
- Rec idle/left/right：`0.5891` / `0.5809` / `0.4045`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5040` | F1m：`0.5018`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5607`
- Val BalAcc_maj（附报）：`0.5737`

**Test 试次级**
- Acc_paper：`0.5712`
- BalAcc_maj：`0.5882`
- F1-macro（众数）：`0.5863`
- Rec idle/left/right：`0.5209` / `0.5491` / `0.6945`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5552` | F1m：`0.5539`

#### Fold 4

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.4704`
- Val BalAcc_maj（附报）：`0.4907`

**Test 试次级**
- Acc_paper：`0.5223`
- BalAcc_maj：`0.5353`
- F1-macro（众数）：`0.5348`
- Rec idle/left/right：`0.5720` / `0.5430` / `0.4910`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5189` | F1m：`0.5188`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-shallow-mi-feat subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-08T23:35:28`
