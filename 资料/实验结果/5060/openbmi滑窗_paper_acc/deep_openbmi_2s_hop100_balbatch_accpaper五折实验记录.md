# 被试独立五折实验记录（20260807_230300 / deep_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-07T23:03:00`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`deep` | Deep4Net-compat（pool=1/1，对齐1s自动缩核；非满血stride3）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\deep_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260807_230300`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6980 ± 0.0316`
- Test Acc_paper：`0.7169 ± 0.0335`
- Test BalAcc_maj：`0.6774 ± 0.0214`
- Test 窗级 BalAcc（附报）：`0.6523 ± 0.0200`

### Task 分折明细

#### Fold 0

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.6926`
- Val BalAcc_maj（附报）：`0.6669`

**Test 试次级**
- Acc_paper：`0.7118`
- BalAcc_maj：`0.6761`
- Acc_majority：`0.7118`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6461` | F1：`0.7573` | Acc：`0.6802`

#### Fold 1

- stopped_epoch：`77` | best_epoch：`57`
- Val Acc_paper（早停）：`0.7400`
- Val BalAcc_maj（附报）：`0.6808`

**Test 试次级**
- Acc_paper：`0.7530`
- BalAcc_maj：`0.7070`
- Acc_majority：`0.7530`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6821` | F1：`0.7959` | Acc：`0.7239`

#### Fold 2

- stopped_epoch：`84` | best_epoch：`64`
- Val Acc_paper（早停）：`0.6519`
- Val BalAcc_maj（附报）：`0.6478`

**Test 试次级**
- Acc_paper：`0.6742`
- BalAcc_maj：`0.6480`
- Acc_majority：`0.6742`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6253` | F1：`0.7277` | Acc：`0.6504`

#### Fold 3

- stopped_epoch：`90` | best_epoch：`70`
- Val Acc_paper（早停）：`0.7256`
- Val BalAcc_maj（附报）：`0.6869`

**Test 试次级**
- Acc_paper：`0.7573`
- BalAcc_maj：`0.6945`
- Acc_majority：`0.7573`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6668` | F1：`0.8038` | Acc：`0.7256`

#### Fold 4

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.6800`
- Val BalAcc_maj（附报）：`0.6128`

**Test 试次级**
- Acc_paper：`0.6883`
- BalAcc_maj：`0.6615`
- Acc_majority：`0.6883`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6411` | F1：`0.7397` | Acc：`0.6652`

### Three
- Val Acc_paper：`0.5212 ± 0.0381`
- Test Acc_paper：`0.5400 ± 0.0296`
- Test BalAcc_maj：`0.5574 ± 0.0279`
- Test 窗级 BalAcc（附报）：`0.5313 ± 0.0237`

### Three 分折明细

#### Fold 0

- stopped_epoch：`86` | best_epoch：`66`
- Val Acc_paper（早停）：`0.5263`
- Val BalAcc_maj（附报）：`0.5463`

**Test 试次级**
- Acc_paper：`0.5242`
- BalAcc_maj：`0.5427`
- F1-macro（众数）：`0.5389`
- Rec idle/left/right：`0.6618` / `0.4355` / `0.5309`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5170` | F1m：`0.5146`

#### Fold 1

- stopped_epoch：`72` | best_epoch：`52`
- Val Acc_paper（早停）：`0.5478`
- Val BalAcc_maj（附报）：`0.5630`

**Test 试次级**
- Acc_paper：`0.5779`
- BalAcc_maj：`0.5915`
- F1-macro（众数）：`0.5921`
- Rec idle/left/right：`0.5655` / `0.6664` / `0.5427`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5613` | F1m：`0.5610`

#### Fold 2

- stopped_epoch：`65` | best_epoch：`45`
- Val Acc_paper（早停）：`0.4978`
- Val BalAcc_maj（附报）：`0.5152`

**Test 试次级**
- Acc_paper：`0.5067`
- BalAcc_maj：`0.5258`
- F1-macro（众数）：`0.5251`
- Rec idle/left/right：`0.5873` / `0.4900` / `0.5000`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5012` | F1m：`0.5010`

#### Fold 3

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.5715`
- Val BalAcc_maj（附报）：`0.5848`

**Test 试次级**
- Acc_paper：`0.5733`
- BalAcc_maj：`0.5903`
- F1-macro（众数）：`0.5899`
- Rec idle/left/right：`0.6045` / `0.5409` / `0.6255`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5569` | F1m：`0.5568`

#### Fold 4

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.4626`
- Val BalAcc_maj（附报）：`0.4841`

**Test 试次级**
- Acc_paper：`0.5180`
- BalAcc_maj：`0.5367`
- F1-macro（众数）：`0.5354`
- Rec idle/left/right：`0.6200` / `0.5160` / `0.4740`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5199` | F1m：`0.5190`

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

- 结束：`2026-08-08T08:29:54`
