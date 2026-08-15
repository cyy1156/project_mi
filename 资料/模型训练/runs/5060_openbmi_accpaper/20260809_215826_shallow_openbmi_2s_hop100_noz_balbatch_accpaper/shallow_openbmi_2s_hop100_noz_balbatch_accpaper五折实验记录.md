# 被试独立五折实验记录（20260809_215826 / shallow_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-09T21:58:26`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\shallow_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260809_215826`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6883 ± 0.0305`
- Test Acc_paper：`0.6964 ± 0.0276`
- Test BalAcc_maj：`0.6701 ± 0.0199`
- Test 窗级 BalAcc（附报）：`0.6521 ± 0.0181`

### Task 分折明细

#### Fold 0

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.7093`
- Val BalAcc_maj（附报）：`0.6839`

**Test 试次级**
- Acc_paper：`0.6970`
- BalAcc_maj：`0.6707`
- Acc_majority：`0.6970`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6498` | F1：`0.7496` | Acc：`0.6758`

#### Fold 1

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.7263`
- Val BalAcc_maj（附报）：`0.6669`

**Test 试次级**
- Acc_paper：`0.7394`
- BalAcc_maj：`0.7000`
- Acc_majority：`0.7394`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6819` | F1：`0.7879` | Acc：`0.7172`

#### Fold 2

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.6474`
- Val BalAcc_maj（附报）：`0.6425`

**Test 试次级**
- Acc_paper：`0.6621`
- BalAcc_maj：`0.6414`
- Acc_majority：`0.6621`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6268` | F1：`0.7199` | Acc：`0.6456`

#### Fold 3

- stopped_epoch：`60` | best_epoch：`40`
- Val Acc_paper（早停）：`0.7007`
- Val BalAcc_maj（附报）：`0.6747`

**Test 试次级**
- Acc_paper：`0.7112`
- BalAcc_maj：`0.6805`
- Acc_majority：`0.7112`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6585` | F1：`0.7629` | Acc：`0.6891`

#### Fold 4

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.6578`
- Val BalAcc_maj（附报）：`0.5992`

**Test 试次级**
- Acc_paper：`0.6723`
- BalAcc_maj：`0.6578`
- Acc_majority：`0.6723`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6437` | F1：`0.7293` | Acc：`0.6589`

### Three
- Val Acc_paper：`0.5242 ± 0.0383`
- Test Acc_paper：`0.5422 ± 0.0282`
- Test BalAcc_maj：`0.5561 ± 0.0301`
- Test 窗级 BalAcc（附报）：`0.5354 ± 0.0250`

### Three 分折明细

#### Fold 0

- stopped_epoch：`55` | best_epoch：`35`
- Val Acc_paper（早停）：`0.5407`
- Val BalAcc_maj（附报）：`0.5541`

**Test 试次级**
- Acc_paper：`0.5109`
- BalAcc_maj：`0.5261`
- F1-macro（众数）：`0.5259`
- Rec idle/left/right：`0.5445` / `0.4982` / `0.5355`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5080` | F1m：`0.5079`

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5537`
- Val BalAcc_maj（附报）：`0.5659`

**Test 试次级**
- Acc_paper：`0.5736`
- BalAcc_maj：`0.5879`
- F1-macro（众数）：`0.5885`
- Rec idle/left/right：`0.5455` / `0.6818` / `0.5364`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5640` | F1m：`0.5640`

#### Fold 2

- stopped_epoch：`64` | best_epoch：`44`
- Val Acc_paper（早停）：`0.5100`
- Val BalAcc_maj（附报）：`0.5241`

**Test 试次级**
- Acc_paper：`0.5103`
- BalAcc_maj：`0.5191`
- F1-macro（众数）：`0.5167`
- Rec idle/left/right：`0.5327` / `0.6091` / `0.4155`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5057` | F1m：`0.5028`

#### Fold 3

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.5607`
- Val BalAcc_maj（附报）：`0.5704`

**Test 试次级**
- Acc_paper：`0.5736`
- BalAcc_maj：`0.5915`
- F1-macro（众数）：`0.5912`
- Rec idle/left/right：`0.6155` / `0.5518` / `0.6073`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5611` | F1m：`0.5609`

#### Fold 4

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.4559`
- Val BalAcc_maj（附报）：`0.4700`

**Test 试次级**
- Acc_paper：`0.5423`
- BalAcc_maj：`0.5560`
- F1-macro（众数）：`0.5557`
- Rec idle/left/right：`0.5380` / `0.5930` / `0.5370`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5380` | F1m：`0.5379`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100_noz",
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
  "protocol": "2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 4,
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

- 结束：`2026-08-10T01:01:42`
