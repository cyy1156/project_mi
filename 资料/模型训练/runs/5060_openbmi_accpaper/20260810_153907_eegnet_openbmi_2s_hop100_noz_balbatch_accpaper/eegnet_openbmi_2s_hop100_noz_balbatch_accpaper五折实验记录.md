# 被试独立五折实验记录（20260810_153907 / eegnet_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-10T15:39:07`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`eegnet` | EEGNet F1=8, D=2, F2=16（默认池化）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\eegnet_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260810_153907`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
# 被试独立五折实验记录（20260810_153907 / eegnet_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-10T18:07:47`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`eegnet` | EEGNet F1=8, D=2, F2=16（默认池化）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\eegnet_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260810_153907`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6814 ± 0.0299`
- Test Acc_paper：`0.6814 ± 0.0146`
- Test BalAcc_maj：`0.6241 ± 0.0200`
- Test 窗级 BalAcc（附报）：`0.6140 ± 0.0162`

### Task 分折明细

#### Fold 0

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.6956`
- Val BalAcc_maj（附报）：`0.6131`

**Test 试次级**
- Acc_paper：`0.6964`
- BalAcc_maj：`0.6200`
- Acc_majority：`0.6964`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6071` | F1：`0.7780` | Acc：`0.6827`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.7330`
- Val BalAcc_maj（附报）：`0.6619`

**Test 试次级**
- Acc_paper：`0.6985`
- BalAcc_maj：`0.6466`
- Acc_majority：`0.6985`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6359` | F1：`0.7690` | Acc：`0.6856`

#### Fold 2

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6504`
- Val BalAcc_maj（附报）：`0.5981`

**Test 试次级**
- Acc_paper：`0.6645`
- BalAcc_maj：`0.5916`
- Acc_majority：`0.6645`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5885` | F1：`0.7552` | Acc：`0.6569`

#### Fold 3

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.6689`
- Val BalAcc_maj（附报）：`0.6256`

**Test 试次级**
- Acc_paper：`0.6824`
- BalAcc_maj：`0.6436`
- Acc_majority：`0.6824`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6255` | F1：`0.7436` | Acc：`0.6620`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6593`
- Val BalAcc_maj（附报）：`0.5661`

**Test 试次级**
- Acc_paper：`0.6653`
- BalAcc_maj：`0.6185`
- Acc_majority：`0.6653`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6133` | F1：`0.7438` | Acc：`0.6575`

### Three
- Val Acc_paper：`0.5061 ± 0.0423`
- Test Acc_paper：`0.5224 ± 0.0266`
- Test BalAcc_maj：`0.5321 ± 0.0267`
- Test 窗级 BalAcc（附报）：`0.5196 ± 0.0215`

### Three 分折明细

#### Fold 0

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.5211`
- Val BalAcc_maj（附报）：`0.5289`

**Test 试次级**
- Acc_paper：`0.4876`
- BalAcc_maj：`0.4964`
- F1-macro（众数）：`0.4869`
- Rec idle/left/right：`0.3127` / `0.5509` / `0.6255`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4925` | F1m：`0.4834`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5526`
- Val BalAcc_maj（附报）：`0.5622`

**Test 试次级**
- Acc_paper：`0.5494`
- BalAcc_maj：`0.5567`
- F1-macro（众数）：`0.5489`
- Rec idle/left/right：`0.3818` / `0.7273` / `0.5609`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5437` | F1m：`0.5369`

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.4741`
- Val BalAcc_maj（附报）：`0.4804`

**Test 试次级**
- Acc_paper：`0.4942`
- BalAcc_maj：`0.5039`
- F1-macro（众数）：`0.4843`
- Rec idle/left/right：`0.2627` / `0.7309` / `0.5182`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4955` | F1m：`0.4788`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5419`
- Val BalAcc_maj（附报）：`0.5496`

**Test 试次级**
- Acc_paper：`0.5491`
- BalAcc_maj：`0.5591`
- F1-macro（众数）：`0.5523`
- Rec idle/left/right：`0.4855` / `0.4327` / `0.7591`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5379` | F1m：`0.5317`

#### Fold 4

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.4407`
- Val BalAcc_maj（附报）：`0.4456`

**Test 试次级**
- Acc_paper：`0.5317`
- BalAcc_maj：`0.5443`
- F1-macro（众数）：`0.5393`
- Rec idle/left/right：`0.4130` / `0.6860` / `0.5340`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5285` | F1m：`0.5246`

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

- 结束：`2026-08-10T18:49:21`
