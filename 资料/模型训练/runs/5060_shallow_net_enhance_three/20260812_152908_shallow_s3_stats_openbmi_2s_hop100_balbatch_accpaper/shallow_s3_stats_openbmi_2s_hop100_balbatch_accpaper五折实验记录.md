# 被试独立五折实验记录（20260812_152908 / shallow_s3_stats_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-12T15:29:08`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_s3_stats` | ShallowFBCSPNet S3_stats | mean/std/max over T + Linear
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_net_enhance_three_accpaper\shallow_s3_stats_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260812_152908`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 1024, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 4, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7053 ± 0.0260`
- Test Acc_paper：`0.7102 ± 0.0188`
- Test BalAcc_maj：`0.6378 ± 0.0143`
- Test 窗级 BalAcc（附报）：`0.6247 ± 0.0143`

### Task 分折明细

#### Fold 0

- stopped_epoch：`76` | best_epoch：`56`
- Val Acc_paper（早停）：`0.7007`
- Val BalAcc_maj（附报）：`0.6236`

**Test 试次级**
- Acc_paper：`0.7206`
- BalAcc_maj：`0.6357`
- Acc_majority：`0.7206`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6259` | F1：`0.7974` | Acc：`0.7063`

#### Fold 1

- stopped_epoch：`45` | best_epoch：`25`
- Val Acc_paper（早停）：`0.7270`
- Val BalAcc_maj（附报）：`0.6411`

**Test 试次级**
- Acc_paper：`0.7264`
- BalAcc_maj：`0.6652`
- Acc_majority：`0.7264`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6517` | F1：`0.7922` | Acc：`0.7105`

#### Fold 2

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.6711`
- Val BalAcc_maj（附报）：`0.6344`

**Test 试次级**
- Acc_paper：`0.6752`
- BalAcc_maj：`0.6291`
- Acc_majority：`0.6752`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6117` | F1：`0.7425` | Acc：`0.6559`

#### Fold 3

- stopped_epoch：`94` | best_epoch：`74`
- Val Acc_paper（早停）：`0.7419`
- Val BalAcc_maj（附报）：`0.6758`

**Test 试次级**
- Acc_paper：`0.7224`
- BalAcc_maj：`0.6243`
- Acc_majority：`0.7224`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6149` | F1：`0.8019` | Acc：`0.7067`

#### Fold 4

- stopped_epoch：`148` | best_epoch：`128`
- Val Acc_paper（早停）：`0.6859`
- Val BalAcc_maj（附报）：`0.5856`

**Test 试次级**
- Acc_paper：`0.7067`
- BalAcc_maj：`0.6348`
- Acc_majority：`0.7067`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6194` | F1：`0.7833` | Acc：`0.6916`

### Three
- Val Acc_paper：`0.5068 ± 0.0301`
- Test Acc_paper：`0.5286 ± 0.0168`
- Test BalAcc_maj：`0.5404 ± 0.0154`
- Test 窗级 BalAcc（附报）：`0.5213 ± 0.0157`

### Three 分折明细

#### Fold 0

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.5089`
- Val BalAcc_maj（附报）：`0.5181`

**Test 试次级**
- Acc_paper：`0.5109`
- BalAcc_maj：`0.5276`
- F1-macro（众数）：`0.5250`
- Rec idle/left/right：`0.4136` / `0.5864` / `0.5827`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5071` | F1m：`0.5047`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5185`
- Val BalAcc_maj（附报）：`0.5322`

**Test 试次级**
- Acc_paper：`0.5409`
- BalAcc_maj：`0.5512`
- F1-macro（众数）：`0.5461`
- Rec idle/left/right：`0.3936` / `0.6645` / `0.5955`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5335` | F1m：`0.5289`

#### Fold 2

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4922`
- Val BalAcc_maj（附报）：`0.5048`

**Test 试次级**
- Acc_paper：`0.5103`
- BalAcc_maj：`0.5224`
- F1-macro（众数）：`0.5194`
- Rec idle/left/right：`0.4682` / `0.6473` / `0.4518`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4992` | F1m：`0.4963`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5530`
- Val BalAcc_maj（附报）：`0.5611`

**Test 试次级**
- Acc_paper：`0.5533`
- BalAcc_maj：`0.5642`
- F1-macro（众数）：`0.5545`
- Rec idle/left/right：`0.3682` / `0.5891` / `0.7355`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5407` | F1m：`0.5313`

#### Fold 4

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.4615`
- Val BalAcc_maj（附报）：`0.4781`

**Test 试次级**
- Acc_paper：`0.5273`
- BalAcc_maj：`0.5367`
- F1-macro（众数）：`0.5356`
- Rec idle/left/right：`0.4680` / `0.5730` / `0.5690`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5261` | F1m：`0.5249`

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
  "batch_eval": 1024,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "prefetch_factor": 4,
  "non_blocking": true,
  "torch_num_threads": 6,
  "cudnn_benchmark": true,
  "deterministic": false,
  "use_amp": true
}
```

- 结束：`2026-08-12T18:38:33`
