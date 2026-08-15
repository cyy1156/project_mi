# 被试独立五折实验记录（20260811_011823 / dbn_raw_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-11T01:18:23`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dbn_raw` | TemporalEncoder(D=64) + DBN；OpenBMI 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\dbn_raw_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260811_011823`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7050 ± 0.0285`
- Test Acc_paper：`0.6880 ± 0.0177`
- Test BalAcc_maj：`0.5838 ± 0.0128`
- Test 窗级 BalAcc（附报）：`0.5831 ± 0.0118`

### Task 分折明细

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.7126`
- Val BalAcc_maj（附报）：`0.6000`

**Test 试次级**
- Acc_paper：`0.6836`
- BalAcc_maj：`0.5848`
- Acc_majority：`0.6836`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5839` | F1：`0.7836` | Acc：`0.6794`

#### Fold 1

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.7304`
- Val BalAcc_maj（附报）：`0.6250`

**Test 试次级**
- Acc_paper：`0.6721`
- BalAcc_maj：`0.5775`
- Acc_majority：`0.6721`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5811` | F1：`0.7767` | Acc：`0.6724`

#### Fold 2

- stopped_epoch：`61` | best_epoch：`41`
- Val Acc_paper（早停）：`0.6715`
- Val BalAcc_maj（附报）：`0.5953`

**Test 试次级**
- Acc_paper：`0.6670`
- BalAcc_maj：`0.5682`
- Acc_majority：`0.6670`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5649` | F1：`0.7697` | Acc：`0.6604`

#### Fold 3

- stopped_epoch：`55` | best_epoch：`35`
- Val Acc_paper（早停）：`0.7389`
- Val BalAcc_maj（附报）：`0.6475`

**Test 试次级**
- Acc_paper：`0.7091`
- BalAcc_maj：`0.6068`
- Acc_majority：`0.7091`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6022` | F1：`0.8001` | Acc：`0.7009`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6719`
- Val BalAcc_maj（附报）：`0.5331`

**Test 试次级**
- Acc_paper：`0.7083`
- BalAcc_maj：`0.5817`
- Acc_majority：`0.7083`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5835` | F1：`0.8113` | Acc：`0.7055`

### Three
- Val Acc_paper：`0.5033 ± 0.0405`
- Test Acc_paper：`0.4969 ± 0.0250`
- Test BalAcc_maj：`0.5046 ± 0.0245`
- Test 窗级 BalAcc（附报）：`0.4952 ± 0.0220`

### Three 分折明细

#### Fold 0

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.5170`
- Val BalAcc_maj（附报）：`0.5270`

**Test 试次级**
- Acc_paper：`0.4870`
- BalAcc_maj：`0.4955`
- F1-macro（众数）：`0.4948`
- Rec idle/left/right：`0.4409` / `0.5345` / `0.5109`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4861` | F1m：`0.4854`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5359`
- Val BalAcc_maj（附报）：`0.5433`

**Test 试次级**
- Acc_paper：`0.5227`
- BalAcc_maj：`0.5312`
- F1-macro（众数）：`0.5272`
- Rec idle/left/right：`0.3973` / `0.5909` / `0.6055`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5195` | F1m：`0.5153`

#### Fold 2

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.4622`
- Val BalAcc_maj（附报）：`0.4733`

**Test 试次级**
- Acc_paper：`0.4552`
- BalAcc_maj：`0.4636`
- F1-macro（众数）：`0.4442`
- Rec idle/left/right：`0.2555` / `0.7218` / `0.4136`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4579` | F1m：`0.4407`

#### Fold 3

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5519`
- Val BalAcc_maj（附报）：`0.5607`

**Test 试次级**
- Acc_paper：`0.5215`
- BalAcc_maj：`0.5279`
- F1-macro（众数）：`0.5237`
- Rec idle/left/right：`0.4227` / `0.4973` / `0.6636`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5140` | F1m：`0.5104`

#### Fold 4

- stopped_epoch：`70` | best_epoch：`50`
- Val Acc_paper（早停）：`0.4493`
- Val BalAcc_maj（附报）：`0.4548`

**Test 试次级**
- Acc_paper：`0.4983`
- BalAcc_maj：`0.5047`
- F1-macro（众数）：`0.4936`
- Rec idle/left/right：`0.3640` / `0.7400` / `0.4100`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4987` | F1m：`0.4880`

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

- 结束：`2026-08-11T03:13:30`
