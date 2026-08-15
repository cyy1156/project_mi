# 被试独立五折实验记录（20260810_010148 / deep_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-10T01:01:48`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`deep` | Deep4Net-compat（pool=1/1，对齐1s自动缩核；非满血stride3）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\deep_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260810_010148`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7020 ± 0.0319`
- Test Acc_paper：`0.7164 ± 0.0139`
- Test BalAcc_maj：`0.6523 ± 0.0252`
- Test 窗级 BalAcc（附报）：`0.6367 ± 0.0218`

### Task 分折明细

#### Fold 0

- stopped_epoch：`101` | best_epoch：`81`
- Val Acc_paper（早停）：`0.7463`
- Val BalAcc_maj（附报）：`0.6647`

**Test 试次级**
- Acc_paper：`0.6988`
- BalAcc_maj：`0.6498`
- Acc_majority：`0.6988`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6353` | F1：`0.7606` | Acc：`0.6788`

#### Fold 1

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.7193`
- Val BalAcc_maj（附报）：`0.6364`

**Test 试次级**
- Acc_paper：`0.7385`
- BalAcc_maj：`0.6720`
- Acc_majority：`0.7385`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6565` | F1：`0.8015` | Acc：`0.7201`

#### Fold 2

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.6559`
- Val BalAcc_maj（附报）：`0.5978`

**Test 试次级**
- Acc_paper：`0.7045`
- BalAcc_maj：`0.6075`
- Acc_majority：`0.7045`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5985` | F1：`0.7884` | Acc：`0.6888`

#### Fold 3

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.7115`
- Val BalAcc_maj（附报）：`0.6325`

**Test 试次级**
- Acc_paper：`0.7203`
- BalAcc_maj：`0.6518`
- Acc_majority：`0.7203`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6339` | F1：`0.7853` | Acc：`0.6984`

#### Fold 4

- stopped_epoch：`55` | best_epoch：`35`
- Val Acc_paper（早停）：`0.6770`
- Val BalAcc_maj（附报）：`0.6031`

**Test 试次级**
- Acc_paper：`0.7197`
- BalAcc_maj：`0.6803`
- Acc_majority：`0.7197`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6593` | F1：`0.7707` | Acc：`0.6955`

### Three
- Val Acc_paper：`0.5157 ± 0.0380`
- Test Acc_paper：`0.5366 ± 0.0247`
- Test BalAcc_maj：`0.5503 ± 0.0245`
- Test 窗级 BalAcc（附报）：`0.5303 ± 0.0216`

### Three 分折明细

#### Fold 0

- stopped_epoch：`72` | best_epoch：`52`
- Val Acc_paper（早停）：`0.5430`
- Val BalAcc_maj（附报）：`0.5559`

**Test 试次级**
- Acc_paper：`0.5055`
- BalAcc_maj：`0.5185`
- F1-macro（众数）：`0.5182`
- Rec idle/left/right：`0.5536` / `0.5109` / `0.4909`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5017` | F1m：`0.5016`

#### Fold 1

- stopped_epoch：`91` | best_epoch：`71`
- Val Acc_paper（早停）：`0.5374`
- Val BalAcc_maj（附报）：`0.5526`

**Test 试次级**
- Acc_paper：`0.5564`
- BalAcc_maj：`0.5676`
- F1-macro（众数）：`0.5673`
- Rec idle/left/right：`0.5182` / `0.6700` / `0.5145`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5476` | F1m：`0.5468`

#### Fold 2

- stopped_epoch：`103` | best_epoch：`83`
- Val Acc_paper（早停）：`0.4963`
- Val BalAcc_maj（附报）：`0.5130`

**Test 试次级**
- Acc_paper：`0.5076`
- BalAcc_maj：`0.5221`
- F1-macro（众数）：`0.5157`
- Rec idle/left/right：`0.3936` / `0.7027` / `0.4700`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5061` | F1m：`0.5006`

#### Fold 3

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.5519`
- Val BalAcc_maj（附报）：`0.5600`

**Test 试次级**
- Acc_paper：`0.5609`
- BalAcc_maj：`0.5712`
- F1-macro（众数）：`0.5666`
- Rec idle/left/right：`0.4200` / `0.5836` / `0.7100`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5489` | F1m：`0.5443`

#### Fold 4

- stopped_epoch：`71` | best_epoch：`51`
- Val Acc_paper（早停）：`0.4500`
- Val BalAcc_maj（附报）：`0.4600`

**Test 试次级**
- Acc_paper：`0.5527`
- BalAcc_maj：`0.5720`
- F1-macro（众数）：`0.5699`
- Rec idle/left/right：`0.5180` / `0.6790` / `0.5190`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5469` | F1m：`0.5453`

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

- 结束：`2026-08-10T10:40:58`
