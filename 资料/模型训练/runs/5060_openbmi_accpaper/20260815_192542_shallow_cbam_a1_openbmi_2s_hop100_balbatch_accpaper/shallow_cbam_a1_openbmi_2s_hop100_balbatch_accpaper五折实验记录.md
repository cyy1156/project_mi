# 被试独立五折实验记录（20260815_192542 / shallow_cbam_a1_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-15T19:25:42`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-cbam-self_model subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_cbam_a1` | A1 ConvTime→full CBAM→ConvSpat (electrode-overlap control) | attn=cbam_time
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_cbam_self_model_accpaper\shallow_cbam_a1_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260815_192542`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-cbam-self_model subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6987 ± 0.0318`
- Test Acc_paper：`0.7056 ± 0.0431`
- Test BalAcc_maj：`0.6864 ± 0.0237`
- Test 窗级 BalAcc（附报）：`0.6583 ± 0.0189`

### Task 分折明细

#### Fold 0

- stopped_epoch：`63` | best_epoch：`43`
- Val Acc_paper（早停）：`0.7030`
- Val BalAcc_maj（附报）：`0.6719`

**Test 试次级**
- Acc_paper：`0.7206`
- BalAcc_maj：`0.6914`
- Acc_majority：`0.7206`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6619` | F1：`0.7608` | Acc：`0.6887`

#### Fold 1

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.7333`
- Val BalAcc_maj（附报）：`0.6928`

**Test 试次级**
- Acc_paper：`0.7482`
- BalAcc_maj：`0.7236`
- Acc_majority：`0.7482`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6879` | F1：`0.7755` | Acc：`0.7095`

#### Fold 2

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.6707`
- Val BalAcc_maj（附报）：`0.6708`

**Test 试次级**
- Acc_paper：`0.6427`
- BalAcc_maj：`0.6600`
- Acc_majority：`0.6427`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6335` | F1：`0.6771` | Acc：`0.6210`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.7319`
- Val BalAcc_maj（附报）：`0.7061`

**Test 试次级**
- Acc_paper：`0.7488`
- BalAcc_maj：`0.6955`
- Acc_majority：`0.7488`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6653` | F1：`0.7909` | Acc：`0.7140`

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6544`
- Val BalAcc_maj（附报）：`0.6267`

**Test 试次级**
- Acc_paper：`0.6677`
- BalAcc_maj：`0.6618`
- Acc_majority：`0.6677`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6428` | F1：`0.7160` | Acc：`0.6495`

### Three
- Val Acc_paper：`0.5278 ± 0.0393`
- Test Acc_paper：`0.5477 ± 0.0263`
- Test BalAcc_maj：`0.5657 ± 0.0282`
- Test 窗级 BalAcc（附报）：`0.5350 ± 0.0252`

### Three 分折明细

#### Fold 0

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.5393`
- Val BalAcc_maj（附报）：`0.5556`

**Test 试次级**
- Acc_paper：`0.5285`
- BalAcc_maj：`0.5479`
- F1-macro（众数）：`0.5436`
- Rec idle/left/right：`0.6864` / `0.4455` / `0.5118`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5160` | F1m：`0.5132`

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5422`
- Val BalAcc_maj（附报）：`0.5548`

**Test 试次级**
- Acc_paper：`0.5724`
- BalAcc_maj：`0.5939`
- F1-macro（众数）：`0.5944`
- Rec idle/left/right：`0.6082` / `0.6127` / `0.5609`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5587` | F1m：`0.5589`

#### Fold 2

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.5059`
- Val BalAcc_maj（附报）：`0.5219`

**Test 试次级**
- Acc_paper：`0.5112`
- BalAcc_maj：`0.5261`
- F1-macro（众数）：`0.5208`
- Rec idle/left/right：`0.6327` / `0.5636` / `0.3818`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4978` | F1m：`0.4943`

#### Fold 3

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.5844`
- Val BalAcc_maj（附报）：`0.5981`

**Test 试次级**
- Acc_paper：`0.5815`
- BalAcc_maj：`0.6009`
- F1-macro（众数）：`0.5996`
- Rec idle/left/right：`0.5718` / `0.5300` / `0.7009`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5641` | F1m：`0.5628`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4670`
- Val BalAcc_maj（附报）：`0.4907`

**Test 试次级**
- Acc_paper：`0.5450`
- BalAcc_maj：`0.5597`
- F1-macro（众数）：`0.5596`
- Rec idle/left/right：`0.5750` / `0.5660` / `0.5380`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5384` | F1m：`0.5384`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-shallow-cbam-self_model subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-16T00:00:00`
