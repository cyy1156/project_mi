# 被试独立五折实验记录（20260823_131435 / eegnet_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-23T13:14:35`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`eegnet` | EEGNet F1=8, D=2, F2=16 · 3s · 5090
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\eegnet_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_131435`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0}`

---
## 最终结论（主报 Acc_paper）

### Task
- （本次跳过 · three-only）

### Three
- Val Acc_paper：`0.5487 ± 0.0343`
- Test Acc_paper：`0.5629 ± 0.0259`
- Test BalAcc_maj：`0.5661 ± 0.0254`
- Test 窗级 BalAcc（附报）：`0.5581 ± 0.0236`

### Three 分折明细

#### Fold 0

- stopped_epoch：`70` | best_epoch：`50`
- Val Acc_paper（早停）：`0.5570`
- Val BalAcc_maj（附报）：`0.5585`

**Test 试次级**
- Acc_paper：`0.5421`
- BalAcc_maj：`0.5464`
- F1-macro（众数）：`0.5437`
- Rec idle/left/right：`0.4491` / `0.6755` / `0.5145`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5368` | F1m：`0.5343`

#### Fold 1

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.5667`
- Val BalAcc_maj（附报）：`0.5693`

**Test 试次级**
- Acc_paper：`0.6036`
- BalAcc_maj：`0.6058`
- F1-macro（众数）：`0.6073`
- Rec idle/left/right：`0.5518` / `0.6482` / `0.6173`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5924` | F1m：`0.5933`

#### Fold 2

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.5322`
- Val BalAcc_maj（附报）：`0.5359`

**Test 试次级**
- Acc_paper：`0.5482`
- BalAcc_maj：`0.5527`
- F1-macro（众数）：`0.5478`
- Rec idle/left/right：`0.5600` / `0.6973` / `0.4009`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5472` | F1m：`0.5424`

#### Fold 3

- stopped_epoch：`71` | best_epoch：`51`
- Val Acc_paper（早停）：`0.5948`
- Val BalAcc_maj（附报）：`0.5993`

**Test 试次级**
- Acc_paper：`0.5830`
- BalAcc_maj：`0.5858`
- F1-macro（众数）：`0.5806`
- Rec idle/left/right：`0.4673` / `0.5064` / `0.7836`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5799` | F1m：`0.5748`

#### Fold 4

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.4930`
- Val BalAcc_maj（附报）：`0.4959`

**Test 试次级**
- Acc_paper：`0.5373`
- BalAcc_maj：`0.5397`
- F1-macro（众数）：`0.5398`
- Rec idle/left/right：`0.5590` / `0.5080` / `0.5520`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5344` | F1m：`0.5345`

### 共用超参
```json
{
  "data_tag": "openbmi_3s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 20,
  "batch_train": 256,
  "batch_eval": 512,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 750,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 0,
  "pin_memory": false,
  "persistent_workers": false,
  "prefetch_factor": 2,
  "non_blocking": true,
  "torch_num_threads": 8,
  "cudnn_benchmark": true,
  "deterministic": false,
  "use_amp": true,
  "t0_weight_alpha": 0.0,
  "t0_filter_max": 0.0
}
```

- 结束：`2026-08-23T13:49:07`
