# 被试独立五折实验记录（20260823_123900 / shallow_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-23T12:39:00`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet · Tw=3s hop=100ms · 方案24·5090
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_123900`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.6, 't0_filter_max': 0.0}`

---
## 最终结论（主报 Acc_paper）

### Task
- （本次跳过 · three-only）

### Three
- Val Acc_paper：`0.5704 ± 0.0278`
- Test Acc_paper：`0.5886 ± 0.0283`
- Test BalAcc_maj：`0.5923 ± 0.0277`
- Test 窗级 BalAcc（附报）：`0.5819 ± 0.0249`

### Three 分折明细

#### Fold 0

- stopped_epoch：`55` | best_epoch：`35`
- Val Acc_paper（早停）：`0.5796`
- Val BalAcc_maj（附报）：`0.5826`

**Test 试次级**
- Acc_paper：`0.5558`
- BalAcc_maj：`0.5597`
- F1-macro（众数）：`0.5595`
- Rec idle/left/right：`0.5845` / `0.5682` / `0.5264`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5507` | F1m：`0.5505`

#### Fold 1

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5826`
- Val BalAcc_maj（附报）：`0.5863`

**Test 试次级**
- Acc_paper：`0.6209`
- BalAcc_maj：`0.6239`
- F1-macro（众数）：`0.6250`
- Rec idle/left/right：`0.6364` / `0.6318` / `0.6036`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6093` | F1m：`0.6103`

#### Fold 2

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.5526`
- Val BalAcc_maj（附报）：`0.5548`

**Test 试次级**
- Acc_paper：`0.5642`
- BalAcc_maj：`0.5697`
- F1-macro（众数）：`0.5674`
- Rec idle/left/right：`0.6364` / `0.6118` / `0.4609`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5651` | F1m：`0.5628`

#### Fold 3

- stopped_epoch：`72` | best_epoch：`52`
- Val Acc_paper（早停）：`0.6093`
- Val BalAcc_maj（附报）：`0.6119`

**Test 试次级**
- Acc_paper：`0.6230`
- BalAcc_maj：`0.6264`
- F1-macro（众数）：`0.6266`
- Rec idle/left/right：`0.5882` / `0.6036` / `0.6873`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6131` | F1m：`0.6131`

#### Fold 4

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.5278`
- Val BalAcc_maj（附报）：`0.5300`

**Test 试次级**
- Acc_paper：`0.5790`
- BalAcc_maj：`0.5817`
- F1-macro（众数）：`0.5814`
- Rec idle/left/right：`0.6150` / `0.5870` / `0.5430`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5711` | F1m：`0.5708`

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
  "t0_weight_alpha": 0.6,
  "t0_filter_max": 0.0
}
```

- 结束：`2026-08-23T13:11:28`
