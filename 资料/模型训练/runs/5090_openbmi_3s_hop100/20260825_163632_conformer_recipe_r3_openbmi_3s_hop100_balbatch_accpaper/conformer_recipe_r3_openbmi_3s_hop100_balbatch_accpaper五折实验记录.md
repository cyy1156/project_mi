# 被试独立五折实验记录（20260825_163632 / conformer_recipe_r3_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-25T16:36:32`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme26` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`conformer_recipe_r3` | Conformer · scheme26 R3 recipe
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_ens_recipe_3s_hop100_accpaper\conformer_recipe_r3_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260825_163632`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0005, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme26', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0, 'optimizer': 'adamw', 'label_smoothing': 0.1, 'grad_clip_norm': 1.0, 'warmup_epochs': 3, 'lr_min': 1e-06, 'early_stop_tie_eps': 0.0005, 'use_swa': False, 'swa_start_frac': 0.5}`

---
## 最终结论（主报 Acc_paper）

### Task
- （本次跳过 · three-only）

### Three
- Val Acc_paper：`0.5699 ± 0.0279`
- Test Acc_paper：`0.5851 ± 0.0294`
- Test BalAcc_maj：`0.5903 ± 0.0297`
- Test 窗级 BalAcc（附报）：`0.5804 ± 0.0262`

### Three 分折明细

#### Fold 0

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.5859`
- Val BalAcc_maj（附报）：`0.5919`

**Test 试次级**
- Acc_paper：`0.5476`
- BalAcc_maj：`0.5527`
- F1-macro（众数）：`0.5529`
- Rec idle/left/right：`0.5036` / `0.5736` / `0.5809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5489` | F1m：`0.5490`

#### Fold 1

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.5778`
- Val BalAcc_maj（附报）：`0.5844`

**Test 试次级**
- Acc_paper：`0.6212`
- BalAcc_maj：`0.6279`
- F1-macro（众数）：`0.6280`
- Rec idle/left/right：`0.6218` / `0.7373` / `0.5245`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6102` | F1m：`0.6102`

#### Fold 2

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.5489`
- Val BalAcc_maj（附报）：`0.5526`

**Test 试次级**
- Acc_paper：`0.5555`
- BalAcc_maj：`0.5606`
- F1-macro（众数）：`0.5583`
- Rec idle/left/right：`0.6064` / `0.6255` / `0.4500`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5519` | F1m：`0.5495`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6081`
- Val BalAcc_maj（附报）：`0.6126`

**Test 试次级**
- Acc_paper：`0.6121`
- BalAcc_maj：`0.6164`
- F1-macro（众数）：`0.6140`
- Rec idle/left/right：`0.5864` / `0.5264` / `0.7364`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6073` | F1m：`0.6048`

#### Fold 4

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.5289`
- Val BalAcc_maj（附报）：`0.5356`

**Test 试次级**
- Acc_paper：`0.5890`
- BalAcc_maj：`0.5940`
- F1-macro（众数）：`0.5938`
- Rec idle/left/right：`0.6080` / `0.6190` / `0.5550`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5838` | F1m：`0.5836`

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
  "weight_decay": 0.0005,
  "drop_prob": 0.5,
  "protocol": "3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme26",
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
  "t0_filter_max": 0.0,
  "optimizer": "adamw",
  "label_smoothing": 0.1,
  "grad_clip_norm": 1.0,
  "warmup_epochs": 3,
  "lr_min": 1e-06,
  "early_stop_tie_eps": 0.0005,
  "use_swa": false,
  "swa_start_frac": 0.5
}
```

- 结束：`2026-08-25T17:45:55`
