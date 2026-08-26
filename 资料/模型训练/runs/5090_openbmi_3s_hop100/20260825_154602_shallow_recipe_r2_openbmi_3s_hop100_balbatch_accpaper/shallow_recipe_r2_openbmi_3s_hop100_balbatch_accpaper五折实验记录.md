# 被试独立五折实验记录（20260825_154602 / shallow_recipe_r2_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-25T15:46:02`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme26` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_recipe_r2` | ShallowFBCSPNet · scheme26 R2 SWA
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_ens_recipe_3s_hop100_accpaper\shallow_recipe_r2_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260825_154602`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme26', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0, 'optimizer': 'adamw', 'label_smoothing': 0.1, 'grad_clip_norm': 1.0, 'warmup_epochs': 3, 'lr_min': 1e-06, 'early_stop_tie_eps': 0.0005, 'use_swa': True, 'swa_start_frac': 0.5}`

---
## 最终结论（主报 Acc_paper）

### Task
- （本次跳过 · three-only）

### Three
- Val Acc_paper：`0.5713 ± 0.0297`
- Test Acc_paper：`0.5874 ± 0.0266`
- Test BalAcc_maj：`0.5936 ± 0.0265`
- Test 窗级 BalAcc（附报）：`0.5806 ± 0.0232`

### Three 分折明细

#### Fold 0

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5841`
- Val BalAcc_maj（附报）：`0.5907`

**Test 试次级**
- Acc_paper：`0.5597`
- BalAcc_maj：`0.5670`
- F1-macro（众数）：`0.5658`
- Rec idle/left/right：`0.6364` / `0.5109` / `0.5536`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5529` | F1m：`0.5519`

#### Fold 1

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.5796`
- Val BalAcc_maj（附报）：`0.5870`

**Test 试次级**
- Acc_paper：`0.6167`
- BalAcc_maj：`0.6209`
- F1-macro（众数）：`0.6221`
- Rec idle/left/right：`0.6418` / `0.6236` / `0.5973`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6050` | F1m：`0.6060`

#### Fold 2

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.5430`
- Val BalAcc_maj（附报）：`0.5481`

**Test 试次级**
- Acc_paper：`0.5627`
- BalAcc_maj：`0.5667`
- F1-macro（众数）：`0.5641`
- Rec idle/left/right：`0.6300` / `0.6155` / `0.4545`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5611` | F1m：`0.5588`

#### Fold 3

- stopped_epoch：`76` | best_epoch：`56`
- Val Acc_paper（早停）：`0.6159`
- Val BalAcc_maj（附报）：`0.6211`

**Test 试次级**
- Acc_paper：`0.6218`
- BalAcc_maj：`0.6288`
- F1-macro（众数）：`0.6288`
- Rec idle/left/right：`0.5964` / `0.5873` / `0.7027`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6107` | F1m：`0.6106`

#### Fold 4

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.5341`
- Val BalAcc_maj（附报）：`0.5396`

**Test 试次级**
- Acc_paper：`0.5763`
- BalAcc_maj：`0.5847`
- F1-macro（众数）：`0.5826`
- Rec idle/left/right：`0.6910` / `0.5370` / `0.5260`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5734` | F1m：`0.5716`

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
  "use_swa": true,
  "swa_start_frac": 0.5
}
```

- 结束：`2026-08-25T16:33:26`
