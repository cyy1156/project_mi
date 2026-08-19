# 被试独立五折实验记录（20260819_204409 / shallow_L0_lib_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-19T20:44:09`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-impl-audit subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_L0_lib` | ShallowFBCSPNet · braindecode 官方 · 方案18 L0
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\MI\code\train_lab\out\5070_shallow_impl_audit_accpaper\shallow_L0_lib_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260819_204409`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-impl-audit subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6854 ± 0.0358`
- Test Acc_paper：`0.6981 ± 0.0353`
- Test BalAcc_maj：`0.6826 ± 0.0239`
- Test 窗级 BalAcc（附报）：`0.6531 ± 0.0170`

### Task 分折明细

#### Fold 0

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.6781`
- Val BalAcc_maj（附报）：`0.6756`

**Test 试次级**
- Acc_paper：`0.6918`
- BalAcc_maj：`0.6895`
- Acc_majority：`0.6918`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6570` | F1：`0.7218` | Acc：`0.6591`

#### Fold 1

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.7367`
- Val BalAcc_maj（附报）：`0.6894`

**Test 试次级**
- Acc_paper：`0.7315`
- BalAcc_maj：`0.7145`
- Acc_majority：`0.7315`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6775` | F1：`0.7592` | Acc：`0.6934`

#### Fold 2

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.6444`
- Val BalAcc_maj（附报）：`0.6481`

**Test 试次级**
- Acc_paper：`0.6679`
- BalAcc_maj：`0.6616`
- Acc_majority：`0.6679`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6326` | F1：`0.7054` | Acc：`0.6382`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.7159`
- Val BalAcc_maj（附报）：`0.6964`

**Test 试次级**
- Acc_paper：`0.7452`
- BalAcc_maj：`0.6980`
- Acc_majority：`0.7452`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6630` | F1：`0.7834` | Acc：`0.7070`

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6519`
- Val BalAcc_maj（附报）：`0.6097`

**Test 试次级**
- Acc_paper：`0.6540`
- BalAcc_maj：`0.6492`
- Acc_majority：`0.6540`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6354` | F1：`0.7056` | Acc：`0.6396`

### Three
- Val Acc_paper：`0.5217 ± 0.0292`
- Test Acc_paper：`0.5444 ± 0.0263`
- Test BalAcc_maj：`0.5591 ± 0.0273`
- Test 窗级 BalAcc（附报）：`0.5324 ± 0.0218`

### Three 分折明细

#### Fold 0

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.5222`
- Val BalAcc_maj（附报）：`0.5381`

**Test 试次级**
- Acc_paper：`0.5236`
- BalAcc_maj：`0.5439`
- F1-macro（众数）：`0.5436`
- Rec idle/left/right：`0.5609` / `0.5645` / `0.5064`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5163` | F1m：`0.5162`

#### Fold 1

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5352`
- Val BalAcc_maj（附报）：`0.5496`

**Test 试次级**
- Acc_paper：`0.5721`
- BalAcc_maj：`0.5861`
- F1-macro（众数）：`0.5869`
- Rec idle/left/right：`0.5909` / `0.6082` / `0.5591`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5587` | F1m：`0.5589`

#### Fold 2

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5085`
- Val BalAcc_maj（附报）：`0.5215`

**Test 试次级**
- Acc_paper：`0.5115`
- BalAcc_maj：`0.5221`
- F1-macro（众数）：`0.5179`
- Rec idle/left/right：`0.6436` / `0.5245` / `0.3982`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5043` | F1m：`0.5008`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5656`
- Val BalAcc_maj（附报）：`0.5807`

**Test 试次级**
- Acc_paper：`0.5779`
- BalAcc_maj：`0.5952`
- F1-macro（众数）：`0.5945`
- Rec idle/left/right：`0.5345` / `0.5809` / `0.6700`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5568` | F1m：`0.5560`

#### Fold 4

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.4770`
- Val BalAcc_maj（附报）：`0.4915`

**Test 试次级**
- Acc_paper：`0.5367`
- BalAcc_maj：`0.5483`
- F1-macro（众数）：`0.5482`
- Rec idle/left/right：`0.5760` / `0.5260` / `0.5430`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5261` | F1m：`0.5261`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100",
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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-shallow-impl-audit subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070",
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

- 结束：`2026-08-19T22:19:51`
