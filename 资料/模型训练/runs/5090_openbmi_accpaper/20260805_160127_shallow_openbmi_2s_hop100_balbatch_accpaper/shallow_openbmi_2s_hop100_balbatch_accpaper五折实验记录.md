# 被试独立五折实验记录（20260805_160127 / shallow_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-05T16:01:27`
- device：`cuda`
- 训练设备：**NVIDIA RTX 5090**（32GB · sm_120 · conda `cyy` · PyTorch 2.11+cu128）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_baseline_openbmi_2s_hop100_accpaper\shallow_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260805_160127`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 0.42}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6829 ± 0.0358`
- Test Acc_paper：`0.6982 ± 0.0295`
- Test BalAcc_maj：`0.6777 ± 0.0159`
- Test 窗级 BalAcc（附报）：`0.6523 ± 0.0130`

### Task 分折明细

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6759`
- Val BalAcc_maj（附报）：`0.6686`

**Test 试次级**
- Acc_paper：`0.6879`
- BalAcc_maj：`0.6786`
- Acc_majority：`0.6879`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6505` | F1：`0.7260` | Acc：`0.6593`

#### Fold 1

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.7352`
- Val BalAcc_maj（附报）：`0.6728`

**Test 试次级**
- Acc_paper：`0.7282`
- BalAcc_maj：`0.6925`
- Acc_majority：`0.7282`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6658` | F1：`0.7702` | Acc：`0.6974`

#### Fold 2

- stopped_epoch：`76` | best_epoch：`56`
- Val Acc_paper（早停）：`0.6433`
- Val BalAcc_maj（附报）：`0.6425`

**Test 试次级**
- Acc_paper：`0.6694`
- BalAcc_maj：`0.6607`
- Acc_majority：`0.6694`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6336` | F1：`0.7122` | Acc：`0.6432`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.7119`
- Val BalAcc_maj（附报）：`0.7025`

**Test 试次级**
- Acc_paper：`0.7379`
- BalAcc_maj：`0.6977`
- Acc_majority：`0.7379`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6677` | F1：`0.7789` | Acc：`0.7050`

#### Fold 4

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.6481`
- Val BalAcc_maj（附报）：`0.6175`

**Test 试次级**
- Acc_paper：`0.6677`
- BalAcc_maj：`0.6590`
- Acc_majority：`0.6677`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6436` | F1：`0.7209` | Acc：`0.6531`

### Three
- Val Acc_paper：`0.5221 ± 0.0313`
- Test Acc_paper：`0.5398 ± 0.0251`
- Test BalAcc_maj：`0.5578 ± 0.0254`
- Test 窗级 BalAcc（附报）：`0.5290 ± 0.0210`

### Three 分折明细

#### Fold 0

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5285`
- Val BalAcc_maj（附报）：`0.5463`

**Test 试次级**
- Acc_paper：`0.5264`
- BalAcc_maj：`0.5467`
- F1-macro（众数）：`0.5455`
- Rec idle/left/right：`0.6009` / `0.5582` / `0.4809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5143` | F1m：`0.5137`

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5430`
- Val BalAcc_maj（附报）：`0.5578`

**Test 试次级**
- Acc_paper：`0.5633`
- BalAcc_maj：`0.5785`
- F1-macro（众数）：`0.5790`
- Rec idle/left/right：`0.5918` / `0.6055` / `0.5382`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5502` | F1m：`0.5503`

#### Fold 2

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5022`
- Val BalAcc_maj（附报）：`0.5170`

**Test 试次级**
- Acc_paper：`0.5015`
- BalAcc_maj：`0.5170`
- F1-macro（众数）：`0.5141`
- Rec idle/left/right：`0.5609` / `0.5836` / `0.4064`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4982` | F1m：`0.4959`

#### Fold 3

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.5633`
- Val BalAcc_maj（附报）：`0.5748`

**Test 试次级**
- Acc_paper：`0.5706`
- BalAcc_maj：`0.5894`
- F1-macro（众数）：`0.5870`
- Rec idle/left/right：`0.5455` / `0.5082` / `0.7145`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5531` | F1m：`0.5510`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4737`
- Val BalAcc_maj（附报）：`0.4937`

**Test 试次级**
- Acc_paper：`0.5373`
- BalAcc_maj：`0.5577`
- F1-macro（众数）：`0.5577`
- Rec idle/left/right：`0.5900` / `0.5510` / `0.5320`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5292` | F1m：`0.5292`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "non_blocking": true,
  "use_amp": true,
  "cudnn_benchmark": false,
  "gpu_memory_fraction": 0.42
}
```

- 结束：`2026-08-05T18:13:52`
