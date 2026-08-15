# 被试独立五折实验记录（20260810_222313 / dgcnn_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-10T22:23:13`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dgcnn` | DGCNN(k=2, layers=[128]) + 2s bandpower
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\dgcnn_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260810_222313`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6326 ± 0.0289`
- Test Acc_paper：`0.6344 ± 0.0318`
- Test BalAcc_maj：`0.5805 ± 0.0277`
- Test 窗级 BalAcc（附报）：`0.5781 ± 0.0248`

### Task 分折明细

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6626`
- Val BalAcc_maj（附报）：`0.5803`

**Test 试次级**
- Acc_paper：`0.6564`
- BalAcc_maj：`0.5593`
- Acc_majority：`0.6564`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5587` | F1：`0.7617` | Acc：`0.6512`

#### Fold 1

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6548`
- Val BalAcc_maj（附报）：`0.5900`

**Test 试次级**
- Acc_paper：`0.6782`
- BalAcc_maj：`0.6302`
- Acc_majority：`0.6782`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6212` | F1：`0.7500` | Acc：`0.6652`

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5819`
- Val BalAcc_maj（附报）：`0.5344`

**Test 试次级**
- Acc_paper：`0.6312`
- BalAcc_maj：`0.5659`
- Acc_majority：`0.6312`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5663` | F1：`0.7297` | Acc：`0.6284`

#### Fold 3

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6215`
- Val BalAcc_maj（附报）：`0.5908`

**Test 试次级**
- Acc_paper：`0.5845`
- BalAcc_maj：`0.5911`
- Acc_majority：`0.5845`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5899` | F1：`0.6503` | Acc：`0.5858`

#### Fold 4

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.6422`
- Val BalAcc_maj（附报）：`0.5506`

**Test 试次级**
- Acc_paper：`0.6217`
- BalAcc_maj：`0.5560`
- Acc_majority：`0.6217`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5543` | F1：`0.7227` | Acc：`0.6182`

### Three
- Val Acc_paper：`0.4504 ± 0.0310`
- Test Acc_paper：`0.4668 ± 0.0273`
- Test BalAcc_maj：`0.4766 ± 0.0260`
- Test 窗级 BalAcc（附报）：`0.4656 ± 0.0228`

### Three 分折明细

#### Fold 0

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.4485`
- Val BalAcc_maj（附报）：`0.4581`

**Test 试次级**
- Acc_paper：`0.4261`
- BalAcc_maj：`0.4379`
- F1-macro（众数）：`0.4176`
- Rec idle/left/right：`0.2036` / `0.4782` / `0.6318`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4334` | F1m：`0.4157`

#### Fold 1

- stopped_epoch：`72` | best_epoch：`52`
- Val Acc_paper（早停）：`0.4689`
- Val BalAcc_maj（附报）：`0.4833`

**Test 试次级**
- Acc_paper：`0.4976`
- BalAcc_maj：`0.5073`
- F1-macro（众数）：`0.5018`
- Rec idle/left/right：`0.3773` / `0.6409` / `0.5036`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4916` | F1m：`0.4866`

#### Fold 2

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.4107`
- Val BalAcc_maj（附报）：`0.4189`

**Test 试次级**
- Acc_paper：`0.4639`
- BalAcc_maj：`0.4727`
- F1-macro（众数）：`0.4519`
- Rec idle/left/right：`0.2418` / `0.7182` / `0.4582`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4544` | F1m：`0.4364`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.4981`
- Val BalAcc_maj（附报）：`0.5059`

**Test 试次级**
- Acc_paper：`0.4958`
- BalAcc_maj：`0.5033`
- F1-macro（众数）：`0.5022`
- Rec idle/left/right：`0.4600` / `0.4764` / `0.5736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4918` | F1m：`0.4907`

#### Fold 4

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.4259`
- Val BalAcc_maj（附报）：`0.4322`

**Test 试次级**
- Acc_paper：`0.4507`
- BalAcc_maj：`0.4620`
- F1-macro（众数）：`0.4414`
- Rec idle/left/right：`0.2200` / `0.6670` / `0.4990`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4571` | F1m：`0.4395`

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

- 结束：`2026-08-10T23:15:18`
