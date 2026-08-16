# 被试独立五折实验记录（20260815_151720 / shallow_cbam_a2_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-15T15:17:20`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-cbam-self_model subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_cbam_a2` | A2 split: channel@Time + temporal-spatial@BN (primary) | attn=split
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_cbam_self_model_accpaper\shallow_cbam_a2_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260815_151720`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-cbam-self_model subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7000 ± 0.0336`
- Test Acc_paper：`0.7089 ± 0.0342`
- Test BalAcc_maj：`0.6876 ± 0.0228`
- Test 窗级 BalAcc（附报）：`0.6601 ± 0.0191`

### Task 分折明细

#### Fold 0

- stopped_epoch：`63` | best_epoch：`43`
- Val Acc_paper（早停）：`0.7070`
- Val BalAcc_maj（附报）：`0.6831`

**Test 试次级**
- Acc_paper：`0.7194`
- BalAcc_maj：`0.6952`
- Acc_majority：`0.7194`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6648` | F1：`0.7579` | Acc：`0.6876`

#### Fold 1

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.7396`
- Val BalAcc_maj（附报）：`0.6914`

**Test 试次级**
- Acc_paper：`0.7500`
- BalAcc_maj：`0.7202`
- Acc_majority：`0.7500`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6856` | F1：`0.7816` | Acc：`0.7135`

#### Fold 2

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.6741`
- Val BalAcc_maj（附报）：`0.6619`

**Test 试次级**
- Acc_paper：`0.6652`
- BalAcc_maj：`0.6602`
- Acc_majority：`0.6652`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6328` | F1：`0.7076` | Acc：`0.6398`

#### Fold 3

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.7293`
- Val BalAcc_maj（附报）：`0.7150`

**Test 试次级**
- Acc_paper：`0.7373`
- BalAcc_maj：`0.6991`
- Acc_majority：`0.7373`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6729` | F1：`0.7795` | Acc：`0.7073`

#### Fold 4

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6500`
- Val BalAcc_maj（附报）：`0.6111`

**Test 试次级**
- Acc_paper：`0.6727`
- BalAcc_maj：`0.6630`
- Acc_majority：`0.6727`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6445` | F1：`0.7242` | Acc：`0.6557`

### Three
- Val Acc_paper：`0.5259 ± 0.0345`
- Test Acc_paper：`0.5488 ± 0.0282`
- Test BalAcc_maj：`0.5653 ± 0.0275`
- Test 窗级 BalAcc（附报）：`0.5357 ± 0.0245`

### Three 分折明细

#### Fold 0

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.5415`
- Val BalAcc_maj（附报）：`0.5600`

**Test 试次级**
- Acc_paper：`0.5345`
- BalAcc_maj：`0.5555`
- F1-macro（众数）：`0.5555`
- Rec idle/left/right：`0.5664` / `0.5400` / `0.5600`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5246` | F1m：`0.5247`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5363`
- Val BalAcc_maj（附报）：`0.5530`

**Test 试次级**
- Acc_paper：`0.5785`
- BalAcc_maj：`0.5942`
- F1-macro（众数）：`0.5943`
- Rec idle/left/right：`0.6209` / `0.6218` / `0.5400`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5605` | F1m：`0.5602`

#### Fold 2

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.5107`
- Val BalAcc_maj（附报）：`0.5289`

**Test 试次级**
- Acc_paper：`0.5055`
- BalAcc_maj：`0.5218`
- F1-macro（众数）：`0.5133`
- Rec idle/left/right：`0.6282` / `0.6000` / `0.3373`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4951` | F1m：`0.4885`

#### Fold 3

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.5719`
- Val BalAcc_maj（附报）：`0.5826`

**Test 试次级**
- Acc_paper：`0.5803`
- BalAcc_maj：`0.5955`
- F1-macro（众数）：`0.5934`
- Rec idle/left/right：`0.5073` / `0.5691` / `0.7100`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5607` | F1m：`0.5591`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4689`
- Val BalAcc_maj（附报）：`0.4881`

**Test 试次级**
- Acc_paper：`0.5450`
- BalAcc_maj：`0.5593`
- F1-macro（众数）：`0.5586`
- Rec idle/left/right：`0.6170` / `0.5520` / `0.5090`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5375` | F1m：`0.5373`

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

- 结束：`2026-08-15T19:03:06`
