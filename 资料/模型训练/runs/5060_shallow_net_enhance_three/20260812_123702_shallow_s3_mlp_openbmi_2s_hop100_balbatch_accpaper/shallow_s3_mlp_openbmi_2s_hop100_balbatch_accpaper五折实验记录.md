# 被试独立五折实验记录（20260812_123702 / shallow_s3_mlp_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-12T12:37:02`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_s3_mlp` | ShallowFBCSPNet S3_mlp | AdaptiveAvgPool + MLP(40→64→C)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_net_enhance_three_accpaper\shallow_s3_mlp_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260812_123702`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 1024, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 4, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6661 ± 0.0345`
- Test Acc_paper：`0.6784 ± 0.0393`
- Test BalAcc_maj：`0.6574 ± 0.0215`
- Test 窗级 BalAcc（附报）：`0.6382 ± 0.0170`

### Task 分折明细

#### Fold 0

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.6474`
- Val BalAcc_maj（附报）：`0.6308`

**Test 试次级**
- Acc_paper：`0.6961`
- BalAcc_maj：`0.6620`
- Acc_majority：`0.6961`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6444` | F1：`0.7493` | Acc：`0.6735`

#### Fold 1

- stopped_epoch：`86` | best_epoch：`66`
- Val Acc_paper（早停）：`0.7215`
- Val BalAcc_maj（附报）：`0.6719`

**Test 试次级**
- Acc_paper：`0.7079`
- BalAcc_maj：`0.6857`
- Acc_majority：`0.7079`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6602` | F1：`0.7499` | Acc：`0.6800`

#### Fold 2

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.6330`
- Val BalAcc_maj（附报）：`0.6233`

**Test 试次级**
- Acc_paper：`0.6203`
- BalAcc_maj：`0.6302`
- Acc_majority：`0.6203`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6112` | F1：`0.6636` | Acc：`0.6032`

#### Fold 3

- stopped_epoch：`66` | best_epoch：`46`
- Val Acc_paper（早停）：`0.6911`
- Val BalAcc_maj（附报）：`0.6786`

**Test 试次级**
- Acc_paper：`0.7233`
- BalAcc_maj：`0.6736`
- Acc_majority：`0.7233`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6474` | F1：`0.7755` | Acc：`0.6950`

#### Fold 4

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.6378`
- Val BalAcc_maj（附报）：`0.5914`

**Test 试次级**
- Acc_paper：`0.6447`
- BalAcc_maj：`0.6352`
- Acc_majority：`0.6447`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6278` | F1：`0.7062` | Acc：`0.6368`

### Three
- Val Acc_paper：`0.5088 ± 0.0351`
- Test Acc_paper：`0.5296 ± 0.0233`
- Test BalAcc_maj：`0.5432 ± 0.0240`
- Test 窗级 BalAcc（附报）：`0.5205 ± 0.0205`

### Three 分折明细

#### Fold 0

- stopped_epoch：`61` | best_epoch：`41`
- Val Acc_paper（早停）：`0.5056`
- Val BalAcc_maj（附报）：`0.5144`

**Test 试次级**
- Acc_paper：`0.5224`
- BalAcc_maj：`0.5379`
- F1-macro（众数）：`0.5375`
- Rec idle/left/right：`0.5709` / `0.5427` / `0.5000`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5146` | F1m：`0.5143`

#### Fold 1

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5319`
- Val BalAcc_maj（附报）：`0.5481`

**Test 试次级**
- Acc_paper：`0.5497`
- BalAcc_maj：`0.5609`
- F1-macro（众数）：`0.5593`
- Rec idle/left/right：`0.4991` / `0.6773` / `0.5064`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5389` | F1m：`0.5368`

#### Fold 2

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.4978`
- Val BalAcc_maj（附报）：`0.5119`

**Test 试次级**
- Acc_paper：`0.4982`
- BalAcc_maj：`0.5100`
- F1-macro（众数）：`0.5087`
- Rec idle/left/right：`0.5918` / `0.5000` / `0.4382`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4881` | F1m：`0.4872`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5567`
- Val BalAcc_maj（附报）：`0.5681`

**Test 试次级**
- Acc_paper：`0.5624`
- BalAcc_maj：`0.5782`
- F1-macro（众数）：`0.5755`
- Rec idle/left/right：`0.5309` / `0.5073` / `0.6964`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5459` | F1m：`0.5438`

#### Fold 4

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.4522`
- Val BalAcc_maj（附报）：`0.4674`

**Test 试次级**
- Acc_paper：`0.5153`
- BalAcc_maj：`0.5290`
- F1-macro（众数）：`0.5291`
- Rec idle/left/right：`0.5310` / `0.5300` / `0.5260`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5148` | F1m：`0.5148`

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
  "batch_eval": 1024,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "prefetch_factor": 4,
  "non_blocking": true,
  "torch_num_threads": 6,
  "cudnn_benchmark": true,
  "deterministic": false,
  "use_amp": true
}
```

- 结束：`2026-08-12T15:29:00`
