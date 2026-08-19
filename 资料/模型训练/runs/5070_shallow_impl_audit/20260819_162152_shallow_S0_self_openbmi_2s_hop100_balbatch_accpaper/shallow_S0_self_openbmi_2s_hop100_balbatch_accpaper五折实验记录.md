# 被试独立五折实验记录（20260819_162152 / shallow_S0_self_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-19T16:21:52`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-impl-audit subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_S0_self` | ShallowFBCSPNet · self_model/shallowfbcsp · attn=None · 方案18 S0
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\MI\code\train_lab\out\5070_shallow_impl_audit_accpaper\shallow_S0_self_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260819_162152`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-impl-audit subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6851 ± 0.0359`
- Test Acc_paper：`0.6979 ± 0.0356`
- Test BalAcc_maj：`0.6825 ± 0.0240`
- Test 窗级 BalAcc（附报）：`0.6531 ± 0.0170`

### Task 分折明细

#### Fold 0

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.6781`
- Val BalAcc_maj（附报）：`0.6756`

**Test 试次级**
- Acc_paper：`0.6912`
- BalAcc_maj：`0.6891`
- Acc_majority：`0.6912`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6571` | F1：`0.7218` | Acc：`0.6591`

#### Fold 1

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.7359`
- Val BalAcc_maj（附报）：`0.6889`

**Test 试次级**
- Acc_paper：`0.7315`
- BalAcc_maj：`0.7145`
- Acc_majority：`0.7315`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6776` | F1：`0.7592` | Acc：`0.6933`

#### Fold 2

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.6437`
- Val BalAcc_maj（附报）：`0.6475`

**Test 试次级**
- Acc_paper：`0.6679`
- BalAcc_maj：`0.6618`
- Acc_majority：`0.6679`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6326` | F1：`0.7055` | Acc：`0.6383`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.7163`
- Val BalAcc_maj（附报）：`0.6967`

**Test 试次级**
- Acc_paper：`0.7455`
- BalAcc_maj：`0.6982`
- Acc_majority：`0.7455`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6631` | F1：`0.7834` | Acc：`0.7070`

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6515`
- Val BalAcc_maj（附报）：`0.6094`

**Test 试次级**
- Acc_paper：`0.6533`
- BalAcc_maj：`0.6487`
- Acc_majority：`0.6533`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6354` | F1：`0.7054` | Acc：`0.6395`

### Three
- Val Acc_paper：`0.5224 ± 0.0298`
- Test Acc_paper：`0.5427 ± 0.0243`
- Test BalAcc_maj：`0.5574 ± 0.0268`
- Test 窗级 BalAcc（附报）：`0.5313 ± 0.0214`

### Three 分折明细

#### Fold 0

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.5215`
- Val BalAcc_maj（附报）：`0.5370`

**Test 试次级**
- Acc_paper：`0.5230`
- BalAcc_maj：`0.5385`
- F1-macro（众数）：`0.5381`
- Rec idle/left/right：`0.5745` / `0.5364` / `0.5045`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5137` | F1m：`0.5135`

#### Fold 1

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.5404`
- Val BalAcc_maj（附报）：`0.5607`

**Test 试次级**
- Acc_paper：`0.5642`
- BalAcc_maj：`0.5821`
- F1-macro（众数）：`0.5828`
- Rec idle/left/right：`0.5627` / `0.6327` / `0.5509`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5553` | F1m：`0.5552`

#### Fold 2

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5085`
- Val BalAcc_maj（附报）：`0.5219`

**Test 试次级**
- Acc_paper：`0.5118`
- BalAcc_maj：`0.5224`
- F1-macro（众数）：`0.5182`
- Rec idle/left/right：`0.6445` / `0.5245` / `0.3982`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5043` | F1m：`0.5009`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5652`
- Val BalAcc_maj（附报）：`0.5804`

**Test 试次级**
- Acc_paper：`0.5764`
- BalAcc_maj：`0.5939`
- F1-macro（众数）：`0.5934`
- Rec idle/left/right：`0.5355` / `0.5791` / `0.6673`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5569` | F1m：`0.5561`

#### Fold 4

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.4767`
- Val BalAcc_maj（附报）：`0.4922`

**Test 试次级**
- Acc_paper：`0.5380`
- BalAcc_maj：`0.5500`
- F1-macro（众数）：`0.5499`
- Rec idle/left/right：`0.5780` / `0.5290` / `0.5430`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5263` | F1m：`0.5263`

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

- 结束：`2026-08-19T18:49:56`
