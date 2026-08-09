# 被试独立五折实验记录（20260805_103942 / eegnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-05T10:39:42`
- device：`cuda`
- 训练设备：**NVIDIA RTX 5090**（32GB · sm_120 · conda `cyy` · PyTorch 2.11+cu128）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=1`
- model：`eegnet` | EEGNet F1=8, D=2, F2=16（默认池化）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5090_baseline_openbmi_2s_hop100_accpaper\eegnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260805_103942`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 1, 'patience': 1, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6372 ± 0.0000`
- Test Acc_paper：`0.6548 ± 0.0000`
- Test BalAcc_maj：`0.6600 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.6404 ± 0.0000`

### Task 分折明细

#### Fold 0

- stopped_epoch：`1` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6372`
- Val BalAcc_maj（附报）：`0.6486`

**Test 试次级**
- Acc_paper：`0.6548`
- BalAcc_maj：`0.6600`
- Acc_majority：`0.6548`
- n_trials：`6600`

**Test 窗级（附报）**
- BalAcc：`0.6404` | F1：`0.6977` | Acc：`0.6366`

### Three
- （本次跳过）

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 1,
  "patience": 1,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true
}
```

- 结束：`2026-08-05T10:47:25`
