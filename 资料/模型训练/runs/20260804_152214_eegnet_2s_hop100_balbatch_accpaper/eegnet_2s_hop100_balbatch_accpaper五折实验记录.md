# 被试独立五折实验记录（20260804_152214 / eegnet_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T15:22:14`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-T-only` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`eegnet` | from baselines_2s_hop100/eegnet input_kind=time
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\eegnet_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_152214`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 2, 'patience': 2, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-T-only', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'bci2a_T_only': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6303 ± 0.0000`
- Test Acc_paper：`0.6086 ± 0.0000`
- Test BalAcc_maj：`0.5635 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.5305 ± 0.0000`

### Task 分折明细

#### Fold 0

- stopped_epoch：`2` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6303`
- Val BalAcc_maj（附报）：`0.6211`

**Test 试次级**
- Acc_paper：`0.6086`
- BalAcc_maj：`0.5635`
- Acc_majority：`0.6086`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5305` | F1：`0.6440` | Acc：`0.5533`

### Three
- （本次跳过）

### 共用超参
```json
{
  "data_tag": "bci2a_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 2,
  "patience": 2,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-balbatch-accpaper-T-only",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "bci2a_T_only": true
}
```

- 结束：`2026-08-04T15:22:39`
