# 被试独立五折实验记录（20260816_184618 / shallow_hier_s0_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`20260816_184618` · chain step **T0_task**
- device：`cuda` · **train_mode=`fast`**（5090 全量五折）
- 训练设备：**NVIDIA RTX 5090**（32GB · conda cyy · PyTorch 2.11+cu128）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（OpenBMI hop100 · blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-hier-loss subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5090` | early_stop=**Acc_paper** | **balbatch** | no_rap
- model：`shallow_hier_s0_openbmi_2s_hop100_balbatch_accpaper` | S0 plain CE · shallow (Task T0)
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_three_hier_loss_accpaper\shallow_hier_s0_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260816_184618`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-hier-loss subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5090', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'keep_fold_packs': False, 'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---

## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6837 ± 0.0348`
- Test Acc_paper：`0.6909 ± 0.0380`
- Test BalAcc_maj：`0.6765 ± 0.0209`
- Test 窗级 BalAcc（附报）：`0.6506 ± 0.0159`

### Task 分折明细

#### Fold 0

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.6796`
- Val BalAcc_maj（附报）：`0.6639`

**Test 试次级**
- Acc_paper：`0.6988`
- BalAcc_maj：`0.6827`
- Acc_majority：`0.6988`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6548` | F1：`0.7374` | Acc：`0.6690`

#### Fold 1

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.7304`
- Val BalAcc_maj（附报）：`0.6844`

**Test 试次级**
- Acc_paper：`0.7161`
- BalAcc_maj：`0.7018`
- Acc_majority：`0.7161`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6684` | F1：`0.7481` | Acc：`0.6818`

#### Fold 2

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.6437`
- Val BalAcc_maj（附报）：`0.6547`

**Test 试次级**
- Acc_paper：`0.6458`
- BalAcc_maj：`0.6541`
- Acc_majority：`0.6458`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6292` | F1：`0.6843` | Acc：`0.6235`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.7159`
- Val BalAcc_maj（附报）：`0.6903`

**Test 试次级**
- Acc_paper：`0.7439`
- BalAcc_maj：`0.6936`
- Acc_majority：`0.7439`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6656` | F1：`0.7889` | Acc：`0.7124`

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6489`
- Val BalAcc_maj（附报）：`0.6136`

**Test 试次级**
- Acc_paper：`0.6500`
- BalAcc_maj：`0.6500`
- Acc_majority：`0.6500`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6350` | F1：`0.6992` | Acc：`0.6353`

