# 被试独立五折实验记录（20260801_173124 / shallow_wce2_balbatch_balacc）

- 开始：`2026-08-01T17:31:24`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`shallow_wce2_balbatch_balacc`（新脚本 baseline_shallow_balbatch.py；不改 baseline_shallow.py）
- 臂：`B2` — 加权CE w0=2,w1=1 + train batch balance 1:1 + BalAcc早停
- Task 采样：train WeightedRandomSampler（类逆频，约 1:1）；val/test 真实比例
- 早停：Balanced Acc；仅跑 Task（无 Three）
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1 仅附报）
- 结构：ShallowFBCSPNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\shallow_wce2_balbatch_balacc\merged_2s\run_20260801_173124`

---
## 最终结论

### Task（静息/任务）— 臂 B2
- Val BalAcc（选模）：`0.5716 ± 0.0238`
- Test Spec：`0.6775 ± 0.1435`
- Test Rec：`0.4244 ± 0.1247`
- Test BalAcc：`0.5510 ± 0.0293`
- Test F1（附报）：`0.5345 ± 0.0943`
- Test Acc：`0.4935 ± 0.0546`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优（Balanced Acc）并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`68`
- 验证最优轮次（best_epoch）：`50`
- Val 选模分数（Balanced Acc）：`0.5980`
- Val F1（最优 checkpoint 时，附报）：`0.5344`
- Val loss（最优时）：`0.7555`

**Test（overall）**
- Accuracy：`0.4777`
- Recall：`0.3807`
- Specificity：`0.7172`
- Precision：`0.7688`
- F1：`0.5093`
- Balanced Acc：`0.5490`
- 混淆矩阵：TP=`1922` TN=`1466` FP=`578` FN=`3126`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5682` F1=`0.5799` BalAcc=`0.6357`
- `stieger_only`：Acc=`0.4724` F1=`0.5053` BalAcc=`0.5435`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5773`
- Val F1（最优 checkpoint 时，附报）：`0.6991`
- Val loss（最优时）：`0.6725`

**Test（overall）**
- Accuracy：`0.4147`
- Recall：`0.2647`
- Specificity：`0.7971`
- Precision：`0.7688`
- F1：`0.3939`
- Balanced Acc：`0.5309`
- 混淆矩阵：TP=`1154` TN=`1363` FP=`347` FN=`3205`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4296` F1=`0.2884` BalAcc=`0.5700`
- `stieger_only`：Acc=`0.4137` F1=`0.3999` BalAcc=`0.5270`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.5346`
- Val F1（最优 checkpoint 时，附报）：`0.2796`
- Val loss（最优时）：`0.7843`

**Test（overall）**
- Accuracy：`0.4850`
- Recall：`0.3615`
- Specificity：`0.8480`
- Precision：`0.8748`
- F1：`0.5116`
- Balanced Acc：`0.6047`
- 混淆矩阵：TP=`1719` TN=`1372` FP=`246` FN=`3036`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6256` F1=`0.6756` BalAcc=`0.6495`
- `stieger_only`：Acc=`0.4758` F1=`0.4998` BalAcc=`0.6039`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5549`
- Val F1（最优 checkpoint 时，附报）：`0.4363`
- Val loss（最优时）：`0.7781`

**Test（overall）**
- Accuracy：`0.5059`
- Recall：`0.4839`
- Specificity：`0.5552`
- Precision：`0.7095`
- F1：`0.5754`
- Balanced Acc：`0.5196`
- 混淆矩阵：TP=`2743` TN=`1402` FP=`1123` FN=`2926`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3967` F1=`0.3934` BalAcc=`0.4555`
- `stieger_only`：Acc=`0.5110` F1=`0.5826` BalAcc=`0.5224`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.5929`
- Val F1（最优 checkpoint 时，附报）：`0.6634`
- Val loss（最优时）：`0.6808`

**Test（overall）**
- Accuracy：`0.5842`
- Recall：`0.6313`
- Specificity：`0.4700`
- Precision：`0.7426`
- F1：`0.6824`
- Balanced Acc：`0.5507`
- 混淆矩阵：TP=`3721` TN=`1144` FP=`1290` FN=`2173`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3533` F1=`0.2059` BalAcc=`0.5015`
- `stieger_only`：Acc=`0.5889` F1=`0.6885` BalAcc=`0.5514`

### 实验超参
```json
{
  "data_tag": "merged_2s",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 18,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "arm": "B2",
  "task_sampler": "balanced_invfreq",
  "task_weight_mode": "fixed",
  "task_w0": 1.5,
  "task_w1": 1.0,
  "task_early_stop": "balanced_accuracy",
  "task_only": true
}
```

- 结束：`2026-08-01T17:38:12`
