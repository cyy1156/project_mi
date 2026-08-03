# 被试独立五折实验记录（20260730_203548 / dbn）

- 开始：`2026-07-30T20:35:48`
- device：`cpu`
- data：`D:\360MoveData\Users\ckgxnn\Desktop\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`）
- model：`dbn`（单脚本；无 registry）
- 输入：bandpower 立方体 `(1719, 8, 2)`（非时域 500）
- 结构：DBN(hidden 300/400)；监督 forward，无 RBM 预训练；drop_prob 忽略
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\360MoveData\Users\ckgxnn\Desktop\MI\code\train_lab\out\baseline\dbn\bci2a_2s\run_20260730_203548`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8086 ± 0.0029`
- Test F1：`0.8085 ± 0.0058`
- Test Acc：`0.6786 ± 0.0082`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.2381 ± 0.0919`
- Test F1-macro：`0.1731 ± 0.0094`
- Test Acc：`0.3365 ± 0.0139`

### 共用超参
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
  "drop_prob": 0.5
}
```

- 结束：`2026-07-30T20:36:15`
