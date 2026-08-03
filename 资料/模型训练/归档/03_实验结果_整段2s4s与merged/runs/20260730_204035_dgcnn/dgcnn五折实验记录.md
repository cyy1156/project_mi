# 被试独立五折实验记录（20260730_204035 / dgcnn）

- 开始：`2026-07-30T20:40:35`
- device：`cpu`
- data：`D:\360MoveData\Users\ckgxnn\Desktop\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`）
- model：`dgcnn`（单脚本；无 registry）
- 输入：bandpower 立方体 `(1719, 8, 2)`（非时域 500）
- 结构：DGCNN(k=2, layers=[128], dropout=shared drop_prob)
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\360MoveData\Users\ckgxnn\Desktop\MI\code\train_lab\out\baseline\dgcnn\bci2a_2s\run_20260730_204035`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8092 ± 0.0039`
- Test F1：`0.8085 ± 0.0058`
- Test Acc：`0.6786 ± 0.0082`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.3847 ± 0.0673`
- Test F1-macro：`0.3631 ± 0.0977`
- Test Acc：`0.4065 ± 0.0585`

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

- 结束：`2026-07-30T20:42:01`
