# 被试独立五折实验记录（20260730_203625 / gcbnet）

- 开始：`2026-07-30T20:36:25`
- device：`cpu`
- data：`D:\360MoveData\Users\ckgxnn\Desktop\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`）
- model：`gcbnet`（单脚本；无 registry）
- 输入：bandpower 立方体 `(1719, 8, 2)`（非时域 500）
- 结构：GCBNet(k=2, layers=[128], dropout=shared drop_prob)；8 导联偶数
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\360MoveData\Users\ckgxnn\Desktop\MI\code\train_lab\out\baseline\gcbnet\bci2a_2s\run_20260730_203625`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8086 ± 0.0029`
- Test F1：`0.8085 ± 0.0058`
- Test Acc：`0.6786 ± 0.0082`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.3833 ± 0.0738`
- Test F1-macro：`0.3859 ± 0.0775`
- Test Acc：`0.4279 ± 0.0537`

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

- 结束：`2026-07-30T20:38:20`
