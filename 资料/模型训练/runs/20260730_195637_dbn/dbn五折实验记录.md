# 被试独立五折实验记录（20260730_195637 / dbn）

- 开始：`2026-07-30T19:56:37`
- device：`cpu`
- data：`D:\360MoveData\Users\ckgxnn\Desktop\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`dbn`（单脚本；无 registry）
- 输入：bandpower 立方体 `(36056, 8, 2)`（非时域 500）
- 结构：DBN(hidden 300/400)；监督 forward，无 RBM 预训练；drop_prob 忽略
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\360MoveData\Users\ckgxnn\Desktop\MI\code\train_lab\out\baseline\dbn\merged_2s\run_20260730_195637`

---
