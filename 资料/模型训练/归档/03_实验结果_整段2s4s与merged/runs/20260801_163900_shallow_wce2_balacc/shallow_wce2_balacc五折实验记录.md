# 被试独立五折实验记录（20260801_163900 / shallow_wce2_balacc）

- 开始：`2026-08-01T16:39:00`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`shallow_wce2_balacc`（单脚本；无 registry）
- Task 目标：加权CE w0=2.2, w1=1.0, mode=fixed；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1 仅附报）
- 结构：ShallowFBCSPNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\shallow_wce2_balacc\merged_2s\run_20260801_163900`

---
