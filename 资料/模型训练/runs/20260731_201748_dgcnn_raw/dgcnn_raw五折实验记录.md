# 被试独立五折实验记录（20260731_201748 / dgcnn_raw）

- 开始：`2026-07-31T20:17:48`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`dgcnn_raw`（TemporalEncoder + DGCNN）
- 输入：raw `(36056, 8, 500)` → Encoder → 节点特征 D=64（非 bandpower）
- 结构：DGCNNRaw(k=2, layers=[128], in_channels=64, relu_is=1)
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\dgcnn_raw\merged_2s\run_20260731_201748`

---
