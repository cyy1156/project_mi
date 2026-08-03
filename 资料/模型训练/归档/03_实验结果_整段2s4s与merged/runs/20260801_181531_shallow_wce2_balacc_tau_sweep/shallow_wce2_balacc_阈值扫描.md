# 阈值扫描（20260801_181531）

- run_dir：`D:\cyy\MI\code\train_lab\out\baseline\shallow_wce2_balacc\merged_2s\run_20260801_161301`
- data：`merged_2s`
- 选 τ 规则：val 上优先 Spec≥0.4 且 Rec≥0.75，取 BalAcc 最大

## 五折汇总（Test）

- **τ=0.5（训练默认）**：Spec `0.4088±0.1626` | Rec `0.7244±0.1427` | BalAcc `0.5666±0.0394`
- **val 选定 τ 后**：τ `0.44±0.06` | Spec `0.3050±0.2070` | Rec `0.7936±0.1965` | BalAcc `0.5493±0.0252`
- val 存在双过关 τ 的折数：`1/5`
- test 曲线上存在双过关 τ 的折数：`1/5`

## 各折

### Fold 0
- val 选 τ=`0.40` (val Spec=0.4678 Rec=0.7645 BalAcc=0.6162)
- test@选中τ：Spec=`0.3567` Rec=`0.7607` BalAcc=`0.5587` F1=`0.7527`
- test@0.5：Spec=`0.5460` Rec=`0.5664` BalAcc=`0.5562`
- val 双过关τ：`True` | test 曲线双过关τ：`False`

### Fold 1
- val 选 τ=`0.55` (val Spec=0.2966 Rec=0.8142 BalAcc=0.5554)
- test@选中τ：Spec=`0.6807` Rec=`0.4233` BalAcc=`0.5520` F1=`0.5467`
- test@0.5：Spec=`0.6029` Rec=`0.5348` BalAcc=`0.5688`
- val 双过关τ：`False` | test 曲线双过关τ：`False`

### Fold 2
- val 选 τ=`0.40` (val Spec=0.3086 Rec=0.8516 BalAcc=0.5801)
- test@选中τ：Spec=`0.2330` Rec=`0.9445` BalAcc=`0.5887` F1=`0.8565`
- test@0.5：Spec=`0.4580` Rec=`0.8221` BalAcc=`0.6400`
- val 双过关τ：`False` | test 曲线双过关τ：`True`

### Fold 3
- val 选 τ=`0.40` (val Spec=0.3911 Rec=0.7732 BalAcc=0.5821)
- test@选中τ：Spec=`0.0958` Rec=`0.9321` BalAcc=`0.5140` F1=`0.7984`
- test@0.5：Spec=`0.1964` Rec=`0.8541` BalAcc=`0.5253`
- val 双过关τ：`False` | test 曲线双过关τ：`False`

### Fold 4
- val 选 τ=`0.45` (val Spec=0.3198 Rec=0.8384 BalAcc=0.5791)
- test@选中τ：Spec=`0.1586` Rec=`0.9074` BalAcc=`0.5330` F1=`0.8048`
- test@0.5：Spec=`0.2408` Rec=`0.8448` BalAcc=`0.5428`
- val 双过关τ：`False` | test 曲线双过关τ：`False`

