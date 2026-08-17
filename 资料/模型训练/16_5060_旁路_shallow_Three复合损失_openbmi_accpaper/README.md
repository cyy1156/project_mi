# 16 · 5060 · Shallow Three 复合损失（旁路）

> **机位：RTX 5060 / ~16GB** · 低内存试探与门控。  
> **全量五折请用** [`../16_5090_旁路_shallow_Three复合损失_openbmi_accpaper/`](../16_5090_旁路_shallow_Three复合损失_openbmi_accpaper/)。

| 项 | 路径 |
|----|------|
| **完整方案** | [`方案.md`](./方案.md) |
| 结果登记 | [`总结/结果登记表.md`](./总结/结果登记表.md) |
| 代码 | `code/train_lab/src/step/5060_three_hier_loss_accpaper/` |
| out | `code/train_lab/out/5060_three_hier_loss_accpaper/` |
| 5090 姊妹 | `code/train_lab/src/step/5090_three_hier_loss_accpaper/` |

**骨干冻结：braindecode ShallowFBCSPNet（小参数 ~1.6e4）**。  
目标冲刺：Task **0.75** / Three **0.60**（靠损失，不加大模型）。  
禁止写入正式表；不与方案 15 CBAM 混跑同一 out。

**内存**：默认 `stream_windows=True` + **只 mmap float16**（不再先开 5GB float32）。  
batch 默认 64/128，`cudnn_benchmark=False`。  

仍被 Event 2004 杀掉时：关 PyCharm，并把「虚拟内存」固定到 **≥32768MB**（本机 commit 上限约 43GB 时，CUDA 很容易把 python 顶到 ~33GB 被杀）。
