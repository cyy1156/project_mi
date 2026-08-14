# ShallowFBCSPNet · 无 braindecode 完整复现（2s / hop100 Acc_paper）

> 本页为总览；完整代码在：
>
> - [`shallow_fbcsp_net.md`](./shallow_fbcsp_net.md) — 模型（**先写各模块，再组装**）  
> - [`train_shallow_hop100_accpaper.md`](./train_shallow_hop100_accpaper.md) — 五折滑窗训练  

---

## 0. 为什么要「先模块、后拼接」

原先照搬 braindecode，用 `nn.Sequential` + `add_module` 一路往里塞，有两个问题：

1. **读不清边界**：时间卷积、空间卷积、池化、分类头都挤在同一个 `__init__` 里。  
2. **不好改**：你后面要加注意力 / 换一层时，只能在长函数里抠。

更合适的写法（已写进 `shallow_fbcsp_net.md`）：

```text
① 每个积木单独成类：Ensure4d / DimShuffle / ConvTime / ConvSpat / …
② ShallowFBCSPNet 只负责实例化 + 接线
③ forward_features() 拼特征，forward() = 分类头(特征)
```

以后加模块：新建一个 `nn.Module`，再在 `forward_features` 里插入一行。

---

## 1. 协议（与正式臂一致）

| 项 | 取值 |
|----|------|
| 数据 | `code/preprocess_lab/out/bci2a_2s_hop100/` |
| 窗 | 2s @ 250Hz → `(N, 8, 500)`，hop=100ms |
| 损失 | 窗级 CE + train batch balance |
| 早停 / 主报 | **Acc_paper**（试次内正确窗占比 > 0.5） |
| 划分 | 被试独立五折，`seed=42` |
| 优化 | Adam `1e-4`，`wd=1e-4`，`drop=0.5`，`patience=18` |

正式入口：`code/train_lab/src/step/baselines_2s_hop100_accpaper/baseline_shallow.py`。

---

## 2. 怎么跑

把两个 md 里的代码块另存为同名 `.py` 后：

```bat
cd /d D:\360MoveData\Users\ckgxnn\Desktop\MI\self_model
conda activate bci_cyy
python train_shallow_hop100_accpaper.py --max-folds 1 --max-epochs 2 --patience 2
python train_shallow_hop100_accpaper.py
```

---

## 3. 数据流

```text
(B,8,500)
 → Ensure4d / DimShuffle → (B,1,500,8)
 → ConvTime(k=25) → ConvSpat(C→40) → BN → Square
 → AvgPool(75,stride=15) → SafeLog → Dropout
 → FinalClassifier → (B, n_outputs)
```

完整模块定义与组装代码见 [`shallow_fbcsp_net.md`](./shallow_fbcsp_net.md)。

> 说明：示例为「模块化可读版」。`state_dict` 键名会变成 `conv_time.conv.weight` 等，**与 braindecode 检查点键名不完全相同**；结构与默认超参仍对齐，可从零训练复现滑窗协议。若要对齐官方权重键名，再在组装时用扁平 `nn.Conv2d` 命名即可。

---

## 4. 训练脚本

[`train_shallow_hop100_accpaper.md`](./train_shallow_hop100_accpaper.md)

---

## 5. 后续加模块

在 `forward_features` 中插入你的模块，或包一层 `ShallowWithExtra`，保持：

```python
def build_model(n_chans, n_times, n_outputs, drop_prob) -> nn.Module: ...
```
