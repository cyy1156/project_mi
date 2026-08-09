# 05 · OpenBMI shallow · 前半微调 / 后半评（旁路）

协议对齐 [02](../02_微调_前半训后半评/)，差异仅：

1. **init** = OpenBMI 正式 Acc_paper shallow（`run_20260807_135828`）  
2. **通道重排** 游戏序 → OpenBMI 序（同 [04](../04_旁路_OpenBMI权重_游戏零样本与门控/)）  
3. **仅 shallow**

状态：**已结案**（主记录见 [`总结/结果登记表.md`](总结/结果登记表.md)）。

## 文档

| 文件 | 说明 |
|------|------|
| [方案.md](方案.md) | 冻结范围 |
| [总结/结果登记表.md](总结/结果登记表.md) | 主报登记 |
| `out/` | split_manifest |
| `results/` | 跑数产物 |

## 跑数

```bash
PY=D:/cyy/MI/.venv/Scripts/python.exe
cd code/train_lab/src/step/game_ft_openbmi_hop100_accpaper
$PY build_splits.py
$PY baseline_shallow.py
# 冒烟：$PY baseline_shallow.py --smoke --subjects sub02
```

上级部署口径：[../README.md](../README.md)。
