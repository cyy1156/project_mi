# 协议：滑窗投票 · Acc_paper 早停 · OpenBMI（仅 Three）

> 对齐 OpenBMI Acc_paper；本臂 **v1.12** 起仅跑 **Three**。  
> 划分：`iter_subject_kfold`（Test/Val **按被试**）。

**选模/早停/读数口径不得另起一套。**

---

## 1. 滑窗几何与数据（冻结）

| 项 | 取值 |
|----|------|
| \(T_w\) / hop | **2 s** / **100 ms** @ 250 Hz |
| 通道 | 8（OpenBMI 预处理序） |
| 数据 | A0：旧 `openbmi_2s_hop100`；A1+：新臂见 [`数据切片与边界过滤说明.md`](../数据切片与边界过滤说明.md) |
| post-MI 尾段 | MI 有效段后额外保留 **≥1.6 s（400 点）**，保证末窗也有真 future |
| 训练 / Acc_paper 窗 | past+cur+future **齐全**；**同一套**；缺 past **裁掉** |
| 冷启动 | 可见 &lt;600 → 不预测；之后均够 |
| 新方法输入 | `(B,8,1000)`；A0 基线 `(B,8,500)` |

---

## 2. Acc_paper（早停与主报）

- 试次内正确窗占比 **>0.5** 计对；=0.5 计错  
- Val 早停 / Test 主报 = **Acc_paper**；BalAcc_maj 只附报  

口径：

```text
Tw=2s hop=100ms openbmi_sess01+02 postMI>=1.6s
subject_key=openbmi:subjNN Three-only
early_stop=val_acc_paper select=test_acc_paper
balbatch Adam batch=256/512 patience=20
```

---

## 3. 超参锚点

| 项 | 值 |
|----|-----|
| folds / val_ratio / seed | 5 / 0.2 / 42（**Val 按被试**） |
| max_epochs / patience | 300 / **20** |
| batch | **256 / 512** |
| lr / wd / drop | 1e-4 / 1e-4 / 0.5 |
| optimizer | **Adam** |
| 头任务 | **仅 Three** |
| Encoder | A0 主表=自写 shallow（先与 braindecode 对齐）；P0+=自写 |
| SIGReg | LeJEPA，`num_slices=1024` |
| D | **40**（不做 40→128） |

---

## 4. A0 对照

1. braindecode 正式臂：量级参考（可引用旧 run）。  
2. **必做**：自写 `shallowfbcsp` + 本协议超参重训 Three → 与上对照；对齐后作为主表 A0。

---

## 5. 检查清单

- [ ] post-MI≥1.6s；齐全窗；缺 past 裁掉  
- [ ] 仅 Three；Adam；batch 256/512；patience 20  
- [ ] Val/Test 按被试（`iter_subject_kfold`）  
- [ ] 自写 A0 已与 braindecode 对齐  
- [ ] §3.2.1 索引单测通过（失败停训；future 扰动比值 ≥3）  

- [ ] 无 40→128 投影  
