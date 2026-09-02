# experiment_game · 从 0 到 1：自己写出完整代码

> **给谁看**：要用自己的手写出与原版同构流水线的人（可用 AI 辅助，但业务逻辑落在你的仓库）。  
> **你的起点**：串口 / COM 较熟；LSL、会话、推理、计分、微调可以不会。  
> **目标**：在独立工程（推荐 `mi_scratch/`）中实现完整链路，原版 `experiment_game` 只作**标准答案与对照**。

## 怎么用这个文件夹

0. **代码不熟必读** [`00_写代码小白指南.md`](00_写代码小白指南.md)（对照原版、写测试、卡死怎么办）  
0b. **要看完整示例代码** → [`examples/README.md`](examples/README.md)（按模块的教学参考答案；作业仍在 `self_learing` 自己重写）  
1. 再读 [`01_整体设计逻辑.md`](01_整体设计逻辑.md)（地图）  
2. 按 [`02_学习路线图.md`](02_学习路线图.md) 从 **M00** 往下  
3. 每模块笔记里都有：**原版精读表（打开哪个文件、搜哪个函数）→ 函数签名草稿 → 逐步仿写 → 常见坑**  
4. 每学完：改 [`进度.md`](进度.md)

## 核心纪律（写出导向）

1. **另开目录**：业务写在 `mi_scratch/`（或自定名），不要当练习场大拆现网采人代码。  
2. **先测后码**：每个模块先红灯测试，再实现到绿。  
3. **行为对齐原版**：事件名、时序数字、窗几何、F5 公式与冻结/`protocol.yaml` 一致。  
4. **允许复用**：`adapt_engine` 骨干、`pipeline/finetune` 训练核——须在 README 声明；调度与数据流必须自写。  
5. **禁止整文件粘贴** `orchestrator.py` / `session_v3.py` / `inference_v2.py` 当「复现」。  
6. **冲突裁决**：`docs/框架冻结确认_*.md` + `config/protocol.yaml` > 旧文档。

## 目录

| 文件 | 作用 |
|------|------|
| [`00_写代码小白指南.md`](00_写代码小白指南.md) | **不会写时先看** |
| [`examples/`](examples/README.md) | **按模块完整教学代码示例**（M01–M08） |
| [`01_整体设计逻辑.md`](01_整体设计逻辑.md) | 系统地图、两条流 |
| [`02_学习路线图.md`](02_学习路线图.md) | M00–M11 顺序与过关线 |
| [`进度.md`](进度.md) | 你的进度 |
| [`modules/`](modules/) | 参考哪些原版文件/函数、怎么仿写 |

## 权威文档（实现时对照）

| 优先级 | 路径 |
|--------|------|
| 1 | `../docs/框架冻结确认_20260829.md` |
| 2 | `../config/protocol.yaml`（现行数字以仓库文件为准） |
| 3 | `../docs/范式对齐_OpenBMI与fnz_v3_20260827.md` |
| 4 | `../docs/方案讲解_写给同学_20260829.md` |
| 5 | `../README.md` |

## 一句话架构（你要亲手实现的）

```text
SampleSource → RingBuffer → EegBus → CSV
                ↓
         TrialFSM 打 events
                ↓
         切窗+预处理 → E1f/Fake → F5 计分
                ↓
         session 落盘 + alignment
                ↓
         Leave-Next 调度 →（复用或自写）FT → current
```

## 推荐工程骨架（M00 创建）

```text
mi_scratch/
  config/protocol.yaml
  src/mi_scratch/
  tests/
  tools/run_synthetic_session.py
  README.md          # 对照表：原版路径 ↔ 你的路径 ↔ 复用声明
```

---

*更新：2026-09-01 · 目标从「读懂原版」改为「自己写出完整代码」*
