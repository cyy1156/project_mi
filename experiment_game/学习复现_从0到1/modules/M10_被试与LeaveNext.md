# M10 · 被试登录与 Leave-Next 微调

## 目标

分清：**会话中不在线改权重**；采后用 Leave-Next 爬坡，再 promote 到 `models/current`。

## 概念

| 模式 | 含义 |
|------|------|
| 轮间在线 FT | 旧方案；**现行关闭**（F7） |
| **Leave-Next** | 用前面若干 run 训练，hold 下一个 run 评估，逐步爬坡 |
| promote | 把某次 `ft_runs` 权重提升为 `models/current` |
| 强制晋升 | 门控 FAIL 仍可警告后替换（F8，如 fnz） |

replay 默认 **0.10** 混入源域窗防遗忘（F9）。

## 对应代码

| 内容 | 路径 | 看什么 |
|------|------|--------|
| 微调主实现 | `pipeline/finetune.py` | `run_subject_finetune` |
| CLI 壳 | `tools/ft_subject_from_v3.py` | 参数如何进 pipeline |
| Leave-Next 工具 | `tools/run_leave_next_e1f_task_ramp.py` | 过滤 `phase_mode==v3_session` |
| 操作台触发 | `orchestrator.py` `_handle_finetune_*` | 线程跑 FT、promote |
| 文档 | `docs/被试登录与按session微调方案_20260827.md` | |
| 真人证据 | `资料/模型训练/31_.../总结/结果登记表.md` | syj / fnz |

## 精读顺序

1. 被试登录文档前半：目录约定  
2. `run_leave_next_e1f_task_ramp.py` 文件头注释 + 过滤 v3 的函数  
3. `pipeline/finetune.py`：只读函数签名与 docstring（全文很长，抓协议 `openbmi_align`）  
4. 冻结 F7/F8/F9/F11  

## 动手题

1. 解释：为何 v4 帽检 session 不能进 Leave-Next 训练集。  
2. 对照方案 31：syj 末档 MI 94.4% 与零样本对比说明什么。  
3. （有数据时）在 Summary 走一遍 FT；或只读已有 `ft_runs/*summary.json`。

## 验收

- [ ] 能画出「多场 v3 → Leave-Next → current」  
- [ ] 能说出 replay=0.10 的用途  

## 下一模块

[`M11_端到端验收.md`](M11_端到端验收.md)
