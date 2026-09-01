# M09 · 落盘结构与 alignment

## 目标

拿到任意一场 session，能判断「数据是否可进 FT / 是否对齐合格」。

## 典型目录

```text
data/sessions/<subject>_<session>_<stamp>/
  eeg.csv / events.jsonl
  run_config.json          ← 本场参数快照（复现关键）
  session.meta.json
  continuous/
  by_phase/                ← 分阶段切片（若启用）
  alignment/
    trial_table.csv
    verify_report.json     ← passed: true/false
  manifest.json
  99_summary/              ← Phase4 指针等
```

被试维度（登录后）：

```text
data/subjects/<id>/
  sessions/ · models/current · models/ft_runs/ · …
```

## 对应代码

| 内容 | 路径 | 看什么 |
|------|------|--------|
| 建会话目录 | `experiment/session.py` | `create_session_dir` |
| 布局整理 | `experiment/session_layout.py` | continuous / by_phase |
| 对齐 | `experiment/alignment.py` | verify |
| 被试注册 | `experiment/subject_registry.py` | subjects/{id} |
| Phase4 | `offline/` + README「一键 Phase4」 | 切 epoch |

## 动手题

1. 打开 M00 的 `run_config.json`，找出 `prep_s` / `imagine_s` / `readout_mode`。  
2. 读 `verify_report.json` 的 checks 列表。  
3. 对比：历史 `phase2_full` session（若有）与现行 Align 的 meta 差异。

## 验收

- [ ] 能说明 `run_config.json` 为何重要  
- [ ] 知道 FT 默认只要 `v3_session`，排除 `v4_session`（F11）  

## 下一模块

[`M10_被试与LeaveNext.md`](M10_被试与LeaveNext.md)
