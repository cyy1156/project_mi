# M09 · 落盘产品化（详细：参考谁、怎么写）

---

## 一句话目标

会话目录达到：**可复现、可校验、能告诉 FT 这是不是完整 v3**。

---

## 原版精读表

| 步 | 文件 | 搜 / 读 | 看懂即停 |
|----|------|---------|----------|
| 1 | 你的对照场目录 | 列出文件名 | 原版有哪些文件 |
| 2 | 对照场 `run_config.json` | 全文 | 快照里有哪些键 |
| 3 | 对照场 `session.meta.json` | 全文 | `phase_mode` 等 |
| 4 | 对照场 `alignment/verify_report.json` | `passed`/`checks` | 报告形状 |
| 5 | `experiment_game/experiment/session.py` | `create_session_dir` | 创建逻辑 |
| 6 | `experiment_game/experiment/session_layout.py` | 类/函数名浏览 | continuous/by_phase（可简化） |
| 7 | `experiment_game/experiment/alignment.py` | verify 主函数 | 补全你的 checks |
| 8 | `experiment_game/experiment/subject_registry.py` | `ensure` / 注册函数 | subjects 目录 |

---

## 你要写的签名草稿

### `session_layout.py`

```python
def write_run_config(out_dir: Path, cfg: dict) -> None:
    (out_dir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def write_session_meta(out_dir: Path, *, subject: str, phase_mode: str, complete: bool) -> None:
    ...
```

`cfg` 至少放入：从 protocol 读的 timing、windowing、fs、channel、readout。

### `subject_registry.py`

```python
def ensure_subject(root: Path, subject_id: str) -> Path:
    p = root / "subjects" / subject_id
    (p / "models" / "current").mkdir(parents=True, exist_ok=True)
    (p / "models" / "ft_runs").mkdir(parents=True, exist_ok=True)
    (p / "sessions").mkdir(parents=True, exist_ok=True)
    return p
```

### 增强 `alignment.py`

在 M03 基础上增加 1～2 个 check，例如：是否存在 `mi_start`、数量是否 >0。

### 测试

- run_config 含 `imagine_s`。  
- abort 路径：`complete is False`。

---

## 逐步仿写

1. 对照场 `run_config.json` 键列表 → 你的 `write_run_config` 填同样键（值来自你的 yaml）。  
2. meta 写上 `phase_mode: "v3_session"`。  
3. session 正常结束 `complete=true`；做个 `abort` 参数测 false。  
4. `finalize_session_dir` 一次调用写完所有收尾（避免漏文件）。

---

## 常见坑

| 坑 | 做法 |
|----|------|
| 参数只活在内存 | 必须落 run_config |
| phase_mode 随便写 | 用与原版过滤兼容的字符串 |
| 半场也标 complete | 中止必须 false（服务 M10） |

---

## 验收

- [ ] 能说明 run_config 为何重要  
- [ ] incomplete / v3 标记可测 |

## 下一模块

[`M10_被试与LeaveNext.md`](M10_被试与LeaveNext.md)
