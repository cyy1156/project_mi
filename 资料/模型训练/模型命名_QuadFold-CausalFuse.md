# 对外模型命名（冻结）

> 日期：2026-09-04  
> 规则：首现写定义句，之后只称短名。工程内部代号（S0 / E1f-A59 / F5 / all4）**不改路径、不改 JSON id**。

## 一对主名

| 场景 | 对外名 | 读法 |
|------|--------|------|
| 离线 · 指定集交卷 | **QuadFold-59** | quad-fold fifty-nine |
| 在线 · 系统主线 | **CausalFuse-8** | causal-fuse eight |
| 在线 · 个体适配态 | **CausalFuse-8FT** | … eight-F-T |

## 音节 ↔ 事实

### QuadFold-59
| 音节 | 事实 |
|------|------|
| **Quad** | 四成员：Shallow、Shallow-b（T-Shallow）、EEGNet、Conformer；59 ch · 3 s 单窗 · 从零 |
| **Fold** | 融合参数逐折拟合（Fold-Fitted）：LOSO 六折各一套温度 + 融合权；测试 = 六折概率平均（S-ensemble） |
| **59** | 59 通道 |

**首现定义句：**
> QuadFold-59：四成员（Shallow、Shallow-b、EEGNet、Conformer）于 59 通道、3 s 单窗协议下从零训练；融合采用成员温度缩放 + 权重单纯形加权，参数随 LOSO 六折逐折拟合（共 6 套），测试预测为六折模型概率平均（S-ensemble）。嵌套复核 0.511±0.066，折内读数 0.558±0.069（口径说明见方法论节）。

### CausalFuse-8
| 音节 | 事实 |
|------|------|
| **Causal** | 因果滑窗：3 s / hop 100 ms；平滑 lookback=2；试次多数票 |
| **Fuse** | 温度缩放后加权融合；T 由 Val 拟合后冻结；融合权全局常量 `[0.2, 0.2, 0.3, 0.3]`，上线后不随会话重拟合 |
| **8** | OpenBMI 8 通道预训练底座 |

**首现定义句：**
> CausalFuse-8：OpenBMI 54 人预训练四成员底座（8 通道），成员温度由验证集拟合后冻结，融合权重固定为全局常量；在线解码为因果滑窗（3 s/hop 100 ms、lookback=2 平滑、多数票），单窗前向 1.11 ms。被试适配版本记 **CausalFuse-8FT**（Leave-Next 全成员微调 scope=all4；融合参数仍冻结）。

## 后缀（只允许这三个）

| 后缀 | 含义 |
|------|------|
| （无） | 底座 / 初始态 |
| **FT** | Leave-Next 全成员微调（all4）个体适配态 |
| **FT(so)** | 仅浅层成员微调的旁路对照（归档，不作线上名） |

## 复现映射

| 对外名 | 内部代号 | 定义文件 / 产物 |
|--------|----------|-----------------|
| QuadFold-59 | S0 / E1f-A59 | `5070_challenge_mi_59ch_accpaper` · e1f json · `submission_exp34_e1f_a59_sens_full_20260902_1930.csv` |
| CausalFuse-8 / -8FT | E1f 四成员 / F5 / all4 | `experiment_game/config/e1f_four_member.json` · `ft_policy.json` |
| （归档）R-B8 | TWF bridge | Exp39/40 submissions |
