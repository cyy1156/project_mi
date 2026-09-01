# M08 · 编排、WebSocket 与前端

## 目标

理解操作台是「路由器」：浏览器点按钮 → WS JSON → `OperatorService` 调会话；**业务计算不在 JS 里做完**。

## 架构直觉

```text
operator.html / operator.js
        │  WebSocket :8765
        ▼
orchestrator.OperatorService     ← 消息路由（文件很大）
        │
        ├─ 开采集 AcquisitionFacade
        ├─ 跑 session_v3 / v2 / v4
        ├─ 推送波形与判定到前端
        └─ 触发微调 job（线程）
        │
        ▼
index.html（被试诱导 / 游戏画面）
```

## 对应代码

| 内容 | 路径 | 看什么 |
|------|------|--------|
| 入口 | `tools/open_operator.py` | 起 HTTP+WS |
| 编排 | `experiment/orchestrator.py` | **按需搜**：`_handle_` · `start` · `finetune` · `v3`；勿通读 |
| WS 协议 | `docs/ws_protocol.md` | 消息 type 列表 |
| 操作台前端 | `web/js/operator.js` · `web/operator.html` | Setup / Run / Summary |
| 被试页 | `web/index.html` · 相关 js | 诱导与反馈 |
| 按键设计 | `docs/操作按键设计.md` | 操作者 vs 被试 |

## 精读策略（防劝退）

1. 读 `ws_protocol.md` 前两节。  
2. 在 `operator.js` 搜 `start` / `finetune`，看发出的 JSON。  
3. 在 `orchestrator.py` **全文搜索**同一个 type 字符串，只读对应 `_handle_*` 函数。  
4. 从 handle 跳到 `session_v3` 调用行即停。

## 动手题

1. 打开浏览器开发者工具 WS 帧，点「开始」看一条消息。  
2. 对照协议文档写出该消息的 `type`。  
3. 回答：计分最终权威在后端哪（`trial_scoring`），前端伸手动画只是反馈。

## 验收

- [ ] 能描述「点击 → WS → handle → session」  
- [ ] 知道不要通读 orchestrator  

## 下一模块

[`M09_落盘与alignment.md`](M09_落盘与alignment.md)
