/** @typedef {{ type: string, [k: string]: any }} Msg */

export class WsClient {
  /**
   * @param {string} url
   * @param {(msg: Msg) => void} onMessage
   * @param {(s: string) => void} onStatus
   */
  constructor(url, onMessage, onStatus) {
    this.url = url;
    this.onMessage = onMessage;
    this.onStatus = onStatus;
    this.ws = null;
    this._retry = 0;
  }

  connect() {
    this.onStatus("连接 WebSocket…");
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      this._retry = 0;
      this.onStatus("已连接");
      this.send({ type: "ready" });
      this.send({ type: "sync" });
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        this.onMessage(msg);
      } catch {
        /* ignore */
      }
    };
    ws.onclose = () => {
      this.onStatus("连接断开，重试中…");
      if (this._retry >= 8) {
        this.onStatus("服务已结束（可关闭页面）");
        return;
      }
      const delay = Math.min(3000, 400 + this._retry * 400);
      this._retry += 1;
      setTimeout(() => this.connect(), delay);
    };
    ws.onerror = () => {
      this.onStatus("WebSocket 错误");
    };
  }

  /** @param {Msg} msg */
  send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }
}
