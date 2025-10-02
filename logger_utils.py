# logger_utils.py
from __future__ import annotations
import logging
from typing import Optional

class DebugLogger:
    """
    可选的调试/日志工具：
    - enabled=False 时，panel() 返回空字符串，不影响页面
    - setup() 配置文件/控制台日志
    - info/debug/exception 等方法直接代理到 logging
    - panel(stats) 生成嵌入 HTML 的调试面板（含日志尾部）
    - js_auto_refresh(refresh_sec) 返回 JS 兜底刷新脚本
    """
    def __init__(self, enabled: bool = False):
        self.enabled: bool = enabled
        self.path: Optional[str] = None
        self._logger = logging.getLogger("arb")

    def setup(self, log_path: str, *, verbose: bool = False, level: int = logging.DEBUG):
        self.path = log_path
        self._logger.handlers.clear()
        self._logger.setLevel(level)

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        # 文件日志
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        # 控制台输出
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG if verbose else logging.INFO)
        sh.setFormatter(fmt)
        self._logger.addHandler(sh)

        self._logger.debug("DebugLogger initialized -> %s", log_path)

    # 代理常用方法
    def info(self, *a, **k): self._logger.info(*a, **k)
    def debug(self, *a, **k): self._logger.debug(*a, **k)
    def warning(self, *a, **k): self._logger.warning(*a, **k)
    def error(self, *a, **k): self._logger.error(*a, **k)
    def exception(self, *a, **k): self._logger.exception(*a, **k)

    def _tail(self, n=200) -> str:
        if not self.path:
            return ""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception:
            return ""

    def panel(self, stats: dict) -> str:
        """
        生成 HTML 调试面板（简单表格 + 日志尾部）。
        enabled=False 时返回空字符串。
        """
        if not self.enabled:
            return ""
        esc = lambda s: (str(s) if s is not None else "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in stats.items())
        tail = self._tail(120)
        tail_html = f"<details open><summary>Last log lines</summary><pre style='font-size:12px;white-space:pre-wrap;background:#0b1022;color:#cbd5e1;padding:8px;border-radius:8px'>{esc(tail)}</pre></details>" if tail else ""
        return f"""
        <div style="border:1px dashed #94a3b8;padding:10px;border-radius:8px;margin:12px 0;background:#f8fafc">
          <b>Debug</b>
          <table class="t" style="margin-top:6px"><tbody>{rows}</tbody></table>
          {tail_html}
        </div>
        """

    @staticmethod
    def js_auto_refresh(refresh_sec: int) -> str:
        """返回一个 setTimeout 刷新的 JS 片段（用于 meta refresh 失效的兜底）。"""
        try:
            sec = int(refresh_sec)
        except Exception:
            sec = 8
        return f"<script>setTimeout(function(){{ location.reload(true); }}, {sec}*1000);</script>"
