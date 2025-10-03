# logger_utils.py
from __future__ import annotations
import logging, os, sys, threading, traceback, time
from typing import Optional

__all__ = ["init_debug_logger", "NullLogger", "SafeDebugLogger"]

class NullLogger:
    """兜底空 logger：所有方法都是 no-op，panel() 返回空字符串。"""
    def info(self, msg, *args, **kwargs):     pass
    def warn(self, msg, *args, **kwargs):     pass
    def warning(self, msg, *args, **kwargs):  pass
    def error(self, msg, *args, **kwargs):    pass
    def exception(self, msg, *args, **kwargs):pass
    def panel(self, *a, **k) -> str:          return ""
    def js_auto_refresh(self, seconds) -> str: return "" 

class SafeDebugLogger:
    """
    安全 Logger：
    - 任何 log / panel 异常都被吞掉，写 stderr 兜底，不影响主流程
    - 文件 & 控制台双写可选
    """
    def __init__(self, log_file: Optional[str], to_console: bool = True):
        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"arbplanner.{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        # 避免重复 handler
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)

        formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")

        if log_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                self._logger.addHandler(fh)
            except Exception:
                # 文件句柄失败也不抛，降级仅 console
                print("[SafeDebugLogger] FileHandler init failed:", file=sys.stderr)
                traceback.print_exc()

        if to_console:
            try:
                sh = logging.StreamHandler(sys.stdout)
                sh.setLevel(logging.INFO)
                sh.setFormatter(formatter)
                self._logger.addHandler(sh)
            except Exception:
                print("[SafeDebugLogger] StreamHandler init failed:", file=sys.stderr)
                traceback.print_exc()

    # 基础安全输出
    def _safe(self, level: str, msg: str):
        try:
            with self._lock:
                getattr(self._logger, level, self._logger.info)(msg)
        except Exception:
            # 永不抛到主程序
            try:
                sys.stderr.write(f"[SafeDebugLogger.{level} FAILED] {msg}\n")
                traceback.print_exc()
            except Exception:
                pass

    # 用一个统一的发射函数，兼容 logging 风格 (msg, *args, **kwargs)
    def _emit(self, level: str, msg, *args, **kwargs):
        try:
            # logging 约定：如果有 args，就用 % 做格式化
            if args:
                try:
                    msg = msg % args
                except Exception:
                    # 容错兜底：拼接字符串，避免抛错影响主流程
                    msg = " ".join([str(msg)] + [str(x) for x in args])

            exc_info = kwargs.get("exc_info", None)
            with self._lock:
                getattr(self._logger, level, self._logger.info)(msg, exc_info=exc_info)
        except Exception:
            try:
                sys.stderr.write(f"[SafeDebugLogger.{level} FAILED] {msg}\n")
                traceback.print_exc()
            except Exception:
                pass

    def info(self, msg, *args, **kwargs):     self._emit("info", msg, *args, **kwargs)
    def warn(self, msg, *args, **kwargs):     self._emit("warning", msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs):  self._emit("warning", msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs):    self._emit("error", msg, *args, **kwargs)
    def exception(self, msg, *args, **kwargs):
        # 与 logging.exception 一致：自动带栈
        kwargs["exc_info"] = True
        self._emit("error", msg, *args, **kwargs)

    # HTML 调试面板：返回一段可嵌入的 <details>，失败则返回空串
    def panel(self, title, html_body: str = "") -> str:
        try:
            import json
            if not isinstance(title, str):
                try:
                    title_str = json.dumps(title, ensure_ascii=False)
                except Exception:
                    title_str = str(title)
            else:
                title_str = title
            safe_title = (
                title_str.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            )
            body = html_body or ""
            return (
                "<details style='margin-top:12px'>"
                f"<summary>{safe_title}</summary>"
                "<div style='padding:8px;border:1px dashed #cbd5e1;margin-top:8px'>"
                f"{body}"
                "</div></details>"
            )
        except Exception:
            self._emit("warning", "[panel failed] title=%r", title)
            return ""

    # ← 新增：和旧代码兼容，返回 head 区域可用的刷新标签/脚本
    def js_auto_refresh(self, seconds) -> str:
        try:
            s = int(seconds)
        except Exception:
            s = 8
        # 同时给 meta 和 JS（有些页面只放 head_extra）
        return (
            f"<meta http-equiv='refresh' content='{s}'>"
            f"<script>setTimeout(function(){{location.reload()}}, {s*1000});</script>"
        )

def _decide_log_path(out_html: Optional[str]) -> Optional[str]:
    if not out_html:
        return None
    base, _ = os.path.splitext(out_html)
    return base + ".log"

def init_debug_logger(app, out_html: Optional[str], verbose: bool) -> SafeDebugLogger | NullLogger:
    """
    统一入口：永远返回一个可用的 logger（不会是 None）
    - 若 app.settings.debug 为 False，则返回 NullLogger（完全 no-op）
    - 否则返回 SafeDebugLogger（文件 + 控制台）
    """
    try:
        settings = getattr(app, "settings", {}) or {}
        debug_enabled = bool(settings.get("debug", True))
        to_console = bool(verbose)
    except Exception:
        # app 异常也不要影响主流程
        debug_enabled, to_console = True, bool(verbose)

    if not debug_enabled:
        return NullLogger()

    log_path = _decide_log_path(out_html)
    try:
        lg = SafeDebugLogger(log_file=log_path, to_console=to_console)
        lg.info(f"DebugLogger initialized -> {log_path or '(console only)'}")
        return lg
    except Exception:
        # 任何初始化问题都降级到 NullLogger
        try:
            sys.stderr.write("[init_debug_logger] fallback to NullLogger\n")
        except Exception:
            pass
        return NullLogger()
