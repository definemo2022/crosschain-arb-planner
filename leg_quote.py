# leg_quote.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from assets import Token
from chains import Chain

# 状态码
LEG_OK = "ok"
LEG_NO_ADDR = "no-addr"
LEG_NO_QUOTE = "no-quote"
LEG_ERROR = "error"

@dataclass
class LegQuote:
    """单腿报价结果"""
    chain: Chain           # 链名称
    a: Token              # 代币 A 符号
    b: Token              # 代币 B 符号
    base_in: float      # 基础输入数量
    in_wei: Optional[int]=None   # 输入 wei 数量
    out_b: Optional[float]=None  # 输出基础数量
    out_wei: Optional[int]=None           # 输出 wei 数量
    adapter: Optional[str]=None        # 适配器名称
    status: Optional[str]=None         # 状态码
    note: Optional[str]=None      # 备注信息
   
    def as_html_row(self) -> str:
        def esc(s):
            s = "" if s is None else str(s)
            return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        def fmt(x, n=6):
            if x is None: return "—"
            try: return f"{float(x):,.{n}f}"
            except Exception: return esc(x)
        def fmt_int(x):
            if x in (None, ""): return "—"
            try: return f"{int(x):,}"
            except Exception: return esc(x)

        return (
            "<tr>"
            f"<td>{esc(self.chain.name)}</td>"
            f"<td>{esc(self.a.name)}</td>"
            f"<td>{esc(self.b.name)}</td>"
            f"<td class='num'>{fmt(self.base_in, 6)}</td>"
            f"<td>{esc(self.adapter or '-')}</td>"
            f"<td class='num'>{fmt(self.out_b, 6)}</td>"
            f"<td class='num'>{fmt_int(self.out_wei)}</td>"
            f"<td class='{esc(self.status)}'>{esc(self.status)}</td>"
            f"<td>{esc(self.note)}</td>"
            "</tr>"
        )

    def __str__(self) -> str:
        if self.status != LEG_OK:
            return f"{self.chain.name}:{self.a.name}>{self.b.name} [{self.status}] {self.note}"
        return (f"{self.chain.name}:{self.a.name}>{self.b.name} "
                f"{self.base_in:.2f}->{self.out_b:.2f} "
                f"via {self.adapter}")

# leg_quote.py 追加/替换 TwoLegResult 定义
@dataclass
class TwoLegResult:
    
    leg1: LegQuote
    leg2: LegQuote
    final_a: Optional[float]=None
    pnl: Optional[float]=None
    pnl_pct: Optional[float]=None
    adapter: Optional[str]=None
    status: Optional[str]=None
    note: Optional[str]=None

    
    def as_html_row(self) -> str:
        def esc(s):
            s = "" if s is None else str(s)
            return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        def fmt(x, n=6):
            if x is None: return "—"
            try: return f"{float(x):,.{n}f}"
            except Exception: return esc(x)
        def fmt_pct(x, n=4):
            if x is None: return "—"
            try: return f"{float(x):,.{n}f}%"
            except Exception: return esc(x)

        l1 = self.leg1
        l2 = self.leg2
        return (
            "<tr>"
            f"<td>{esc(self.leg1.chain.name)}</td>"
            f"<td>{esc(self.leg2.chain.name)}</td>"
            f"<td>{esc(self.leg1.a.name)}</td>"
            f"<td>{esc(self.leg1.b.name)}</td>"
            f"<td class='num'>{fmt(self.leg1.base_in)}</td>"
            f"<td>{esc(getattr(l1, 'adapter', '-') or '-')}</td>"
            f"<td class='num'>{fmt(getattr(l1, 'out_b', None))}</td>"
            f"<td>{esc(getattr(l2, 'adapter', '-') if l2 else '-')}</td>"
            f"<td class='num'>{fmt(self.final_a)}</td>"
            f"<td class='num'>{fmt(self.pnl)}</td>"
            f"<td class='num'>{fmt_pct(self.pnl_pct)}</td>"
            f"<td class='{esc(self.status)}'>{esc(self.status)}</td>"
            f"<td>{esc(self.note)}</td>"
            "</tr>"
        )
