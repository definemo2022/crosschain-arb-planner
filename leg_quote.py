# leg_quote.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

LEG_OK = "ok"
LEG_NO_ADDR = "no-addr"
LEG_NO_QUOTE = "no-quote"
LEG_ERROR = "error"

@dataclass
class LegQuote:
    chain: str
    a: str
    b: str
    base_in: float
    in_wei: Optional[int]
    out_b: Optional[float]
    out_wei: int
    adapter: str
    status: str
    note: str = ""
    elapsed_ms: Optional[int] = None

    @classmethod
    def from_row_dict(cls, row: Dict[str, Any]) -> "LegQuote":
        def _f(x, dv=None):
            return dv if x is None else x
        return cls(
            chain=str(row.get("chain","?")),
            a=str(row.get("a","?")).upper(),
            b=str(row.get("b","?")).upper(),
            base_in=float(_f(row.get("base_in"), 0.0)),
            in_wei=(int(row["in_wei"]) if row.get("in_wei") is not None else None),
            out_b=(float(row["out_b"]) if row.get("out_b") is not None else None),
            out_wei=int(_f(row.get("out_wei"), 0)),
            adapter=str(row.get("adapter","-")),
            status=str(row.get("status","")),
            note=str(row.get("note","")),
            elapsed_ms=(int(row["elapsed_ms"]) if row.get("elapsed_ms") is not None else None),
        )

    def to_row_dict(self) -> Dict[str, Any]:
        return {
            "chain": self.chain, "a": self.a, "b": self.b, "base_in": self.base_in,
            "in_wei": self.in_wei, "out_b": self.out_b, "out_wei": self.out_wei,
            "adapter": self.adapter, "status": self.status, "note": self.note,
            "elapsed_ms": self.elapsed_ms,
        }

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
            f"<td>{esc(self.chain)}</td>"
            f"<td>{esc(self.a)}</td>"
            f"<td>{esc(self.b)}</td>"
            f"<td class='num'>{fmt(self.base_in, 6)}</td>"
            f"<td>{esc(self.adapter or '-')}</td>"
            f"<td class='num'>{fmt(self.out_b, 6)}</td>"
            f"<td class='num'>{fmt_int(self.out_wei)}</td>"
            f"<td class='{esc(self.status)}'>{esc(self.status)}</td>"
            f"<td>{esc(self.note)}</td>"
            "</tr>"
        )

# leg_quote.py 追加/替换 TwoLegResult 定义
@dataclass
class TwoLegResult:
    from_chain: str
    to_chain: str
    a: str
    b: str
    base_in_a: float
    leg1: "LegQuote"
    leg2: Optional["LegQuote"]
    final_a: Optional[float]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    status: str
    note: str = ""
    # 新增：size 优化结果（可选）
    best_size: Optional[float] = None
    best_pnl: Optional[float] = None
    best_pnl_pct: Optional[float] = None

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
            f"<td>{esc(self.from_chain)}</td>"
            f"<td>{esc(self.to_chain)}</td>"
            f"<td>{esc(self.a)}</td>"
            f"<td>{esc(self.b)}</td>"
            f"<td class='num'>{fmt(self.base_in_a)}</td>"
            f"<td>{esc(getattr(l1, 'adapter', '-') or '-')}</td>"
            f"<td class='num'>{fmt(getattr(l1, 'out_b', None))}</td>"
            f"<td>{esc(getattr(l2, 'adapter', '-') if l2 else '-')}</td>"
            f"<td class='num'>{fmt(self.final_a)}</td>"
            f"<td class='num'>{fmt(self.pnl)}</td>"
            f"<td class='num'>{fmt_pct(self.pnl_pct)}</td>"
            # 新增三列（无结果时显示 em dash）
            f"<td class='num'>{fmt(self.best_size)}</td>"
            f"<td class='num'>{fmt(self.best_pnl)}</td>"
            f"<td class='num'>{fmt_pct(self.best_pnl_pct)}</td>"
            f"<td class='{esc(self.status)}'>{esc(self.status)}</td>"
            f"<td>{esc(self.note)}</td>"
            "</tr>"
        )
