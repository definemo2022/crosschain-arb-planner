# adapters.py
from __future__ import annotations
import asyncio
from typing import Optional, Callable, Dict, Any
from leg_quote import LegQuote, LEG_OK, LEG_NO_ADDR, LEG_NO_QUOTE, LEG_ERROR

# callable 签名：await fn(app, chain, in_addr, out_addr, in_wei) -> 任意(raw)
QuoteFn = Callable[[Any, Any, str, str, int], Any]

class BaseAdapter:
    """
    标准适配器基类：
      - name: 适配器名（如 "kyber", "odos"）
      - quote_fn: 具体报价函数指针（由主程序注入，避免循环依赖）
      - supports_chain(): 可由 settings.adapter_support[name] 控制
      - quote(): 统一返回 LegQuote
    """
    name: str = "base"

    def __init__(self, app: Any, quote_fn: Optional[QuoteFn] = None):
        self.app = app
        self.quote_fn = quote_fn

    def supports_chain(self, chain_name: str) -> bool:
        support = self.app.settings.get("adapter_support") or {}
        if not support:  # 未配置则默认全链可用
            return True
        allow = support.get(self.name)
        if not allow:
            return False
        chain_name = (chain_name or "").lower()
        return chain_name in [c.lower() for c in allow]

    async def quote(self, chain: Any, a_sym: str, b_sym: str, base_in: float) -> LegQuote:
        raise NotImplementedError

    # —— 通用的资产 / 地址解析 —— #

    def _get_asset(self, sym: str):
        try:
            return self.app.assets[sym.upper()]
        except Exception:
            return None

    @staticmethod
    def _addr(asset: Any, chain_name: str) -> Optional[str]:
        m = getattr(asset, "address", None) or {}
        return m.get(chain_name) or m.get(chain_name.lower()) or m.get(chain_name.upper())

    def _decimals(self, asset: Any, default: int = 18) -> int:
        try:
            return int(getattr(asset, "decimals", default))
        except Exception:
            return default

    # —— 统一 raw → out_wei 归一化 —— #

    @staticmethod
    def _normalize_out_wei(raw: Any) -> int:
        """
        支持多形态返回：dict/tuple/纯数字；尽量取到 int(out_wei)。
        """
        if raw is None:
            return 0
        # dict-like
        if isinstance(raw, dict):
            out = raw.get("out_wei")
            if out is None:
                out = raw.get("amountOut") or raw.get("out")
            try:
                return int(out) if out is not None else 0
            except Exception:
                try:
                    return int(float(out))
                except Exception:
                    return 0
        # tuple/list
        if isinstance(raw, (tuple, list)):
            if len(raw) >= 2:
                a, b = raw[0], raw[1]
                # (adapter, out) 或 (out, adapter)
                if isinstance(a, str):
                    try:
                        return int(b) if not isinstance(b, str) else int(float(b))
                    except Exception:
                        pass
                if isinstance(b, str):
                    try:
                        return int(a) if not isinstance(a, str) else int(float(a))
                    except Exception:
                        pass
            for item in raw:
                if isinstance(item, dict):
                    v = BaseAdapter._normalize_out_wei(item)
                    if v > 0:
                        return v
            return 0
        # 数字/字符串
        try:
            return int(raw)
        except Exception:
            try:
                return int(float(raw))
            except Exception:
                return 0

    # —— 统一组装 LegQuote —— #

    def _make_leg(self, chain_name: str, a_sym: str, b_sym: str, base_in: float,
                  adapter: str, out_wei: int, dB: int, status: str, note: str = "") -> LegQuote:
        out_b = (out_wei / float(10 ** dB)) if (out_wei and dB is not None) else None
        return LegQuote(
            chain=chain_name, a=a_sym, b=b_sym, base_in=float(base_in),
            in_wei=None, out_b=out_b, out_wei=int(out_wei or 0),
            adapter=adapter, status=status, note=note
        )


class KyberAdapter(BaseAdapter):
    name = "kyber"

    async def quote(self, chain: Any, a_sym: str, b_sym: str, base_in: float) -> LegQuote:
        chain_name = getattr(chain, "name", "?")
        if not self.supports_chain(chain_name):
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, 18, LEG_NO_QUOTE, "adapter disabled on chain")

        if not callable(self.quote_fn):
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, 18, LEG_ERROR, "adapter quote_fn missing")

        A = self._get_asset(a_sym); B = self._get_asset(b_sym)
        if not (A and B):
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, 18, LEG_ERROR, "unknown asset")

        in_addr  = self._addr(A, chain_name)
        out_addr = self._addr(B, chain_name)
        if not (in_addr and out_addr):
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, 18, LEG_NO_ADDR, "missing token address")

        dA = self._decimals(A); dB = self._decimals(B)
        try:
            in_wei = int(float(base_in) * (10 ** dA))
        except Exception:
            in_wei = 0

        if in_wei <= 0:
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, dB, LEG_ERROR, "invalid base_in")

        try:
            raw = await self.quote_fn(self.app, chain, in_addr, out_addr, in_wei)
            out_wei = self._normalize_out_wei(raw)
            if out_wei <= 0:
                return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, dB, LEG_NO_QUOTE, "no quote")
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, out_wei, dB, LEG_OK, "")
        except asyncio.CancelledError:
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, dB, LEG_NO_QUOTE, "cancelled")
        except Exception as e:
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, dB, LEG_ERROR, str(e))


class OdosAdapter(BaseAdapter):
    name = "odos"

    async def quote(self, chain: Any, a_sym: str, b_sym: str, base_in: float) -> LegQuote:
        chain_name = getattr(chain, "name", "?")
        if not self.supports_chain(chain_name):
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, 18, LEG_NO_QUOTE, "adapter disabled on chain")

        if not callable(self.quote_fn):
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, 18, LEG_ERROR, "adapter quote_fn missing")

        A = self._get_asset(a_sym); B = self._get_asset(b_sym)
        if not (A and B):
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, 18, LEG_ERROR, "unknown asset")

        in_addr  = self._addr(A, chain_name)
        out_addr = self._addr(B, chain_name)
        if not (in_addr and out_addr):
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, 18, LEG_NO_ADDR, "missing token address")

        dA = self._decimals(A); dB = self._decimals(B)
        try:
            in_wei = int(float(base_in) * (10 ** dA))
        except Exception:
            in_wei = 0

        if in_wei <= 0:
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, dB, LEG_ERROR, "invalid base_in")

        try:
            raw = await self.quote_fn(self.app, chain, in_addr, out_addr, in_wei)
            out_wei = self._normalize_out_wei(raw)
            if out_wei <= 0:
                return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, dB, LEG_NO_QUOTE, "no quote")
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, out_wei, dB, LEG_OK, "")
        except asyncio.CancelledError:
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, dB, LEG_NO_QUOTE, "cancelled")
        except Exception as e:
            return self._make_leg(chain_name, a_sym, b_sym, base_in, self.name, 0, dB, LEG_ERROR, str(e))


# —— 构造与聚合 —— #

def build_adapters(app: Any, quote_fns: Dict[str, Optional[QuoteFn]]):
    """
    从主程序注入的报价函数字典构建适配器列表。
    例如：{"kyber": kyber_quote, "odos": odos_quote}
    """
    lst = []
    if quote_fns.get("kyber") is not None:
        lst.append(KyberAdapter(app, quote_fns.get("kyber")))
    if quote_fns.get("odos") is not None:
        lst.append(OdosAdapter(app, quote_fns.get("odos")))
    return lst

async def best_quote_leg(app: Any, chain: Any, a_sym: str, b_sym: str, base_in: float,
                         adapters: list[BaseAdapter]) -> LegQuote:
    """
    并发调用各 adapter.quote(...)，选 out_wei 最大的 LegQuote。
    全部失败时返回 status=no-quote 的行。
    """
    if not adapters:
        return LegQuote(chain=getattr(chain, "name", "?"), a=a_sym, b=b_sym, base_in=float(base_in),
                        in_wei=None, out_b=None, out_wei=0, adapter="-", status=LEG_NO_QUOTE,
                        note="no adapters")

    soft = float(app.settings.get("soft_timeout_sec", 5.0))
    tasks = []
    for ad in adapters:
        # 跳过不支持的链
        if not ad.supports_chain(getattr(chain, "name", "")):
            continue
        # 每个适配器再套一层 timeout
        tasks.append(asyncio.create_task(asyncio.wait_for(ad.quote(chain, a_sym, b_sym, base_in), timeout=soft)))

    if not tasks:
        return LegQuote(chain=getattr(chain, "name", "?"), a=a_sym, b=b_sym, base_in=float(base_in),
                        in_wei=None, out_b=None, out_wei=0, adapter="-", status=LEG_NO_QUOTE,
                        note="no adapters enabled on chain")

    done, pending = await asyncio.wait(tasks, timeout=soft, return_when=asyncio.ALL_COMPLETED)
    for p in pending:
        p.cancel()

    cands: list[LegQuote] = []
    for t in done:
        try:
            leg: LegQuote = t.result()
            if isinstance(leg, LegQuote) and (leg.out_wei or 0) > 0 and leg.status == LEG_OK:
                cands.append(leg)
        except Exception:
            pass

    if not cands:
        # 尝试收集一个“最有信息”的失败腿（如果存在）
        for t in done:
            try:
                leg = t.result()
                if isinstance(leg, LegQuote):
                    return leg
            except Exception:
                pass
        return LegQuote(chain=getattr(chain, "name", "?"), a=a_sym, b=b_sym, base_in=float(base_in),
                        in_wei=None, out_b=None, out_wei=0, adapter="-", status=LEG_NO_QUOTE, note="no quote")

    # 按 out_wei 最大选最优
    best = max(cands, key=lambda x: x.out_wei)
    return best
