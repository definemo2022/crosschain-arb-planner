# adapters.py
from __future__ import annotations
import asyncio
from typing import Optional, Callable, Dict, Any, List
from leg_quote import LegQuote, LEG_OK, LEG_NO_ADDR, LEG_NO_QUOTE, LEG_ERROR
from assets import Token
from logger_utils import NullLogger
import json
# callable 签名：await fn(app, chain, in_addr, out_ad
# dr, in_wei) -> 任意(raw)
QuoteFn = Callable[[Any, Any, str, str, int], Any]

logger = NullLogger()

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
        # Get logger from app if available
        global logger
        logger = getattr(app, 'logger', NullLogger())

    async def _fetch_http(self, method: str, url: str, *, headers=None, params=None, json_body=None, timeout=12):
        """HTTP 请求封装，统一处理异常"""
        try:
            import httpx
            limits = httpx.Limits(max_keepalive_connections=16, max_connections=32)
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                if method.upper() == "GET":
                    r = await client.get(url, headers=headers, params=params)
                else:
                    r = await client.post(url, headers=headers, params=params, json=json_body)
                return r.status_code, r.text, dict(r.headers)
        except Exception as e:
            # 统一吞掉网络异常，返回"不可用"给上层
            return -1, f"__EXC__:{type(e).__name__}:{str(e)[:180]}", {}
        
    def _safe_json_parse(self, resp_text: str, *, where: str) -> Any:
        """JSON 解析封装，统一处理异常"""
        try:
            return json.loads(resp_text)
        except Exception as e:
            snippet = (resp_text or "")[:400].replace("\n", "\\n")
            raise ValueError(f"{where}: non-JSON or empty body. body[:400]={snippet!r}") from e
        

    def supports_chain(self, chain_name: str) -> bool:
        support = self.app.settings.get("adapter_support") or {}
        if not support:  # 未配置则默认全链可用
            return True
        allow = support.get(self.name)
        if not allow:
            return False
        chain_name = (chain_name or "").lower()
        return chain_name in [c.lower() for c in allow]

    async def quote(self, chain: Any, a_sym: str, b_sym: str, base_in: float,
                   a_token: Optional[Token] = None, b_token: Optional[Token] = None) -> LegQuote:
        """虚方法，由子类实现"""
        raise NotImplementedError("Subclass must implement quote()")

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
                  adapter: str, out_wei: int, dB: int, status: str, note: str = "",
                  a_token: Optional[Token] = None, b_token: Optional[Token] = None) -> LegQuote:
        out_b = (out_wei / float(10 ** dB)) if (out_wei and dB is not None) else None
        return LegQuote(
            chain=chain_name, 
            a=a_sym, 
            b=b_sym, 
            base_in=float(base_in),
            in_wei=None, 
            out_b=out_b, 
            out_wei=int(out_wei or 0),
            adapter=adapter, 
            status=status, 
            note=note,
            a_token=a_token,  # Add Token objects
            b_token=b_token
        )


class KyberAdapter(BaseAdapter):
    name = "kyber"
    
    async def quote(self, chain: Any, a_sym: str, b_sym: str, base_in: float,
                   a_token: Optional[Token] = None, b_token: Optional[Token] = None) -> LegQuote:
        """Get quote from Kyber"""
        chain_name = getattr(chain, "name", "?")
        
        logger.info(f"[DEBUG] KyberAdapter input: base_in={base_in}, decimals={a_token.decimals if a_token else 'N/A'}")

        if not self.supports_chain(chain_name):
            return self._make_leg(chain_name, a_sym, b_sym, base_in, 
                                self.name, 0, None, LEG_NO_QUOTE, "adapter disabled on chain",
                                a_token, b_token)

        if not (a_token and b_token):
            return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                self.name, 0, None, LEG_NO_ADDR, "missing token objects",
                                a_token, b_token)

        try:
            in_wei = int(float(base_in) * (10 ** a_token.decimals))
            logger.info(f"[DEBUG] KyberAdapter in_wei: {in_wei}")
            
            # Original kyber_quote logic
            base = self.app.api.get("kyber_base", "https://aggregator-api.kyberswap.com").rstrip("/")
            qs = {
                "tokenIn": a_token.address,
                "tokenOut": b_token.address,
                "amountIn": str(in_wei),
                "gasInclude": "1",
                "saveGas": "0"
            }
            
            # Get kyber_slug or fall back to chain name
            chain_path = getattr(chain, "kyber_slug", None) or chain.name.lower()
            
            # Try routes endpoint first
            url1 = f"{base}/{chain_path}/api/v1/routes"
            code, text, headers = await self._fetch_http("GET", url1, params=qs)
            if code == 200 and text:
                try:
                    data = self._safe_json_parse(text, where=f"kyber {url1}")
                    out_amt = None
                    d = data.get("data") if isinstance(data, dict) else None
                    if isinstance(d, dict):
                        rs = d.get("routeSummary")
                        if isinstance(rs, dict) and "amountOut" in rs:
                            out_amt = int(rs["amountOut"])
                            out_base = out_amt / (10 ** b_token.decimals)
                            logger.info(f"[DEBUG] KyberAdapter routes output: out_wei={out_amt}, out_base={out_base}, decimals={b_token.decimals}")
                    if out_amt is None and "route" in data and "amountOut" in data["route"]:
                        out_amt = int(data["route"]["amountOut"])
                        out_base = out_amt / (10 ** b_token.decimals)
                        logger.info(f"[DEBUG] KyberAdapter route output: out_wei={out_amt}, out_base={out_base}, decimals={b_token.decimals}")
                    if out_amt is not None:
                        return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                           self.name, out_amt, b_token.decimals, LEG_OK)
                except Exception:
                    pass

            # Try fallback endpoint
            url2 = f"{base}/{chain_path}/api/v1/route"
            code2, text2, _ = await self._fetch_http("GET", url2, params=qs)
            if code2 == 200 and text2:
                try:
                    data2 = self._safe_json_parse(text2, where=f"kyber {url2}")
                    out_amt = None
                    d2 = data2.get("data") if isinstance(data2, dict) else None
                    if isinstance(d2, dict) and "routeSummary" in d2 and "amountOut" in d2["routeSummary"]:
                        out_amt = int(d2["routeSummary"]["amountOut"])
                    if out_amt is None and "route" in data2 and "amountOut" in data2["route"]:
                        out_amt = int(data2["route"]["amountOut"])
                    if out_amt is not None:
                        return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                           self.name, out_amt, b_token.decimals, LEG_OK)
                except Exception as e:
                    return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                        self.name, 0, None, LEG_ERROR, str(e),
                                        a_token, b_token)

            body = (text or text2 or "")[:200]
            return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                self.name, 0, None, LEG_NO_QUOTE,
                                f"HTTP {code or code2}", a_token, b_token)

        except Exception as e:
            return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                self.name, 0, None, LEG_ERROR, str(e),
                                a_token, b_token)

class OdosAdapter(BaseAdapter):
    name = "odos"
    
    async def quote(self, chain: Any, a_sym: str, b_sym: str, base_in: float,
                   a_token: Optional[Token] = None, b_token: Optional[Token] = None) -> LegQuote:
        """Get quote from Odos"""
        chain_name = getattr(chain, "name", "?")
        
        logger.info(f"[DEBUG] OdosAdapter input: base_in={base_in}, decimals={a_token.decimals}")


        if not self.supports_chain(chain_name):
            return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                self.name, 0, None, LEG_NO_QUOTE, "adapter disabled on chain",
                                a_token, b_token)

        if not (a_token and b_token):
            return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                self.name, 0, None, LEG_NO_ADDR, "missing token objects",
                                a_token, b_token)

        try:
            # Convert input amount to wei
            in_wei = int(float(base_in) * (10 ** a_token.decimals))
            logger.info(f"[DEBUG] OdosAdapter in_wei: {in_wei}")
            
            # Prepare API request
            base = self.app.api.get("odos_base", "https://api.odos.xyz").rstrip("/")
            url = f"{base}/sor/quote/v2"
            payload = {
                "chainId": chain.id,
                "inputTokens": [{"tokenAddress": a_token.address, "amount": str(in_wei)}],
                "outputTokens": [{"tokenAddress": b_token.address}],
                "userAddr": self.app.api.get("odos_user_addr", "0x0000000000000000000000000000000000000001"),
                "slippageLimitPercent": float(self.app.settings.get("slippage_limit_percent", 0.5)),
                "compact": True
            }
            
            code, text, _ = await self._fetch_http("POST", url, json_body=payload)
            if code != 200:
                return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                    self.name, 0, None, LEG_NO_QUOTE,
                                    f"HTTP {code}", a_token, b_token)

            data = self._safe_json_parse(text, where=f"odos {url}")
            out_amt = None
            
            # Parse response
            if isinstance(data, dict):
                if "outAmounts" in data and isinstance(data["outAmounts"], list) and data["outAmounts"]:
                    out_amt = int(data["outAmounts"][0])
                elif "outAmount" in data:
                    out_amt = int(data["outAmount"])
                elif "outputTokens" in data and data["outputTokens"] and "amount" in data["outputTokens"][0]:
                    out_amt = int(data["outputTokens"][0]["amount"])

            if out_amt is not None:
                # Add debug log for output amount
                out_base = out_amt / (10 ** b_token.decimals)
                logger.info(f"[DEBUG] OdosAdapter output: out_wei={out_amt}, out_base={out_base}, decimals={b_token.decimals}")
                return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                    self.name, out_amt, b_token.decimals, LEG_OK)

            return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                self.name, 0, None, LEG_NO_QUOTE,
                                "missing outAmount", a_token, b_token)

        except Exception as e:
            return self._make_leg(chain_name, a_sym, b_sym, base_in,
                                self.name, 0, None, LEG_ERROR, str(e),
                                a_token, b_token)


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

async def best_quote_leg(
    app: Any,
    chain: Any,
    a_sym: str,
    b_sym: str,
    base_in: float,
    adapters: List["BaseAdapter"],
    a_token: Token,  # Added Token object parameter
    b_token: Token   # Added Token object parameter
) -> LegQuote:
    """Get best quote from multiple adapters
    
    Args:
        app: App config
        chain: Chain object
        a_sym: Token A symbol
        b_sym: Token B symbol 
        base_in: Input amount
        adapters: List of adapter instances
        a_token: Token A object
        b_token: Token B object
    """
    if not adapters:
        return LegQuote(
            chain=chain.name,
            a=a_sym,
            b=b_sym,
            base_in=base_in,
            in_wei=None,
            out_b=None,
            out_wei=0,
            adapter="-",
            status=LEG_NO_QUOTE,
            note="no adapters",
            a_token=a_token,  # Add Token objects
            b_token=b_token
        )

    # 并发查询所有适配器
    quotes = await asyncio.gather(*[
        a.quote(chain, a_sym, b_sym, base_in, a_token, b_token)
        for a in adapters
        if a.supports_chain(chain.name)
    ])
    
    # 找出最佳报价
    best = None
    for q in quotes:
        if q.status != LEG_OK:
            continue
        if not best or q.out_b > best.out_b:
            best = q
            
    return best or LegQuote(
        chain=chain.name,
        a=a_sym,
        b=b_sym,
        base_in=base_in,
        in_wei=None,
        out_b=None,
        out_wei=0,
        adapter="-",
        status=LEG_NO_QUOTE,
        note="no quotes"
    )
