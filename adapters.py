# adapters.py
from __future__ import annotations
from typing import Optional, Any
from leg_quote import LegQuote, LEG_OK, LEG_NO_ADDR, LEG_NO_QUOTE, LEG_ERROR
from assets import Token
from logger_utils import NullLogger, SafeDebugLogger
import json

class BaseAdapter:
    """
    标准适配器基类：
      - name: 适配器名（如 "kyber", "odos"）
      - supports_chain(): 可由 settings.adapter_support[name] 控制
      - quote(): 统一返回 LegQuote
    """
    name: Optional[str] = None  # 适配器名，由子类覆盖

    def __init__(self, app: Any, dbg: SafeDebugLogger | NullLogger):
        self.app = app
        self.logger = dbg

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

    async def quote(self, leg: LegQuote) -> LegQuote:
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

class KyberAdapter(BaseAdapter):
    name = "kyber"
    
    async def quote(self, leg: LegQuote) -> LegQuote:
        """Get quote from Kyber"""
        leg.adapter = self.name

        if not (leg.a and leg.b):
            leg.status = LEG_NO_ADDR
            leg.note = f"Missing token objects for {leg.a.name} or {leg.b.name}"
            return leg

        try:
            base = self.app.api.get("kyber_base", "https://aggregator-api.kyberswap.com").rstrip("/")
            qs = {
                "tokenIn": leg.a.address,
                "tokenOut": leg.b.address,
                "amountIn": str(leg.in_wei),
                "gasInclude": "1",
                "saveGas": "0",
                "excludeSources": "liquidation"
            }
            
            chain_path = getattr(leg.chain, "kyber_slug", "") or leg.chain.name.lower()
            self.logger.info(f"[DEBUG] KyberAdapter using chain_path: {chain_path}") 
            url = f"{base}/{chain_path}/api/v1/routes"
            self.logger.info(f"[DEBUG] KyberAdapter full URL: {url}")
            self.logger.info(f"[DEBUG] KyberAdapter querying: {url}")
            self.logger.info(f"[DEBUG] KyberAdapter params: {qs}")
            
            code, text, headers = await self._fetch_http("GET", url, params=qs)
            
            if code == 200 and text:
                data = self._safe_json_parse(text, where=f"kyber {url}")
                self.logger.info(f"[DEBUG] KyberAdapter response: {data}")
                
                if data.get("code") == 0 and "data" in data:
                    route_summary = data["data"].get("routeSummary", {})
                    if "amountOut" in route_summary:
                        out_amt = int(route_summary["amountOut"])
                        out_base = out_amt / (10 ** leg.b.decimals)
                        self.logger.info(f"[DEBUG] KyberAdapter quote: {out_amt} wei -> {out_base} {leg.b.name}")
                        
                        leg.out_wei = out_amt
                        leg.out_b = out_base
                        leg.status = LEG_OK
                        return leg
            
            leg.status = LEG_NO_QUOTE
            leg.note = f"No valid quote received (HTTP {code})"
            return leg

        except Exception as e:
            self.logger.error(f"[ERROR] KyberAdapter error: {str(e)}")
            leg.status = LEG_ERROR
            leg.note = str(e)
            return leg

class OdosAdapter(BaseAdapter):
    name = "odos"
    
    async def quote(self, leg: LegQuote) -> LegQuote:
        """Get quote from Odos"""
        leg.adapter = self.name
        
        if not self.supports_chain(leg.chain.name):
            leg.status = LEG_NO_QUOTE
            leg.note = f"Adapter disabled on chain {leg.chain.name}"
            return leg

        if not (leg.a and leg.b):
            leg.status = LEG_NO_ADDR
            leg.note = f"Missing token objects for {leg.a.name} or {leg.b.name}"
            return leg

        try:
            # Remove redundant conversion since in_wei is already set
            self.logger.info(f"[DEBUG] OdosAdapter using in_wei: {leg.in_wei}")
            
            # Prepare API request
            base = self.app.api.get("odos_base", "https://api.odos.xyz").rstrip("/")
            url = f"{base}/sor/quote/v2"
            
            payload = {
                "chainId": leg.chain.id,
                "inputTokens": [{"tokenAddress": leg.a.address, "amount": str(leg.in_wei)}],
                "outputTokens": [{"tokenAddress": leg.b.address}],
                "userAddr": self.app.api.get("odos_user_addr", "0x0000000000000000000000000000000000000001"),
                "slippageLimitPercent": float(self.app.settings.get("slippage_limit_percent", 0.5)),
                "compact": True
            }
            
            self.logger.info(f"[DEBUG] OdosAdapter querying: {url}")
            self.logger.info(f"[DEBUG] OdosAdapter payload: {payload}")

            code, text, _ = await self._fetch_http("POST", url, json_body=payload)
            self.logger.info(f"[DEBUG] OdosAdapter response code: {code}")
            self.logger.info(f"[DEBUG] OdosAdapter response text: {text[:200]}")

            if code == 200 and text:
                data = self._safe_json_parse(text, where=f"odos {url}")
                self.logger.info(f"[DEBUG] OdosAdapter parsed data: {json.dumps(data)[:200]}")

                out_amt = None
                if isinstance(data, dict):
                    # Try all possible paths for output amount
                    if "outAmounts" in data and isinstance(data["outAmounts"], list) and data["outAmounts"]:
                        out_amt = int(data["outAmounts"][0])
                        self.logger.info(f"[DEBUG] Found amount in outAmounts: {out_amt}")
                    elif "outAmount" in data:
                        out_amt = int(data["outAmount"])
                        self.logger.info(f"[DEBUG] Found amount in outAmount: {out_amt}")
                    elif "outputTokens" in data and data["outputTokens"] and "amount" in data["outputTokens"][0]:
                        out_amt = int(data["outputTokens"][0]["amount"])
                        self.logger.info(f"[DEBUG] Found amount in outputTokens: {out_amt}")

                if out_amt is not None:
                    out_base = out_amt / (10 ** leg.b.decimals)
                    self.logger.info(f"[DEBUG] OdosAdapter quote: {out_amt} wei -> {out_base} {leg.b.name}")

                    leg.out_wei = out_amt
                    leg.out_b = out_base
                    leg.status = LEG_OK
                    return leg
                else:
                    self.logger.error(f"[DEBUG] No output amount found in response")
            
            leg.status = LEG_NO_QUOTE
            leg.note = f"No valid quote received (HTTP {code})"
            return leg

        except Exception as e:
            self.logger.error(f"[ERROR] OdosAdapter error: {str(e)}")
            leg.status = LEG_ERROR
            leg.note = str(e)
            return leg

