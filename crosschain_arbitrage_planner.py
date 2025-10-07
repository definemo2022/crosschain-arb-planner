# 1. Imports
from __future__ import annotations
import os, json, time, argparse, asyncio, traceback
from dataclasses import dataclass
from typing import Dict, Any,  List, Optional
from size_optimizer import OptimizeConfig, optimize_size_async
from leg_quote import LegQuote, TwoLegResult, LEG_OK, LEG_NO_ADDR, LEG_NO_QUOTE, LEG_ERROR
from adapters import build_adapters, best_quote_leg
from logger_utils import init_debug_logger, NullLogger
from chains import load_chains_from_config, Chains

@dataclass
class Chain:
    name: str
    chain_id: int
    kyber_slug: str

@dataclass
class Asset:
    symbol: str
    decimals: int
    address: Dict[str, str]  # chain_name -> addr

@dataclass
class AppConfig:
    api: Dict[str, Any]
    chains: List[Chain]  # Changed from Dict[str, Chain] to List[Chain]
    assets: Dict[str, Asset]
    settings: Dict[str, Any]


# 2. Global variables
DBG = NullLogger()

# Add a global counter
QUOTE_COUNTER = 0

# ---- 基础资产：配置与 CLI 覆盖 ----
CLI_BASE_ASSETS: set[str] = set()

def _init_debug(app, out_html: str, verbose: bool):
    """统一初始化全局 DBG；永远返回一个可用的 logger。"""
    global DBG
    DBG = init_debug_logger(app, out_html, verbose)

# 3. Helper functions
def _ensure_dir(path: str):
    """Directory creation helper"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _fmt(x, decimals: int = 6, dash: str = "—") -> str:
    """把数字格式化为千分位；x 为 None/无效时显示 dash。"""
    if x is None:
        return dash
    try:
        xf = float(x)
    except (TypeError, ValueError):
        # 已是字符串或非数值，直接返回
        return str(x)
    return f"{xf:,.{decimals}f}"

def _now_str() -> str:
    """Time string helper"""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _write_placeholder_html(path: str, title: str, refresh: int):
    _ensure_dir(path)
    html = f"""<!doctype html><html><head>
<meta charset="utf-8"><meta http-equiv="refresh" content="{refresh}">
<title>{title}</title>
<style>body{{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,'PingFang SC','Microsoft Yahei';margin:20px;color:#0f172a}}</style>
</head><body>
<h2>{title}</h2>
<p>Waiting first quotes… (auto refresh {refresh}s)</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

def _write_html(path: str, html: str):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

def _safe_json_parse(resp_text: str, *, where: str) -> Any:
    try:
        return json.loads(resp_text)
    except Exception as e:
        snippet = (resp_text or "")[:400].replace("\n", "\\n")
        raise ValueError(f"{where}: non-JSON or empty body. body[:400]={snippet!r}") from e

def _iter_asset_pairs(app: AppConfig, symbols: List[str]):
    """
    产出 (s, b) 的有序资产对，满足：
      - s != b
      - 通过基础资产规则（只允许 基础 ↔ 非基础）
    """
    base_set = _base_assets(app)  # 例如 {"USDT","USDC"}
    for s in symbols:
        for b in symbols:
            if s == b:
                continue
            if _skip_pair_by_base(base_set, s, b):
                continue
            yield (s, b)

def _from_wei(amount: int, decimals: int) -> float:
    return float(amount) / (10 ** decimals)

def _parse_two_leg_spec(pair_spec: str):
    """
    解析形如: 'chainA:A>B,chainB:B>A'，返回 (ci_name, a_sym, b_sym, cj_name)
    """
    if not pair_spec or "," not in pair_spec:
        raise ValueError(f"two-leg 需要 'chainA:A>B,chainB:B>A'，当前: {pair_spec!r}")
    part1, part2 = [x.strip() for x in pair_spec.split(",", 1)]
    if ":" not in part1 or ">" not in part1 or ":" not in part2 or ">" not in part2:
        raise ValueError(f"two-leg 需要 'chainA:A>B,chainB:B>A'，当前: {pair_spec!r}")

    ci_name, l1 = [x.strip() for x in part1.split(":", 1)]
    a1, b1 = [x.strip().upper() for x in l1.split(">", 1)]

    cj_name, l2 = [x.strip() for x in part2.split(":", 1)]
    b2, a2 = [x.strip().upper() for x in l2.split(">", 1)]

    if a1 != a2 or b1 != b1:
        # 要求第二腿正好是 B>A
        raise ValueError(f"two-leg 第二腿应为 {b1}>{a1}，当前: {b2}>{a2}")
    return ci_name, a1, b1, cj_name

# 从 settings 或 assets 自动得到“资产宇宙”
def _asset_universe(app, universe_str: str | None = None) -> list[str]:
    """
    返回要遍历的资产符号列表（大写），仅保留 app.assets 里存在的符号。
    优先级：CLI 传入的 universe_str > settings.universe > assets 全量
    - settings.universe 支持 list 或 逗号字符串
    """
    if universe_str:
        symbols = [x.strip().upper() for x in universe_str.split(",") if x.strip()]
    else:
        uni = app.settings.get("universe", None)
        if isinstance(uni, list):
            symbols = [str(x).strip().upper() for x in uni]
        elif isinstance(uni, str):
            symbols = [x.strip().upper() for x in uni.split(",") if x.strip()]
        else:
            symbols = list(app.assets.keys())
    return [s for s in symbols if s in app.assets]

# =============== 适配器白名单 ===============
def _adapter_allows_chain(app: AppConfig, adapter: str, chain: Chain) -> bool:
    """
    settings.adapter_support[adapter] 若配置了白名单，则只在白名单内的链上请求；
    未配置则默认允许所有链。支持链名、kyber_slug、或 chain_id（int/字符串）。
    """
    sup = app.settings.get("adapter_support", {})
    allow = sup.get(adapter)
    if not allow:
        return True
    tokens = {chain.name.lower(), str(chain.chain_id), chain.kyber_slug.lower()}
    for item in allow:
        if isinstance(item, int) and item == chain.chain_id:
            return True
        if isinstance(item, str) and item.lower() in tokens:
            return True
    return False

def _parse_base_assets_arg(s: str | None) -> set[str]:
    if not s:
        return set()
    return {x.strip().upper() for x in s.split(",") if x.strip()}

def _base_assets(app: AppConfig) -> set[str]:
    # CLI 优先；否则读 config.json settings.base_assets
    if CLI_BASE_ASSETS:
        return {x for x in CLI_BASE_ASSETS if x in app.assets}
    cfg = app.settings.get("base_assets", [])
    if not isinstance(cfg, (list, tuple)):
        return set()
    return {str(x).strip().upper() for x in cfg if str(x).strip().upper() in app.assets}

def _skip_pair_by_base(base_set: set[str], a_sym: str, b_sym: str) -> bool:
    """
    仅允许一端在基础资产集合、另一端不在（XOR）：
      - 基础↔基础：跳过
      - 非基础↔非基础：跳过
      - 基础↔非基础：允许
    """
    a_in = a_sym in base_set
    b_in = b_sym in base_set
    # 允许条件是异或（恰好一端在基础集合）
    allow = (a_in ^ b_in)
    return not allow  # 返回 True=跳过

def find_chain(chains: List[Chain], name: str) -> Optional[Chain]:
    """Helper to find chain by name"""
    name = name.lower()
    return next((c for c in chains if c.name.lower() == name), None)

async def _process_watchlist(app: AppConfig, pairs: list[dict], adapters) -> List[TwoLegResult]:
    """处理 watchlist 中的交易对"""
    results = []
    DBG.info(f"[CHECK] Processing {len(pairs)} pairs")

    for i, pair in enumerate(pairs):
        from_chain = pair["from_chain"]
        to_chain = pair["to_chain"]
        token_a = pair["a"]
        token_b = pair["b"]
        base = pair.get("base")

        # Find chains using helper
        from_chain_obj = find_chain(app.chains, from_chain)
        to_chain_obj = find_chain(app.chains, to_chain)
        
        if not from_chain_obj or not to_chain_obj:
            DBG.error(f"Chain not found: {from_chain} or {to_chain}")
            continue

        DBG.info(f"[CHECK] Processing pair {i+1}: {from_chain}:{token_a} > {to_chain}:{token_b}")
        try:
            result = await quote_two_leg_once(
                app=app,
                ci=from_chain_obj,
                cj=to_chain_obj,
                a_sym=token_a,
                b_sym=token_b,
                base_in_a=base
            )
            DBG.info(f"[CHECK] Pair {i+1} result status: {result.status}")
            results.append(result)
        except Exception as e:
            DBG.error(f"Error processing pair {i+1} {pair}: {e}")
            continue

    DBG.info(f"[CHECK] Finished processing all pairs, got {len(results)} results")
    return results

# 4. Core functions
# 全局变量定义
CHAINS: Optional[Chains] = None

def load_config(config_path: str) -> AppConfig:
    """加载配置文件"""
    global CHAINS
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            
        # 初始化全局 CHAINS
        CHAINS = load_chains_from_config(raw)
        DBG.info(f"Loaded {len(CHAINS)} chains")
        
        assets = {a["symbol"].upper(): Asset(a["symbol"].upper(), 
                                           int(a["decimals"]), 
                                           {k.lower(): v for k, v in a["address"].items()}) 
                 for a in raw["assets"]}
        
        return AppConfig(
            api=raw.get("api", {}),  # Added api parameter
            chains=list(CHAINS),
            assets=assets,
            settings=raw.get("settings", {})
        )
    except Exception as e:
        DBG.error(f"Failed to load config: {e}")
        raise

def _load_watchlist_json(path: str) -> list[dict]:
    """读取 watchlist.json，返回条目列表"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"watchlist file not found: {path}")
    except Exception as e:
        raise ValueError(f"bad watchlist json: {e}")

    if not isinstance(data, list):
        raise ValueError("watchlist must be a JSON array")

    norm_list = []
    for i, it in enumerate(data):
        if not isinstance(it, dict):
            DBG.warn("[watchlist] item %d not a dict, skip", i)
            continue

        try:
            pair_spec = {
                "from_chain": str(it["A_chain"]).lower(),
                "to_chain": str(it["B_chain"]).lower(),
                "a": str(it["A"]).upper(),
                "b": str(it["B"]).upper(),
                "base": float(it["base"]) if "base" in it else None
            }
            norm_list.append(pair_spec)
        except (KeyError, ValueError) as e:
            DBG.warn("[watchlist] item %d parse error, skip: %s", i, e)
            continue

    return norm_list

async def _fetch_http(method: str, url: str, *, headers=None, params=None, json_body=None, timeout=12):
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
        # 统一吞掉网络异常，返回“不可用”给上层
        return -1, f"__EXC__:{type(e).__name__}:{str(e)[:180]}", {}

async def kyber_quote(app: AppConfig, chain: Chain, sell_addr: str, buy_addr: str, amount_wei: int) -> Dict[str, Any]:
    """Get quote from Kyber
    
    If chain.kyber_slug is not available, falls back to chain.name
    """
    base = app.api.get("kyber_base", "https://aggregator-api.kyberswap.com").rstrip("/")
    qs = {
        "tokenIn": sell_addr,
        "tokenOut": buy_addr,
        "amountIn": str(amount_wei),
        "gasInclude": "1",
        "saveGas": "0"
    }
    
    # Get kyber_slug or fall back to chain name
    chain_path = getattr(chain, "kyber_slug", None) or chain.name.lower()
    
    # Try routes endpoint first
    url1 = f"{base}/{chain_path}/api/v1/routes"
    code, text, headers = await _fetch_http("GET", url1, params=qs)
    if code == 200 and text:
        try:
            data = _safe_json_parse(text, where=f"kyber {url1}")
            out_amt = None
            d = data.get("data") if isinstance(data, dict) else None
            if isinstance(d, dict):
                rs = d.get("routeSummary")
                if isinstance(rs, dict) and "amountOut" in rs:
                    out_amt = int(rs["amountOut"])
            if out_amt is None and "route" in data and "amountOut" in data["route"]:
                out_amt = int(data["route"]["amountOut"])
            if out_amt is not None:
                return {"ok": True, "adapter": "kyber", "out_wei": out_amt, "raw": data}
        except Exception:
            pass
    url2 = f"{base}/{chain.kyber_slug}/api/v1/route"
    code2, text2, headers2 = await _fetch_http("GET", url2, params=qs, timeout=int(app.settings.get("timeout_sec", 12)))
    if code2 == 200 and text2:
        try:
            data2 = _safe_json_parse(text2, where=f"kyber {url2}")
            out_amt = None
            d2 = data2.get("data") if isinstance(data2, dict) else None
            if isinstance(d2, dict) and "routeSummary" in d2 and "amountOut" in d2["routeSummary"]:
                out_amt = int(d2["routeSummary"]["amountOut"])
            if out_amt is None and "route" in data2 and "amountOut" in data2["route"]:
                out_amt = int(data2["route"]["amountOut"])
            if out_amt is not None:
                return {"ok": True, "adapter": "kyber", "out_wei": out_amt, "raw": data2}
        except Exception as e:
            return {"ok": False, "adapter": "kyber", "err": str(e)}
    body = (text or text2 or "")[:200]
    return {"ok": False, "adapter": "kyber", "err": f"HTTP {code or code2}", "where": f"GET {url1} / {url2}", "body": body}

async def odos_quote(app: AppConfig, chain: Chain, sell_addr: str, buy_addr: str, amount_wei: int) -> Dict[str, Any]:
    base = app.api.get("odos_base", "https://api.odos.xyz").rstrip("/")
    url = f"{base}/sor/quote/v2"
    payload = {
        "chainId": chain.id,  # Changed from chain.chain_id to chain.id
        "inputTokens": [{"tokenAddress": sell_addr, "amount": str(amount_wei)}],
        "outputTokens": [{"tokenAddress": buy_addr}],
        "userAddr": app.api.get("odos_user_addr", "0x0000000000000000000000000000000000000001"),
        "slippageLimitPercent": float(app.settings.get("slippage_limit_percent", 0.5)),
        "compact": True
    }
    code, text, headers = await _fetch_http("POST", url, json_body=payload, timeout=int(app.settings.get("timeout_sec", 12)))
    if code != 200:
        return {"ok": False, "adapter": "odos", "err": f"HTTP {code}", "where": f"POST {url}", "body": (text or "")[:200]}
    try:
        data = _safe_json_parse(text, where=f"odos {url}")
    except Exception as e:
        return {"ok": False, "adapter": "odos", "err": str(e)}
    out_amt = None
    if isinstance(data, dict):
        if "outAmounts" in data and isinstance(data["outAmounts"], list) and data["outAmounts"]:
            out_amt = int(data["outAmounts"][0])
        elif "outAmount" in data:
            out_amt = int(data["outAmount"])
        elif "outputTokens" in data and data["outputTokens"] and "amount" in data["outputTokens"][0]:
            out_amt = int(data["outputTokens"][0]["amount"])
    if out_amt is None:
        return {"ok": False, "adapter": "odos", "err": "missing outAmount", "data_keys": list(data.keys())}
    return {"ok": True, "adapter": "odos", "out_wei": out_amt, "raw": data}

async def quote_basic_leg(app, chain, a_sym, b_sym, base_in: float) -> "LegQuote":
    """Get basic quote from adapters for a token pair on a single chain.
    
    Args:
        app: Application config
        chain: Chain configuration 
        a_sym: Input token symbol
        b_sym: Output token symbol
        base_in: Input amount
        
    Returns:
        LegQuote: Quote result from best adapter
    """
    # 由主程序把现有函数指针注入适配器，避免循环 import
    qmap = {
        "kyber": kyber_quote if "kyber_quote" in globals() else None,
        "odos":  odos_quote  if "odos_quote"  in globals() else None,
    }
    adapters = build_adapters(app, qmap)
    # 交给聚合器并发对比，拿到标准 LegQuote
    leg = await best_quote_leg(app, chain, str(a_sym).upper(), str(b_sym).upper(),
                               float(base_in), adapters)
    return leg

async def _quote_two_leg(app, ci, cj, a_sym: str, b_sym: str, base_in_a: float) -> TwoLegResult:
    """获取两条腿的报价

    Args:
        app: 应用配置
        ci: 第一条腿的链
        cj: 第二条腿的链
        a_sym: A 代币符号
        b_sym: B 代币符号
        base_in_a: 初始 A 代币数量

    Returns:
        TwoLegResult: 两条腿的报价结果
    """
    try:
        # 第一条腿: ci 上 A->B
        leg1 = await quote_basic_leg(app, ci, a_sym, b_sym, base_in_a)
        if leg1.status != LEG_OK:
            return TwoLegResult(
                from_chain=ci.name, 
                to_chain=cj.name,
                a=a_sym, 
                b=b_sym,
                base_in_a=base_in_a,
                leg1=leg1,
                leg2=None,
                final_a=None,
                pnl=None,
                pnl_pct=None,
                status=leg1.status,
                note=f"Leg1 failed: {leg1.note}"
            )

        # 第二条腿: cj 上 B->A
        leg2 = await quote_basic_leg(app, cj, b_sym, a_sym, leg1.out_b)
        if leg2.status != LEG_OK:
            return TwoLegResult(
                from_chain=ci.name,
                to_chain=cj.name,
                a=a_sym,
                b=b_sym,
                base_in_a=base_in_a,
                leg1=leg1,
                leg2=leg2,
                final_a=None,
                pnl=None,
                pnl_pct=None,
                status=leg2.status,
                note=f"Leg2 failed: {leg2.note}"
            )

        # 计算收益
        final_a = leg2.out_b
        pnl = final_a - base_in_a
        pnl_pct = (pnl / base_in_a) * 100 if base_in_a else None

        return TwoLegResult(
            from_chain=ci.name,
            to_chain=cj.name,
            a=a_sym,
            b=b_sym,
            base_in_a=base_in_a,
            leg1=leg1,
            leg2=leg2,
            final_a=final_a,
            pnl=pnl,
            pnl_pct=pnl_pct,
            status=LEG_OK,
            note=""
        )

    except Exception as e:
        return TwoLegResult(
            from_chain=ci.name,
            to_chain=cj.name,
            a=a_sym,
            b=b_sym,
            base_in_a=base_in_a,
            leg1=None,
            leg2=None,
            final_a=None,
            pnl=None,
            pnl_pct=None,
            status=LEG_ERROR,
            note=f"Quote error: {str(e)}"
        )

async def quote_two_leg_once(app, ci, cj, a_sym: str, b_sym: str, base_in_a: float) -> TwoLegResult:
    """Quote two legs with optimization

    Args:
        app: 应用配置
        ci: 第一条腿的链
        cj: 第二条腿的链
        a_sym: A 代币符号
        b_sym: B 代币符号
        base_in_a: 初始 A 代币数量

    Returns:
        TwoLegResult: 两条腿的报价结果
    """
    try:
        global QUOTE_COUNTER
        QUOTE_COUNTER = 0
        
        # 获取初始报价
        result = await _quote_two_leg(app, ci, cj, a_sym, b_sym, base_in_a)
        QUOTE_COUNTER += 1
        
        if result.status == LEG_OK and result.pnl is not None and result.pnl > 0:
            DBG.info(f"[OPTIMIZE] Found profitable trade, optimizing size. Initial pnl: {result.pnl}")
            
            # 定义优化目标函数
            async def try_size(size: float) -> float:
                global QUOTE_COUNTER
                QUOTE_COUNTER += 1
                r = await _quote_two_leg(app, ci, cj, a_sym, b_sym, size)
                DBG.info(f"[COUNT] Optimization quote #{QUOTE_COUNTER}: size={size:.2f}, pnl={r.pnl or 0:.2f}")
                return float(r.pnl or 0)
            
            # 仅使用支持的参数
            opt_result = await optimize_size_async(
                try_size,
                OptimizeConfig(
                    initial_size=base_in_a,
                    min_size=base_in_a * 0.5,
                    max_size=base_in_a * 10,
                    method="brent"
                )
            )
            
            if opt_result.best_pnl > result.pnl:
                result = await _quote_two_leg(app, ci, cj, a_sym, b_sym, opt_result.best_size)
                QUOTE_COUNTER += 1
                
        return result
        
    except Exception as e:
        DBG.error(f"Error in quote_two_leg_once: {e}")
        return TwoLegResult(
            from_chain=ci.name,
            to_chain=cj.name,
            a=a_sym,
            b=b_sym,
            base_in_a=base_in_a,
            leg1=None,
            leg2=None,
            final_a=None,
            pnl=None,
            pnl_pct=None,
            status=LEG_ERROR,
            note=f"Quote error: {str(e)}"
        )

# 5. Mode functions
async def run_one_leg_mode(app: AppConfig, pair_spec: str, out_html: str, refresh: int, verbose: bool, once: bool, pair_any: str):
    """运行单腿报价模式：在单一链上查询代币对的报价"""
    _ensure_dir(os.path.dirname(out_html))
    _write_placeholder_html(out_html, "Single Leg Mode", refresh)
    
    global DBG
    DBG = init_debug_logger("single-leg", out_html, verbose)
    
    while True:
        try:
            # 获取代币对信息
            if pair_any:
                if ">" not in pair_any:
                    raise ValueError(f"pair-any 格式应为 'A>B'，当前: {pair_any}")
                a_sym, b_sym = [x.strip().upper() for x in pair_any.split(">")]
                chain_name = "ethereum"  # 默认在 ethereum 链上查询
            else:
                if ":" not in pair_spec or ">" not in pair_spec:
                    raise ValueError(f"pair-spec 格式应为 'chain:A>B'，当前: {pair_spec}")
                chain_name, pair = pair_spec.split(":", 1)
                a_sym, b_sym = [x.strip().upper() for x in pair.split(">")]
            
            # 获取链配置
            chain = find_chain(app.chains, chain_name)
            if not chain:
                raise ValueError(f"Chain not found: {chain_name}")
            
            # 获取报价
            leg = await quote_basic_leg(app, chain, a_sym, b_sym, 1000)  # 使用1000作为基础数量
            
            # 渲染结果
            html = render_legs_page(
                title=f"Single Leg Quote: {chain_name}:{a_sym}>{b_sym}",
                refresh=refresh,
                legs=[leg]
            )
            _write_html(out_html, html)
            
            if once:
                break
                
            await asyncio.sleep(refresh)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            DBG.error(f"Error in single leg mode: {e}")
            if once:
                break
            await asyncio.sleep(refresh)

async def run_two_leg_mode(app: AppConfig, pair_spec: str, out_html: str, refresh: int, verbose: bool, once: bool):
    """运行两腿套利模式，分析跨链套利机会"""
    _ensure_dir(os.path.dirname(out_html))
    _write_placeholder_html(out_html, "Two-Leg Mode", refresh)
    
    global DBG
    DBG = init_debug_logger("two-leg", out_html, verbose)
    DBG.info("[CHECK] Starting two-leg mode")
    
    while True:
        try:
            # 解析交易对规格
            try:
                ci_name, a_sym, b_sym, cj_name = _parse_two_leg_spec(pair_spec)
                DBG.info(f"[CHECK] Parsed pair spec: {ci_name}:{a_sym}>{b_sym} -> {cj_name}:{b_sym}>{a_sym}")
            except ValueError as e:
                DBG.error(f"Invalid pair spec: {e}")
                break

            # 获取链配置
            ci = find_chain(app.chains, ci_name)
            cj = find_chain(app.chains, cj_name)
            
            if not ci or not cj:
                raise ValueError(f"Chain not found: {ci_name} or {cj_name}")
            
            DBG.info(f"[CHECK] Got chain configs: {ci.name} and {cj.name}")
            
            # 使用初始数量查询报价（默认1000）
            base_in_a = 1000.0
            DBG.info(f"[CHECK] Quoting {ci_name}:{a_sym}>{b_sym} -> {cj_name}:{b_sym}>{a_sym} with base {base_in_a}")
            
            # 获取报价并优化数量
            result = await quote_two_leg_once(
                app=app,
                ci=ci,
                cj=cj,
                a_sym=a_sym,
                b_sym=b_sym,
                base_in_a=base_in_a
            )
            DBG.info(f"[CHECK] Got quote result with status: {result.status}")
            
            # 渲染结果页面
            if result.leg1 and result.leg2:
                DBG.info("[CHECK] Both legs available, rendering page")
                html = render_legs_page(
                    title=f"Two-Leg Arbitrage: {ci_name}:{a_sym}>{b_sym} -> {cj_name}:{b_sym}>{a_sym}",
                    refresh=refresh,
                    legs=[result.leg1, result.leg2]
                )
                DBG.info("[CHECK] Page rendered, writing to file")
                _write_html(out_html, html)
                
                # 显示套利结果
                if result.status == LEG_OK:
                    if result.pnl and result.pnl > 0:
                        DBG.info(f"Found arbitrage opportunity! PnL: {_fmt(result.pnl)} {a_sym} ({_fmt(result.pnl_pct, 2)}%)")
                    else:
                        DBG.info(f"No arbitrage opportunity. PnL: {_fmt(result.pnl)} {a_sym}")
                else:
                    DBG.error(f"Quote failed: {result.status} - {result.note}")
            else:
                DBG.error("[CHECK] Missing legs in result")
            
            if once:
                break
                
            await asyncio.sleep(refresh)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            DBG.error(f"Error in two-leg mode: {e}\n{traceback.format_exc()}")
            if once:
                break
            await asyncio.sleep(refresh)

async def run_watchlist_mode(
    app: AppConfig,    # Changed from str to AppConfig
    watchlist_path: str,
    out_html: str,
    refresh: int,
    once: bool,
    verbose: bool = True
) -> None:
    """运行 watchlist 模式"""
    _ensure_dir(os.path.dirname(out_html))
    _write_placeholder_html(out_html, "Watchlist Mode", refresh)
    
    global DBG
    DBG = init_debug_logger("watchlist", out_html, verbose)
    
    pairs = _load_watchlist_json(watchlist_path)
    DBG.info(f"[CHECK] Loaded pairs from watchlist: {len(pairs)}")
    if not pairs:
        DBG.error("No valid pairs found in watchlist")
        return

    # 使用已有的 Kyber 和 Odos adapters
    quote_fns = {
        "kyber": kyber_quote,
        "odos": odos_quote
    }
    adapters = build_adapters(app, quote_fns)
    DBG.info(f"[CHECK] Built adapters: {[a.name for a in adapters]}")

    while True:
        try:
            results = await _process_watchlist(app, pairs, adapters)
            DBG.info(f"[CHECK] Processed results count: {len(results) if results else 0}")
            
            html = render_watchlist_page("Watchlist Mode", results, refresh, len(pairs), len(results))
            DBG.info("[CHECK] HTML rendered")
            _write_html(out_html, html)
            
            if once:
                break
                
            await asyncio.sleep(refresh)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            DBG.error(f"Error in watchlist mode: {e}")
            if once:
                break
            await asyncio.sleep(refresh)

# 5.1 render functions
def _render_page(title: str, refresh: int, rows: List[dict], debug_html: str = "", head_extra: str = "") -> str:
    esc = lambda s: (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    table_rows = []
    if rows:
        for r in rows:
            table_rows.append(
                f"<tr>"
                f"<td style='text-align:left'>{esc(r['leg'])} "
                f"<span style='color:#64748b'>(base {r['sell']})</span></td>"
                f"<td>{esc(r['adapter'])}</td>"
                f"<td>{r['out_str']}</td>"
                f"<td>{r['px_str']}</td>"
                f"</tr>"
            )
        body_html = f"""
        <table class="t">
          <thead><tr><th>Pair</th><th>Adapter</th><th>Out</th><th>Price</th></tr></thead>
          <tbody>{''.join(table_rows)}</tbody>
        </table>
        """
    else:
        body_html = "<div style='color:#64748b'>No available quotes right now.</div>"

    css = """
    body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,'PingFang SC','Microsoft YaHei';margin:20px;color:#0f172a}
    .t{border-collapse:collapse;width:100%} .t th,.t td{border:1px solid #e2e8f0;padding:8px}
    .t th{text-align:left;background:#f8fafc}
    """
    html = f"""<!doctype html><html><head>
<meta charset="utf-8"><meta http-equiv="refresh" content="{refresh}">
<title>{esc(title)}</title>
<style>{css}</style>
{head_extra}  <!-- 这里注入 JS 刷新等 -->
</head><body>
<h2>{esc(title)}</h2>
<div style="color:#64748b;margin:6px 0">Auto refresh {refresh}s · {esc(_now_str())}</div>
{debug_html}  <!-- 这里插入调试面板 -->
{body_html}
</body></html>"""
    return html

def render_legs_page(title: str, refresh: int, legs: list[LegQuote], head_extra: str = "") -> str:
    css = (
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,'PingFang SC','Microsoft YaHei';"
        "margin:20px;color:#0f172a}"
        ".t{border-collapse:collapse;width:100%}.t th,.t td{border:1px solid #e2e8f0;padding:8px}"
        ".t th{text-align:left;background:#f8fafc}"
        "h2{margin:0 0 12px 0}.sub{color:#475569;margin:0 0 16px 0}"
        ".ok{color:#059669}.warn{color:#b45309}.error{color:#b91c1c}.num{text-align:right}"
        ".muted{color:#64748b;font-size:12px;margin-left:6px}"
    )
    head_auto = head_extra or (DBG.js_auto_refresh(refresh) if hasattr(DBG, "js_auto_refresh") else "")

    rows_html = "".join([leg.as_html_row() for leg in (legs or [])])
    table = (
        "<table class='t'>"
        "<thead><tr>"
        "<th>Chain</th><th>A</th><th>B</th><th>Base (A)</th>"
        "<th>Adapter</th><th>Out (B)</th><th>Out (wei)</th>"
        "<th>Status</th><th>Note</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"{head_auto}"
        f"<title>{title}</title><style>{css}</style></head>"
        f"<body><h2>{title}</h2><div class='sub'>Auto-refresh: {int(refresh)}s</div>{table}</body></html>"
    )

def render_watchlist_page(title: str, results: List[TwoLegResult], refresh: int, total_pairs: int, processed_pairs: int) -> str:
    """渲染监控结果页面"""
    progress = f"Processing: {processed_pairs}/{total_pairs} pairs"
    
    css = """
    body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,'PingFang SC','Microsoft YaHei';
        margin:20px;color:#0f172a}
    .t{border-collapse:collapse;width:100%} .t th,.t td{border:1px solid #e2e8f0;padding:8px}
    .t th{text-align:left;background:#f8fafc}
    .profit{color:#059669} .loss{color:#dc2626}
    """
    
    rows = []
    for r in results:
        pnl_class = "profit" if r.pnl and r.pnl > 0 else "loss"
        rows.append(f"""
            <tr>
                <td>{r.from_chain}→{r.to_chain}</td>
                <td>{r.a}→{r.b}→{r.a}</td>
                <td>{_fmt(r.base_in_a)}</td>
                <td class="{pnl_class}">{_fmt(r.pnl)}</td>
                <td class="{pnl_class}">{_fmt(r.pnl_pct, 2)}%</td>
                <td>{r.status}</td>
                <td>{r.note}</td>
            </tr>
        """)

    table = f"""
    <table class="t">
        <thead>
            <tr>
                <th>Chains</th>
                <th>Path</th>
                <th>Base</th>
                <th>PnL</th>
                <th>PnL%</th>
                <th>Status</th>
                <th>Note</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """

    return f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="{refresh}">
        <title>{title}</title>
        <style>{css}</style>
    </head>
    <body>
        <h2>{title}</h2>
        <div style="color:#64748b;margin:6px 0">
            {progress} · Auto refresh {refresh}s · {_now_str()}
        </div>
        {table}
    </body>
    </html>
    """

# 6. Main function
def main():
    """Main entry point"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="开启调试面板并写日志（需要 logger_utils.py）")
    ap.add_argument("--log-file", required=False, help="日志文件路径；缺省为与HTML同名的 .log")
    ap.add_argument("--config", required=True)
    
    #watchList Mode
    ap.add_argument("--watchlist",type=str,help="Path to a watchlist.json (two-leg monitoring list). If set, run watchlist mode."
)    # 单腿“合并输出”模式
    ap.add_argument("--pair-spec", required=False,help='留空=自动遍历(USDT/USDC + USDT/USDE + 与 FRXUSD 双向); 例: "ethereum:USDT>USDE,plasma:USDT>USDE"')
    # 双腿遍历（单对）
    ap.add_argument("--two-leg",action="store_true",help="Enable two-leg mode (requires --pair-spec with two legs, e.g. ethereum:A>B,plasma:B>A)")
    ap.add_argument("--pair-html", required=True)
    ap.add_argument("--refresh", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--once", action="store_true")
    # ★ 新增：基础资产 CLI 覆盖
    ap.add_argument("--base-assets", required=False, help='用逗号分隔，例如 "USDT,USDC"；若提供则覆盖 config.json 的 settings.base_assets')
    ap.add_argument("--pair-any", required=False,help='只写资产对，自动遍历所有共同链，形如 "USDE>USDC"（大小写不敏感）')
    args = ap.parse_args()

    # Load config once at the start
    app_config = load_config(args.config)

    # ★ 保存 CLI 覆盖集合
    global CLI_BASE_ASSETS
    CLI_BASE_ASSETS = _parse_base_assets_arg(args.base_assets)

    # 让 --watchlist 的优先级高于 --two-leg/--pair-any
    if args.watchlist:
        asyncio.run(
            run_watchlist_mode(
                app_config,  # Pass AppConfig instead of config path
                args.watchlist,
                args.pair_html,
                args.refresh,
                args.once,
                args.verbose
            )
        )
        return

    if args.two_leg:
        # 两腿模式
        asyncio.run(run_two_leg_mode(
            app_config,  # Pass AppConfig
            args.pair_spec,
            args.pair_html,
            args.refresh,
            args.verbose,
            args.once
        ))
    elif args.pair_spec or args.pair_any:
        # 单腿模式
        asyncio.run(run_one_leg_mode(
            app_config,  # Pass AppConfig
            args.pair_spec,
            args.pair_html,
            args.refresh,
            args.verbose,
            args.once,
            args.pair_any
        ))
    else:
        ap.error("请选择 --two-leg 或 --pair-spec/--pair-any 之一")

# 7. Entry point
if __name__ == "__main__":
    main()





