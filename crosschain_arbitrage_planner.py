# 1. Imports
from __future__ import annotations
import os, json, time, argparse, asyncio, traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from size_optimizer import OptimizeConfig, optimize_size_async
from leg_quote import LegQuote, TwoLegResult, LEG_OK, LEG_NO_ADDR, LEG_NO_QUOTE, LEG_ERROR
from adapters import BaseAdapter, KyberAdapter, OdosAdapter
from logger_utils import init_debug_logger, NullLogger
from chains import load_chains_from_config, Chains, Chain
from assets import load_assets_from_config, Token, Tokens, Asset, Assets
from appconfig import AppConfig

# 2. Global variables
DBG = NullLogger()

MAINCONFIG:Optional[AppConfig]=None

# Add a global counter
QUOTE_COUNTER = 0
QUOTE_START_TIME = 0  # Add this

CONFIG_PATH = "./config.json"
WATCHLIST_PATH = "./watchlist.json"
OUT_HTML = "./html/output.html"

CHAINS: Optional[Chains] = None
ASSERTS: Optional[Assets] = None
SETTINGS: Dict[str, Any] = {}

ADAPTERS:List[BaseAdapter] = []

# ---- 基础资产：配置与 CLI 覆盖 ----
CLI_BASE_ASSETS: set[str] = set()

#创建全局变量 ADAPTERS ，只在main()执行一次
def build_adapters(app: AppConfig):
    ADAPTERS.append(KyberAdapter(app))
    ADAPTERS.append(OdosAdapter(app))


# 3. Helper functions
async def best_quote_leg(leg : LegQuote ):
    """从多个适配器中选择最佳报价"""
    
    # Create tasks for each adapter
    tasks = []
    for adapter in ADAPTERS :        
        if adapter.supports_chain(leg.chain.name):
            tasks.append((adapter.name, asyncio.create_task(adapter.quote(leg))))
        else:
            DBG.info(f"[DEBUG] Adapter {adapter.name} does not support chain {leg.chain.name}")
            DBG.info(f"[DEBUG] Adapter {adapter.name} settings: {adapter.app.settings.get('adapter_support', {})}")
    
    if not tasks:
        DBG.error(f"[DEBUG] No adapters support chain {leg.chain.name}")
        leg.status = LEG_NO_QUOTE
        leg.note = f"No adapters support chain {leg.chain.name}"
        return leg
    
    # Gather quotes with error handling
    quotes = []
    for adapter_name, task in tasks:
        try:
            DBG.info(f"[DEBUG] Awaiting quote from {adapter_name}")
            quote = await task
            DBG.info(f"[DEBUG] Raw quote from {adapter_name}: {vars(quote)}")
            quotes.append(quote)
        except Exception as e:
            DBG.error(f"[ERROR] Adapter {adapter_name} failed: {str(e)}")
            DBG.error(f"[ERROR] Stack trace: ", exc_info=True)

    DBG.info(f"[DEBUG] Received {len(quotes)} quotes")

    # Find best quote
    best = None
    for quote in quotes:
        DBG.info(f"[DEBUG] Processing quote - adapter: {quote.adapter}, status: {quote.status}, out_wei: {quote.out_wei}")
        if quote.status != LEG_OK:
            DBG.info(f"[DEBUG] Skipping quote - reason: {quote.note}")
            continue
            
        if not best or quote.out_b > best.out_b:
            best = quote
            DBG.info(f"[DEBUG] New best quote from {quote.adapter}: {quote.out_b} {leg.b.name}")
    
    if best:
        DBG.info(f"[DEBUG] Selected best quote from {best.adapter}: {best.out_b} {leg.b.name}")
        return best
    
    # No valid quotes found
    leg.status = LEG_NO_QUOTE
    leg.note = "no valid quotes"
    DBG.error(f"[DEBUG] No valid quotes received from any adapter")
    return leg

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

def _parse_base_assets_arg(s: str | None) -> set[str]:
    if not s:
        return set()
    return {x.strip().upper() for x in s.split(",") if x.strip()}

def _base_assets(app: AppConfig) -> set[str]:
    # CLI 优先；否则读 config.json settings.base_assets
    if CLI_BASE_ASSETS:
        return {x for x in CLI_BASE_ASSETS if x in app.assets}
    cfg = SETTINGS.get("base_assets", [])
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



async def _process_watchlist(pair: Dict) -> Optional[TwoLegResult]:
    """处理单个 watchlist 对"""
    try:
        # Get chains
        ci = CHAINS.get_chain(pair["A_chain"])
        cj = CHAINS.get_chain(pair["B_chain"])
        
        if not ci or not cj:
            DBG.info(f"[WATCH] Chain not found: {pair['A_chain']} or {pair['B_chain']}")
            return None
            
        DBG.info(f"[WATCH] Processing {ci.name}:{pair['A']} > {cj.name}:{pair['B']}")
        
        # Get assets
        a_asset = ASSETS.get_asset(pair["A"])
        b_asset = ASSETS.get_asset(pair["B"])
        
        if not a_asset or not b_asset:
            DBG.error(f"[WATCH] Asset not found: {pair['A']} or {pair['B']}")
            return None
            
        # Get tokens on respective chains
        tokenA_i = a_asset.get_token(ci.name)  # Token A on chain i
        tokenB_i = b_asset.get_token(ci.name)  # Token B on chain i (bridge target)
        tokenA_j = a_asset.get_token(cj.name)  # Token A on chain j
        tokenB_j = b_asset.get_token(cj.name)  # Token B on chain j
        
        if not (tokenA_i and tokenB_i and tokenA_j and tokenB_j):
            DBG.error(f"[WATCH] Missing token addresses on chains")
            return None
        
        # Create legs for USDT->USDC on ethereum, then USDC->USDT on arbitrum
        leg1 = LegQuote(
            chain=ci,
            a=tokenA_i,
            b=tokenB_i,
            base_in=pair["base"]
        )
        
        leg2 = LegQuote(
            chain=cj,
            a=tokenB_j,  # Starting with USDC on arbitrum
            b=tokenA_j,  # Converting back to USDT
            base_in=0    # Will be set after leg1 quote
        )
        
        twoleg = TwoLegResult(leg1, leg2)
        
        DBG.info(f"[WATCH] Created two-leg quote: {ci.name}:{tokenA_i.name}>{tokenB_i.name} -> {cj.name}:{tokenB_j.name}>{tokenA_j.name}")
        
        # Get quotes
        result = await quote_two_leg_once(twoleg)
        DBG.info(f"[WATCH] Quote result: status={result.status}, pnl={result.pnl}")
        return result
        
    except Exception as e:
        DBG.error(f"[WATCH] Error processing pair {pair}: {e}")
        return None
    

# 4. Core functions
def load_config():
    """加载配置文件"""
    global CHAINS, DBG,CONFIG_PATH,ASSETS,MAINCONFIG,SETTINGS

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        DBG.info("[CONFIG] Loading config file...")
        
        # 初始化全局 CHAINS
        CHAINS = load_chains_from_config(raw)
        DBG.info(f"[CONFIG] Loaded {len(CHAINS)} chains")
        
        # 加载资产
        ASSETS = load_assets_from_config(raw, CHAINS)
        DBG.info(f"[CONFIG] Loaded {len(ASSETS.assets)} assets")
        
        # 加载 settings
        SETTINGS = raw.get("settings", {})
        if not isinstance(SETTINGS, dict):
            SETTINGS = {}

        DBG.info(SETTINGS.get("run_mode","watchlist"))       
        DBG.info(f"[CONFIG] Loaded {len(SETTINGS)} settings")
        DBG.info("[CONFIG] Config loaded successfully")
        
        MAINCONFIG=AppConfig(
            api=raw.get("api", {}),
            chains=CHAINS.chains,
            assets=ASSETS,
            settings=SETTINGS
        )

        return MAINCONFIG
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

    # 修改这里：不要转换格式，保持原始格式
    norm_list = []
    for i, it in enumerate(data):
        if not isinstance(it, dict):
            DBG.warn(f"[watchlist] item {i} not a dict, skip")
            continue
            
        # 直接使用原始格式
        norm_list.append(it)

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

'''
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
    code2, text2, headers2 = await _fetch_http("GET", url2, params=qs, timeout=int(SETTINGS.get("timeout_sec", 12)))
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
        "slippageLimitPercent": float(SETTINGS.get("slippage_limit_percent", 0.5)),
        "compact": True
    }
    code, text, headers = await _fetch_http("POST", url, json_body=payload, timeout=int(SETTINGS.get("timeout_sec", 12)))
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
'''
async def quote_basic_leg(leg: LegQuote) -> "LegQuote":
    """Get basic quote from adapters for a token pair on a single chain."""
    try:
        DBG.info(f"[DEBUG] quote_basic_leg start: {leg.a.name}>{leg.b.name} on {leg.chain.name}")

        # Convert input amount to wei
        leg.in_wei = int(float(leg.base_in) * (10 ** leg.a.decimals))
        DBG.info(f"[DEBUG] Input conversion: {leg.base_in} -> {leg.in_wei} wei (decimals={leg.a.decimals})")

        # Get quote
        try:
            await best_quote_leg(leg)
            DBG.info(f"[DEBUG] Quote result: adapter={leg.adapter}, out_wei={leg.out_wei}, status={leg.status}")
        except Exception as quote_error:
            DBG.error(f"[DEBUG] Error in best_quote_leg: {str(quote_error)}")
            leg.status = LEG_ERROR
            leg.note = f"Quote error: {str(quote_error)}"
            return leg

        # Add output conversion debugging
        if leg.out_wei:
            leg.out_b = leg.out_wei / (10 ** leg.b.decimals)
            DBG.info(f"[DEBUG] Output conversion: {leg.out_wei} wei -> {leg.out_b} (decimals={leg.b.decimals})")
            DBG.info(f"[DEBUG] Price impact: in={leg.base_in} {leg.a.name}, out={leg.out_b} {leg.b.name}")
        else:
            DBG.error(f"[DEBUG] No output amount received")
            leg.status = LEG_NO_QUOTE
            leg.note = "No output amount"

        return leg

    except Exception as e:
        DBG.error(f"[DEBUG] Error in quote_basic_leg: {str(e)}")
        leg.status = LEG_ERROR
        leg.note = f"General error: {str(e)}"
        return leg

async def _quote_two_leg(twoleg: TwoLegResult) -> TwoLegResult:
    """获取两条腿的报价"""
    try:
        # 第一条腿: ci 上 A->B
        await quote_basic_leg(twoleg.leg1)
        if twoleg.leg1.status != LEG_OK:
            twoleg.leg1.note = f"Leg1 failed: {twoleg.leg1.note}"
            return twoleg
        
        twoleg.leg2.base_in = twoleg.leg1.out_b

        # 第二条腿: cj 上 B->A
        await quote_basic_leg(twoleg.leg2)
        if twoleg.leg2.status != LEG_OK:
            twoleg.leg2.note = f"Leg2 failed: {twoleg.leg2.note}"
            return twoleg
            

        # 计算收益
        twoleg.final_a = twoleg.leg2.out_b
        twoleg.pnl = twoleg.final_a - twoleg.leg1.base_in
        twoleg.pnl_pct = (twoleg.pnl / twoleg.leg1.base_in) * 100 if twoleg.leg1.base_in else None
        twoleg.status = LEG_OK
        twoleg.note = "Success"
        return twoleg

    except Exception as e:
        twoleg.status = LEG_ERROR
        twoleg.note = f"Quote error: {str(e)}"
        return twoleg
        

async def quote_two_leg_once(twoleg: TwoLegResult) -> TwoLegResult:
    """Quote two legs with concurrent size testing"""
    try:
        global QUOTE_COUNTER, QUOTE_START_TIME
        QUOTE_START_TIME = time.time()
        
        # Get initial quote
        result = await _quote_two_leg(twoleg)
        DBG.info(f"[OPTIMIZE] Initial quote result: status={result.status}, pnl={result.pnl}")
        QUOTE_COUNTER += 1
        
        if result.status == LEG_OK and result.pnl is not None and result.pnl > 0:
            DBG.info(f"[OPTIMIZE] Found profitable trade, testing different sizes. Initial pnl: {result.pnl}")
            
            timeout = SETTINGS.get("optimize_timeout_sec", 20)

            # Create test legs with different sizes
            multipliers = [0.5] + [i for i in range(2, 11)]
            tasks = []
            
            # Create all tasks at once
            for mult in multipliers:
                new_leg1 = LegQuote(
                    chain=twoleg.leg1.chain,
                    a=twoleg.leg1.a,
                    b=twoleg.leg1.b,
                    base_in=twoleg.leg1.base_in * mult
                )
                new_leg2 = LegQuote(
                    chain=twoleg.leg2.chain,
                    a=twoleg.leg2.a,
                    b=twoleg.leg2.b,
                    base_in=0  # Will be set after leg1 quote
                )
                test_twoleg = TwoLegResult(new_leg1, new_leg2)
                task = asyncio.create_task(_quote_two_leg(test_twoleg))
                tasks.append((new_leg1.base_in, task))

            try:
                # Wait for all tasks with timeout
                done, pending = await asyncio.wait(
                    [t[1] for t in tasks],
                    timeout=timeout
                )
                
                # Process completed tasks
                for size, task in tasks:
                    if task in done and not task.exception():
                        try:
                            test_result = task.result()
                            QUOTE_COUNTER += 1
                            DBG.info(f"[COUNT] Size test: size={size:.2f}, pnl={test_result.pnl or 0:.2f}")
                            
                            if (test_result.status == LEG_OK and 
                                test_result.pnl is not None and 
                                test_result.pnl > result.pnl):
                                result = test_result
                                DBG.info(f"[OPTIMIZE] New best result: size={size:.2f}, pnl={test_result.pnl:.2f}")
                        except Exception as e:
                            DBG.error(f"Error processing size {size}: {e}")
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                    
            except asyncio.TimeoutError:
                DBG.warn(f"Optimization timeout after {timeout}s")
                for _, task in tasks:
                    if not task.done():
                        task.cancel()
            
            elapsed = time.time() - QUOTE_START_TIME
            result.note = f"{result.note} (Quotes: {QUOTE_COUNTER}, Time: {elapsed:.2f}s)"
        else:
            elapsed = time.time() - QUOTE_START_TIME
            result.note = f"{result.note} (Quote time: {elapsed:.2f}s)"
            
        return result
        
    except Exception as e:
        elapsed = time.time() - QUOTE_START_TIME
        DBG.error(f"Error in quote_two_leg_once: {e}")
        result.status = LEG_ERROR
        result.note = f"Quote error: {str(e)} (Time: {elapsed:.2f}s)"
        return result

# 5. Mode functions
async def run_one_leg_mode(pair_spec: str,  refresh: int, verbose: bool, once: bool, pair_any: str):
    """运行单腿报价模式"""
    _ensure_dir(os.path.dirname(OUT_HTML))
    _write_placeholder_html(OUT_HTML, "Single Leg Mode", refresh)

    global DBG
    DBG = init_debug_logger("single-leg", OUT_HTML, verbose)

    while True:
        try:
            # Handle pair-any mode
            if pair_any:
                if ">" not in pair_any:
                    raise ValueError(f"Invalid pair-any format: {pair_any}")
                    
                a_sym, b_sym = pair_any.split(">")
                DBG.info(f"[DEBUG] Parsed pair-any tokens: {a_sym}>{b_sym}")

                assetA = ASSETS.get_asset(a_sym)    
                assetB = ASSETS.get_asset(b_sym)

                if not assetA or not assetB:
                    raise ValueError(f"Asset not found: {a_sym} or {b_sym}")    
                

                # Get quotes for all chains
                legs = []
                for chain in MAINCONFIG.chains:
                    tokenA = assetA.get_token(chain.name) if assetA else None   
                    tokenB = assetB.get_token(chain.name) if assetB else None
                    if not tokenA or not tokenB:
                        DBG.info(f"[DEBUG] Token not found on chain {chain.name}: {a_sym} or {b_sym}, skip")
                        continue    
                    leg = LegQuote(chain, tokenA, tokenB, 10000)
                    result = await quote_basic_leg(leg)
                    legs.append(result)
                
                # Render all results
                html = render_legs_page(
                    title=f"Multi-Chain Quote: {pair_any}",
                    refresh=refresh,
                    legs=legs
                )
                _write_html(OUT_HTML, html)
                
            # Handle single chain mode
            else:
                if not pair_spec or ":" not in pair_spec or ">" not in pair_spec:
                    raise ValueError(f"Invalid pair spec format: {pair_spec}")
                    
                chain_name, tokens = pair_spec.split(":")
                a_sym, b_sym = tokens.split(">")
                
                # Add debug logs
                DBG.info(f"[DEBUG] Processing pair: {chain_name}:{a_sym}>{b_sym}")
                
                # Get chain
                chain = CHAINS.get_chain(chain_name)
                if not chain:
                    raise ValueError(f"Chain not found: {chain_name}")
                DBG.info(f"[DEBUG] Found chain: {chain.name}")
                
                assetA=ASSETS.get_asset(a_sym)
                assetB=ASSETS.get_asset(b_sym)
                if not assetA or not assetB:
                    raise ValueError(f"Asset not found: {a_sym} or {b_sym}")    

                tokenA=assetA.get_token(chain.name) if assetA else None
                tokenB=assetB.get_token(chain.name) if assetB else None

                if not tokenA or not tokenB:
                    raise ValueError(f"Token not found on chain {chain.name}: {a_sym} or {b_sym}")      

                leg=LegQuote(chain,tokenA,tokenB,10000)
                # Get quote with debug info
                DBG.info(f"[DEBUG] Calling quote_basic_leg with base_in=10000")
                rleg = await quote_basic_leg(leg)
                DBG.info(f"[DEBUG] Got leg quote: status={rleg.status}, adapter={rleg.adapter}")
                DBG.info(f"[DEBUG] Quote details: in={rleg.base_in}, out={rleg.out_b}, out_wei={rleg.out_wei}")
                
                # Render result
                html = render_legs_page(
                    title=f"Single Leg Quote: {pair_spec}",
                    refresh=refresh,
                    legs=[rleg]
                )
                _write_html(OUT_HTML, html)
            
            if once:
                break
                
            await asyncio.sleep(refresh)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            DBG.error(f"Error in one-   leg mode: {e}")
            traceback.print_exc()
            if once:
                break
            await asyncio.sleep(refresh)

async def run_two_leg_mode(pair_spec: str,  refresh: int, verbose: bool, once: bool):
    """运行两腿套利模式，分析跨链套利机会"""
    _ensure_dir(os.path.dirname(OUT_HTML))
    _write_placeholder_html(OUT_HTML, "Two-Leg Mode", refresh)

    global DBG
    DBG = init_debug_logger("two-leg", OUT_HTML, verbose)
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
            ci = CHAINS.get_chain(ci_name)
            cj = CHAINS.get_chain(cj_name)
            if not ci or not cj:
                raise ValueError(f"Chain not found: {ci_name} or {cj_name}")

            # Get assets and tokens
            assetA=ASSETS.get_asset(a_sym)
            assetB=ASSETS.get_asset(b_sym)
            if not assetA or not assetB:
                raise ValueError(f"Asset not found: {a_sym} or {b_sym}")
            
            tokenA_i=assetA.get_token(ci.name) if assetA else None  
            tokenB_i=assetB.get_token(ci.name) if assetB else None  
            tokenA_j=assetA.get_token(cj.name) if assetA else None  
            tokenB_j=assetB.get_token(cj.name) if assetB else None  
            if not (tokenA_i and tokenB_i and tokenA_j and tokenB_j):       
                raise ValueError(f"Token not found on chains: {ci.name} or {cj.name}")      

            # 使用初始数量查询报价（默认10000）
            base_in = 100000.0


            DBG.info(f"[CHECK] Quoting {ci_name}:{a_sym}>{b_sym} -> {cj_name}:{b_sym}>{a_sym} with base {base_in}")
            

            leg1= LegQuote(
                chain=ci,
                a=tokenA_i,
                b=tokenB_i,
                base_in=base_in
            )
            leg2= LegQuote(
                chain=cj,
                a=tokenB_j,  # 第二腿从 B 开始
                b=tokenA_j,
                base_in=0  # 第二腿的输入数量由第一腿的输出决定
            )   

            twoleg= TwoLegResult(leg1,leg2)
            DBG.info(f"[CHECK] Created two-leg quote object")   
            # 获取报价并优化数量
            result = await quote_two_leg_once(twoleg)  
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
                _write_html(OUT_HTML, html)
                
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

async def run_watchlist_mode(refresh: int, once: bool, verbose: bool = True) -> None:
    """运行监控模式"""
    _ensure_dir(os.path.dirname(OUT_HTML))
    _write_placeholder_html(OUT_HTML, "Watchlist Mode", refresh)
    
    global DBG
    DBG = init_debug_logger("watchlist", OUT_HTML, verbose)
    
    while True:
        try:
            # 读取 watchlist
            pairs = _load_watchlist_json(WATCHLIST_PATH)
            DBG.info(f"[CHECK] Loaded pairs from watchlist: {len(pairs)}")
            
            results = []
            timeout_count = 0
            error_count = 0

            # 创建所有任务
            tasks = []
            for pair in pairs:
                task = asyncio.create_task(_process_watchlist(pair))
                tasks.append((pair, task))

            # 等待所有任务完成或超时
            timeout = SETTINGS.get("timeout_sec", 60)
            DBG.info(f"[CHECK] Processing {len(tasks)} pairs with timeout={timeout}s")
            
            try:
                done, pending = await asyncio.wait(
                    [t[1] for t in tasks],
                    timeout=timeout
                )
                
                # 处理完成的任务
                for pair, task in tasks:
                    if task in done:
                        try:
                            result = task.result()
                            if result:
                                results.append(result)
                                DBG.info(f"[CHECK] Got result for {pair['A_chain']}:{pair['A']}>{pair['B_chain']}:{pair['B']}")
                        except Exception as e:
                            error_count += 1
                            DBG.error(f"Task error for pair {pair}: {e}")
                            # Create a blank TwoLegResult for error case
                            results.append(TwoLegResult(
                                LegQuote(
                                    chain=CHAINS.get_chain(pair["A_chain"]),
                                    a=ASSETS.get_asset(pair["A"]).get_token(pair["A_chain"]),
                                    b=ASSETS.get_asset(pair["B"]).get_token(pair["A_chain"]),
                                    base_in=pair["base"]
                                ),
                                LegQuote(
                                    chain=CHAINS.get_chain(pair["B_chain"]),
                                    a=ASSETS.get_asset(pair["B"]).get_token(pair["B_chain"]),
                                    b=ASSETS.get_asset(pair["A"]).get_token(pair["B_chain"]),
                                    base_in=0
                                ),
                                status="ERROR",
                                note=f"Error: {str(e)}"
                            ))
                    else:
                        timeout_count += 1
                        DBG.warn(f"Task timeout for pair {pair}")
                        task.cancel()
                        # Create a blank TwoLegResult for timeout case
                        results.append(TwoLegResult(
                            LegQuote(
                                chain=CHAINS.get_chain(pair["A_chain"]),
                                a=ASSETS.get_asset(pair["A"]).get_token(pair["A_chain"]),
                                b=ASSETS.get_asset(pair["B"]).get_token(pair["A_chain"]),
                                base_in=pair["base"]
                            ),
                            LegQuote(
                                chain=CHAINS.get_chain(pair["B_chain"]),
                                a=ASSETS.get_asset(pair["B"]).get_token(pair["B_chain"]),
                                b=ASSETS.get_asset(pair["A"]).get_token(pair["B_chain"]),
                                base_in=0
                            ),
                            status="TIMEOUT",
                            note=f"Timeout after {timeout}s"
                        ))

                # Cancel any remaining pending tasks
                for task in pending:
                    task.cancel()
                    
            except asyncio.TimeoutError:
                timeout_count += len(tasks)
                DBG.warn(f"Global timeout after {timeout}s")
                for _, task in tasks:
                    if not task.done():
                        task.cancel()

            # 渲染结果页面
            html = render_watchlist_page(
                f"Watchlist Mode (Timeouts: {timeout_count}, Errors: {error_count})",
                results,
                refresh,
                len(pairs),
                len(results)
            )
            _write_html(OUT_HTML, html)
            
            DBG.info(f"[CHECK] Processed {len(results)}/{len(pairs)} pairs")
            DBG.info(f"[CHECK] Results: {len(results)} (Timeouts: {timeout_count}, Errors: {error_count})")

            if once:
                break

            await asyncio.sleep(refresh)
            
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
    .muted{color:#64748b;font-size:12px}
    """
    
    rows = []
    total_quotes = 0
    for r in results:
        pnl_class = "profit" if r.pnl and r.pnl > 0 else "loss"
        
        # 从 note 中提取 Quotes 数量
        quote_count = 0
        if "Quotes:" in r.note:
            try:
                quote_str = r.note.split("Quotes:")[1].split(",")[0].strip()
                quote_count = int(quote_str)
                total_quotes += quote_count
            except:
                pass
                
        # 添加计数显示
        count_display = f"<span class='muted'>({quote_count} quotes)</span>" if quote_count > 0 else ""
        
        rows.append(f"""
            <tr>
                <td>{r.leg1.chain.name}→{r.leg2.chain.name}</td>
                <td>{r.leg1.a.name}→{r.leg1.b.name}→{r.leg1.a.name} {count_display}</td>
                <td>{_fmt(r.leg1.base_in)}</td>
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

    # 添加总请求数显示
    total_stats = f"Total quotes: {total_quotes}"

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
            {progress} · {total_stats} · Auto refresh {refresh}s · {_now_str()}
        </div>
        {table}
    </body>
    </html>
    """

# 6. Main function
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="开启调试面板并写日志（需要 logger_utils.py）")
    ap.add_argument("--log-file", required=False, help="日志文件路径；缺省为与HTML同名的 .log")
    #ap.add_argument("--config", required=True)
    
    #watchList Mode
    #ap.add_argument("--watchlist",type=str,help="Path to a watchlist.json (two-leg monitoring list). If set, run watchlist mode.")    # 单腿“合并输出”模式
    ap.add_argument("--pair-spec", required=False,help='留空=自动遍历(USDT/USDC + USDT/USDE + 与 FRXUSD 双向); 例: "ethereum:USDT>USDE,plasma:USDT>USDE"')
    # 双腿遍历（单对）
    ap.add_argument("--two-leg",action="store_true",help="Enable two-leg mode (requires --pair-spec with two legs, e.g. ethereum:A>B,plasma:B>A)")
    #ap.add_argument("--pair-html", required=True)

    ap.add_argument("--refresh", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--once", action="store_true")
    # ★ 新增：基础资产 CLI 覆盖
    ap.add_argument("--base-assets", required=False, help='用逗号分隔，例如 "USDT,USDC"；若提供则覆盖 config.json 的 settings.base_assets')
    ap.add_argument("--pair-any", required=False,help='只写资产对，自动遍历所有共同链，形如 "USDE>USDC"（大小写不敏感）')
    args = ap.parse_args()

    # Load config once at the start
    load_config()
    DBG.info("[CHECK] Configuration loaded")
    DBG.info(f"[CHECK] Settings: {SETTINGS['run_mode']}")

    # 创建全局变量 ADAPTERS ，只在main()执行一次
    build_adapters(MAINCONFIG)

    # ★ 保存 CLI 覆盖集合
    global CLI_BASE_ASSETS
    CLI_BASE_ASSETS = _parse_base_assets_arg(args.base_assets)

    try:
        if SETTINGS.get("run_mode") == "watchlist" :
            asyncio.run(
                run_watchlist_mode( 
                    args.refresh,
                    args.once,
                    args.verbose
                )
            )
        elif SETTINGS.get("run_mode") =="two-leg":
            # 两腿模式
            asyncio.run(run_two_leg_mode(  
                args.pair_spec,
                args.refresh,
                args.verbose,
                args.once
            ))
        elif SETTINGS.get("run_mode") =="one-leg":
            # 单腿模式
            asyncio.run(run_one_leg_mode(
                args.pair_spec,
                args.refresh,
                args.verbose,
                args.once,
                args.pair_any
            ))
        else:
            ap.error("请选择 --two-leg 或 --pair-spec/--pair-any 之一")
    except ImportError:
            print("logger_utils.py not found, debug mode is disabled.")
            args.debug = False
    
    

# 7. Entry point
if __name__ == "__main__":
    main()





