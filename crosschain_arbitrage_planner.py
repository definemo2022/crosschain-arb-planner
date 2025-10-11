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
from datetime import datetime

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
FITTING_ADAPTERS:List[BaseAdapter] = []

#创建全局变量 ADAPTERS ，只在main()执行一次
def build_adapters():
    global FITTING_ADAPTERS
    ADAPTERS.append(KyberAdapter(MAINCONFIG, DBG))
    ADAPTERS.append(OdosAdapter(MAINCONFIG, DBG))
    FITTING_ADAPTERS = ADAPTERS.copy() #one-leg 时使用全部适配器

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

def parse_pair_spec(pair_spec: str):
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

def get_fitting_adapters(ci: Chain, cj: Chain) -> List[BaseAdapter]:
    """获取同时支持 ci 和 cj 的适配器列表"""
    fitting = []
    for adapter in ADAPTERS:
        if adapter.supports_chain(ci.name) and adapter.supports_chain(cj.name):
            fitting.append(adapter)
    return fitting

# 4. Core functions
async def process_watchlist(pair: Dict) -> Optional[TwoLegResult]:
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
        result = await quote_optimize_two_leg(twoleg)
        DBG.info(f"[WATCH] Quote result: status={result.status}, pnl={result.pnl}")
        return result
        
    except Exception as e:
        DBG.error(f"[WATCH] Error processing pair {pair}: {e}")
        return None
    
def load_config(config_path: str) -> AppConfig  :
    global CHAINS,ASSETS,MAINCONFIG,SETTINGS

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        # 初始化全局 CHAINS
        CHAINS = load_chains_from_config(raw)
     
        
        # 加载资产
        ASSETS = load_assets_from_config(raw, CHAINS)
    
        # 加载 settings
        SETTINGS = raw.get("settings", {})
        if not isinstance(SETTINGS, dict):
            SETTINGS = {}

        MAINCONFIG=AppConfig(
            api=raw.get("api", {}),
            chains=CHAINS.chains,
            assets=ASSETS,
            settings=SETTINGS
        )

        return MAINCONFIG
    except Exception as e:
        raise

def load_watchlist_json(path: str) -> list[dict]:
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

async def quote_leg_from_adapters(leg : LegQuote ):
    """从多个适配器中选择最佳报价"""
    
    # Create tasks for each adapter
    tasks = []
    for adapter in FITTING_ADAPTERS :        
        tasks.append((adapter.name, asyncio.create_task(adapter.quote(leg))))

    if not tasks:
        DBG.error(f"[DEBUG] No adapters support chain {leg.chain.name}")
        leg.status = LEG_NO_QUOTE
        leg.note = f"No adapters support chain {leg.chain.name}"
        return leg
    
    # Gather quotes with error handling
    quotes = []
    for adapter_name, task in tasks:
        try:
            quote = await task
            DBG.info(f"[DEBUG] Raw quote from {adapter_name}: {vars(quote)}")
            quotes.append(quote)
        except Exception as e:
            DBG.error(f"[ERROR] Adapter {adapter_name} failed: {str(e)}")
            DBG.error(f"[ERROR] Stack trace: ", exc_info=True)

    DBG.info(f"[DEBUG] Received {len(quotes)} quotes")

    # Find best quote
    best = compare_best_leg(quotes)
    if best:
        DBG.info(f"[DEBUG] Selected best quote from {best.adapter}: {best.out_b} {leg.b.name}")
        return best
    
    # No valid quotes found
    leg.status = LEG_NO_QUOTE
    leg.note = "no valid quotes"
    DBG.error(f"[DEBUG] No valid quotes received from any adapter")
    return leg

def compare_best_leg(legs: List[LegQuote]) -> Optional[LegQuote]:
    """比较多个 LegQuote，返回 out_b 最大且状态为 LEG_OK 的 LegQuote"""
    best = None
    for leg in legs:
        if leg.status != LEG_OK:
            continue
        if not best or leg.out_b > best.out_b:
            best = leg
    return best

async def quote_one_leg(leg: LegQuote) -> "LegQuote":
    """Get basic quote from adapters for a token pair on a single chain."""
    try:
        DBG.info(f"[DEBUG] quote_basic_leg start: {leg.a.name}>{leg.b.name} on {leg.chain.name}")

        # Convert input amount to wei
        leg.in_wei = int(float(leg.base_in) * (10 ** leg.a.decimals))
        DBG.info(f"[DEBUG] Input conversion: {leg.base_in} -> {leg.in_wei} wei (decimals={leg.a.decimals})")

        # Get quote
        try:
            await quote_leg_from_adapters(leg)
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

async def quote_two_leg_once(twoleg: TwoLegResult) -> TwoLegResult:
    """获取两条腿的报价"""
    try:
        global FITTING_ADAPTERS
        FITTING_ADAPTERS = get_fitting_adapters(twoleg.leg1.chain, twoleg.leg2.chain)   

        # 第一条腿: ci 上 A->B
        await quote_one_leg(twoleg.leg1)
        if twoleg.leg1.status != LEG_OK:
            twoleg.leg1.note = f"Leg1 failed: {twoleg.leg1.note}"
            return twoleg
        
        twoleg.leg2.base_in = twoleg.leg1.out_b

        # 第二条腿: cj 上 B->A
        await quote_one_leg(twoleg.leg2)
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
        
async def quote_optimize_two_leg(twoleg: TwoLegResult) -> TwoLegResult:
    """Quote two legs with concurrent size testing"""
    try:
        global QUOTE_COUNTER, QUOTE_START_TIME
        QUOTE_START_TIME = time.time()
        
        # Get initial quote
        result = await quote_two_leg_once(twoleg)
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
                task = asyncio.create_task(quote_two_leg_once(test_twoleg))
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
def make_one_legs(pair_spec: str) -> List[LegQuote]:

    #可以不指定链，必须指定TokenA和TokenB
    if not pair_spec or ">" not in pair_spec:
        raise ValueError(f"Invalid pair spec format: {pair_spec}")
    
    chains: Chains
    if ":" not in pair_spec:
        chains=CHAINS
        tokens=pair_spec
    else:
        chain_name, tokens = pair_spec.split(":")
        chains=Chains([CHAINS.get_chain(chain_name)]) if CHAINS.get_chain(chain_name) else Chains([])
    DBG.info(f"[DEBUG] make chains: {len(chains)} from {pair_spec}")


    a_sym, b_sym = tokens.split(">")

    DBG.info(f"[DEBUG] Parsed pair-any tokens: {a_sym}>{b_sym}")

    assetA = ASSETS.get_asset(a_sym)    
    assetB = ASSETS.get_asset(b_sym)

    base_in=SETTINGS.get("base_in")

    if not assetA or not assetB:
        raise ValueError(f"Asset not found: {a_sym} or {b_sym}") 


    legs = []
    for chain in chains:
        tokenA = assetA.get_token(chain.name) if assetA else None
        tokenB = assetB.get_token(chain.name) if assetB else None
        if not tokenA or not tokenB:
            DBG.info(f"[DEBUG] Token not found on chain {chain.name}: {a_sym} or {b_sym}, skip")
            continue
        leg = LegQuote(chain, tokenA, tokenB, base_in)
        legs.append(leg)  
    return legs

async def run_one_leg_mode(legs:List[LegQuote]):
    
    refresh=SETTINGS.get("refresh") 
    verbose=SETTINGS.get("verbose")
    once=SETTINGS.get("once") 
    
    """运行单腿报价模式"""
    _ensure_dir(os.path.dirname(OUT_HTML))
    _write_placeholder_html(OUT_HTML, "One Leg Mode", refresh)

    global DBG,FITTING_ADAPTERS
    

    FITTING_ADAPTERS = get_fitting_adapters(legs[0].chain, legs[0].chain)

    while True:
        try:
            tasks = []
            for leg in legs:
                DBG.info(f"[DEBUG] Processing leg: {leg.a.name}>{leg.b.name} on {leg.chain.name}")
                task = asyncio.create_task(quote_one_leg(leg))
                tasks.append((leg, task))

            # 等待所有任务完成或超时
            timeout = SETTINGS.get("timeout_sec", 60)
            DBG.info(f"[CHECK] Processing {len(tasks)} pairs with timeout={timeout}s")
            
            try:
                done, pending = await asyncio.wait(
                    [t[1] for t in tasks],
                    timeout=timeout
                )
                results = []
                # 处理完成的任务
                for leg, task in tasks:
                    if task in done:
                        try:
                            result = task.result()
                            if result:
                                results.append(result)
                                DBG.info(f"[CHECK] Got result for {leg.chain.name}:{leg.a.name}>{leg.b.name}")
                        except Exception as e:
                            error_count += 1
                            DBG.error(f"Task error for {leg.chain.name}:{leg.a.name}>{leg.b.name}: {e}")
                    else:
                        timeout_count += 1
                        DBG.warn(f"Task timeout for {leg.chain.name}:{leg.a.name}>{leg.b.name}")
                        task.cancel()
                        leg.status = "TIMEOUT"
                        leg.note = f"Timeout after {timeout}s"  
                        # Create a blank TwoLegResult for timeout case
                        results.append(leg)

                # Cancel any remaining pending tasks
                for task in pending:
                    task.cancel()
                    
            except asyncio.TimeoutError:
                timeout_count += len(tasks)
                DBG.warn(f"Global timeout after {timeout}s")
                for _, task in tasks:
                    if not task.done():
                        task.cancel()

            # Render result
            DBG.info(f"[CHECK] Rendering page ")   
            html = render_legs_page(
                title=f"One Leg Quote: {leg.chain.name}:{leg.a.name}>{leg.b.name}",
                refresh=refresh,
                legs=results
            )
            _write_html(OUT_HTML, html)
            DBG.info("[CHECK] Page rendered and written to file")
            if once:
                break

            await asyncio.sleep(refresh)
        except KeyboardInterrupt:
            break
        except Exception as e:
            DBG.error(f"Error in one-leg mode: {e}")
            traceback.print_exc()
            if once:
                break
            await asyncio.sleep(refresh)

def make_two_leg(pair_spec: str) -> TwoLegResult:
    """根据 pair_spec 创建 TwoLegResult 对象"""
    ci_name, a_sym, b_sym, cj_name = parse_pair_spec(pair_spec)
    
    ci = CHAINS.get_chain(ci_name)
    cj = CHAINS.get_chain(cj_name)
    if not ci or not cj:
        raise ValueError(f"Chain not found: {ci_name} or {cj_name}")

    assetA=ASSETS.get_asset(a_sym)
    assetB=ASSETS.get_asset(b_sym)
    if not assetA or not assetB:
        raise ValueError(f"Asset not found: {a_sym} or {b_sym}")
    
    tokenA_i=assetA.get_token(ci.name) if assetA else None  
    tokenB_i=assetB.get_token(ci.name) if assetB else None  
    tokenA_j=assetA.get_token(cj.name) if assetA else None  
    tokenB_j=assetB.get_token(cj.name) if assetB else None  


     # 使用初始数量查询报价
    base_in = SETTINGS.get("base_in")
    
    if not (tokenA_i and tokenB_i and tokenA_j and tokenB_j):       
        raise ValueError(f"Token not found on chains: {ci.name} or {cj.name}")      

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
    return twoleg

async def run_two_leg_mode(twoleg: TwoLegResult) -> None:

    
    refresh=SETTINGS.get("refresh")
    verbose=SETTINGS.get("verbose")
    once=SETTINGS.get("once")   

    """运行两腿套利模式，分析跨链套利机会"""
    _ensure_dir(os.path.dirname(OUT_HTML))
    _write_placeholder_html(OUT_HTML, "Two-Leg Mode", refresh)
    
    ci_name=twoleg.leg1.chain.name
    cj_name=twoleg.leg2.chain.name  
    a_sym=twoleg.leg1.a.name
    b_sym=twoleg.leg1.b.name

    global DBG
    
    DBG.info("[CHECK] Starting two-leg mode")
    
    while True:
        try:
            # 获取报价并优化数量
            result = await quote_optimize_two_leg(twoleg)  
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

async def run_watchlist_mode() -> None:
    refresh=SETTINGS["refresh"]
    once=SETTINGS["once"]
    verbose=SETTINGS["verbose"]

    _ensure_dir(os.path.dirname(OUT_HTML))
    _write_placeholder_html(OUT_HTML, "Watchlist Mode", refresh)
    
    global DBG
    
    while True:
        try:
            # 读取 watchlist
            pairs = load_watchlist_json(WATCHLIST_PATH)
            DBG.info(f"[CHECK] Loaded pairs from watchlist: {len(pairs)}")
            
            results = []
            timeout_count = 0
            error_count = 0

            # 创建所有任务
            tasks = []
            for pair in pairs:
                task = asyncio.create_task(process_watchlist(pair))
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

def _get_row_class(result: TwoLegResult) -> str:
    """Get CSS class for result row"""
    if result.status == LEG_ERROR or result.status == "ERROR":
        return "error"
    elif result.status == "TIMEOUT":
        return "timeout"
    elif result.pnl and result.pnl > 0:
        return "profit"
    else:
        return "normal"

def _format_time() -> str:
    """Format current time for display"""
    return datetime.now().strftime("%H:%M:%S")

def _format_pair(result: TwoLegResult) -> str:
    """Format trading pair for display"""
    if not (result.leg1 and result.leg2):
        return "-"
    return f"{result.leg1.chain.name}:{result.leg1.a.name}>{result.leg1.b.name},{result.leg2.chain.name}:{result.leg2.a.name}>{result.leg2.b.name}"

def _format_pnl(result: TwoLegResult) -> str:
    """Format PnL for display"""
    if result.pnl is None:
        return "-"
    return f"{result.pnl:.6f}"

def render_watchlist_page(title: str, results: List[TwoLegResult], refresh: int, total: int, done: int) -> str:
    """渲染 watchlist 模式的结果页面"""
    css = """
    <style>
        body { font-family: monospace; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        tr.profit { background-color: #90EE90; }
        tr.error { background-color: #FFB6C6; }
        tr.timeout { background-color: #FFE4B5; }
        tr.normal { background-color: white; }
    </style>
    """
    
    meta = f"""
    <meta http-equiv="refresh" content="{refresh}">
    <title>{title}</title>
    """
    
    table_header = """
    <tr>
        <th>Time</th>
        <th>Pair</th>
        <th>Base</th>
        <th>Final A</th>
        <th>PnL</th>
        <th>Status</th>
        <th>Note</th>
    </tr>
    """
    
    rows = []
    for r in results:
        final_a = r.leg2.out_b if r.leg2 and r.leg2.out_b else None
        final_a_str = f"{final_a:.6f}" if final_a is not None else "-"
        
        row = f"""
        <tr class="{_get_row_class(r)}">
            <td>{_format_time()}</td>
            <td>{_format_pair(r)}</td>
            <td>{r.leg1.base_in if r.leg1 else '-'}</td>
            <td>{final_a_str}</td>
            <td>{_format_pnl(r)}</td>
            <td>{r.status or '-'}</td>
            <td>{r.note or '-'}</td>
        </tr>
        """
        rows.append(row)
    
    html = f"""
    <html>
    <head>
        {meta}
        {css}
    </head>
    <body>
        <h2>{title}</h2>
        <p>Processed: {done}/{total}</p>
        <table>
            {table_header}
            {''.join(rows)}
        </table>
    </body>
    </html>
    """
    return html

# 6. Main function
def main():
    global SETTINGS,DBG

    #先读取命令行参数
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="开启调试面板并写日志（需要 logger_utils.py）")
    ap.add_argument("--log-file", required=False, help="日志文件路径；缺省为与HTML同名的 .log")
    ap.add_argument("--config", default=CONFIG_PATH, help="配置文件路径，默认为 ./config.json")
    ap.add_argument("--pair-spec", required=False,help='留空=自动遍历(USDT/USDC + USDT/USDE + 与 FRXUSD 双向); 例: "ethereum:USDT>USDE,plasma:USDT>USDE"')
    ap.add_argument("--two-leg",action="store_true",help="Enable two-leg mode (requires --pair-spec with two legs, e.g. ethereum:A>B,plasma:B>A)")
    ap.add_argument("--refresh", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--base-assets", required=False, help='用逗号分隔，例如 "USDT,USDC"；若提供则覆盖 config.json 的 settings.base_assets')
    ap.add_argument("--pair-any", required=False,help='只写资产对，自动遍历所有共同链，形如 "USDE>USDC"（大小写不敏感）')
    ap.add_argument("--base-in", required=False, type=float, default=10000.0, help="基础交易数量，默认为 10000.0")
    ap.add_argument("--run-mode", required=False, default="watchlist", help="运行模式，默认为 watchlist,还可以接受one-leg或two-leg")

    # 解析命令行参数    
    args = ap.parse_args()

    # Validate args
    if bool(args.run_mode) and args.run_mode not in ["watchlist", "one-leg", "two-leg"]:
        ap.error("无效的运行模式")
    elif args.run_mode == "two-leg" and not args.pair_spec:
        ap.error("--two-leg 模式需要 --pair-spec 参数")
    elif args.run_mode == "one-leg" and not args.pair_spec:
        ap.error("--one-leg 模式需要 --pair-spec 参数")


    #only Load config once
    load_config(args.config)

    #用CLI参数覆盖配置文件
    if args.run_mode:
        SETTINGS["run_mode"] = args.run_mode
    if args.base_in:
        SETTINGS["base_in"] = args.base_in
    if args.refresh:
        SETTINGS["refresh"] = args.refresh
    if args.verbose:    
        SETTINGS["verbose"] = args.verbose  
    if args.once:  
        SETTINGS["once"] = args.once
    if args.debug:
        SETTINGS["debug"] = args.debug    



    #初始化DBG
    DBG=init_debug_logger(OUT_HTML,SETTINGS["verbose"],SETTINGS["debug"])
    DBG.info("[CHECK] Debug logger initialized")    
    
    # 创建全局变量 ADAPTERS ，只在main()执行一次
    build_adapters()

    # 运行不同的模式
    try:
        if SETTINGS.get("run_mode") == "watchlist" :
            asyncio.run(run_watchlist_mode())
        elif SETTINGS.get("run_mode") =="two-leg":
            twoleg=make_two_leg(args.pair_spec)  
            DBG.info("[CHECK] pair_spec validated for two-leg mode")
            # 两腿模式
            asyncio.run(run_two_leg_mode(twoleg))
        elif SETTINGS.get("run_mode") =="one-leg":
            legs=make_one_legs(args.pair_spec) 
            DBG.info(f"[CHECK] Created {len(legs)} legs for one-leg mode from pair_spec")
            # 单腿模式
            asyncio.run(run_one_leg_mode(legs))
        else:
            ap.error("请选择 --two-leg 或 --pair-spec/--pair-any 之一")
    except ImportError:
            print("logger_utils.py not found, debug mode is disabled.")
            args.debug = False

# 7. Entry point
if __name__ == "__main__":
    main()





