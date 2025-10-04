from __future__ import annotations
import os, json, time, argparse, asyncio, traceback
from dataclasses import dataclass
from typing import Dict, Any,  List
from size_optimizer import OptimizeConfig, optimize_size_async
from leg_quote import LegQuote, TwoLegResult, LEG_OK, LEG_NO_ADDR, LEG_NO_QUOTE, LEG_ERROR
from adapters import build_adapters, best_quote_leg


from logger_utils import init_debug_logger, NullLogger
# 全局兜底，避免未初始化时报 NameError
DBG = NullLogger()


def _init_debug(app, out_html: str, verbose: bool):
    """统一初始化全局 DBG；永远返回一个可用的 logger。"""
    global DBG
    DBG = init_debug_logger(app, out_html, verbose)
    


# =============== 基础设施 ===============
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

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

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

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

# =============== 配置与模型 ===============

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
    chains: Dict[str, Chain]
    assets: Dict[str, Asset]
    settings: Dict[str, Any]

def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chains = {c["name"].lower(): Chain(c["name"].lower(), int(c["chain_id"]), c.get("kyber_slug", c["name"].lower())) for c in raw["chains"]}
    assets = {a["symbol"].upper(): Asset(a["symbol"].upper(), int(a["decimals"]), {k.lower(): v for k, v in a["address"].items()}) for a in raw["assets"]}
    return AppConfig(
        api=raw.get("api", {}),
        chains=chains,
        assets=assets,
        settings=raw.get("settings", {}),
    )

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

# =============== 适配器（Kyber / Odos） ===============

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


async def odos_quote(app: AppConfig, chain: Chain, sell_addr: str, buy_addr: str, amount_wei: int) -> Dict[str, Any]:
    base = app.api.get("odos_base", "https://api.odos.xyz").rstrip("/")
    url = f"{base}/sor/quote/v2"
    payload = {
        "chainId": chain.chain_id,
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

async def kyber_quote(app: AppConfig, chain: Chain, sell_addr: str, buy_addr: str, amount_wei: int) -> Dict[str, Any]:
    base = app.api.get("kyber_base", "https://aggregator-api.kyberswap.com").rstrip("/")
    qs = {
        "tokenIn": sell_addr,
        "tokenOut": buy_addr,
        "amountIn": str(amount_wei),
        "gasInclude": "1",
        "saveGas": "0"
    }
    url1 = f"{base}/{chain.kyber_slug}/api/v1/routes"
    code, text, headers = await _fetch_http("GET", url1, params=qs, timeout=int(app.settings.get("timeout_sec", 12)))
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

# =============== 单腿“合并输出”模式（保留旧功能） ===============

def _from_wei(amount: int, decimals: int) -> float:
    return float(amount) / (10 ** decimals)

def _render_page(title: str, refresh: int, rows: List[dict], debug_html: str = "", head_extra: str = "") -> str:
    esc = lambda s: (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    table_rows = []
    if rows:
        for r in rows:
            table_rows.append(
                f"<tr>"
                f"<td style='text-align:left'>{esc(r['leg'])} "
                f"<span style='color:#64748b'>(base {r['base']} {r['sell']})</span></td>"
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
    body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,'PingFang SC','Microsoft Yahei';margin:20px;color:#0f172a}
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

# 用标准适配器聚合，返回 LegQuote；
async def quote_one_leg_leg(app, chain, a_sym, b_sym, base_in: float) -> "LegQuote":
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


# —— 新增：双腿解析小工具 ——
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

    if a1 != a2 or b1 != b2:
        # 要求第二腿正好是 B>A
        raise ValueError(f"two-leg 第二腿应为 {b1}>{a1}，当前: {b2}>{a2}")
    return ci_name, a1, b1, cj_name

#—— 新增：双腿一次计算 ——
import asyncio

async def quote_two_leg_once(app, ci, cj, a_sym: str, b_sym: str, base_in_a: float) -> TwoLegResult:
    """
    计算两腿：ci 上 A->B，然后 cj 上 B->A。
    成功则 status='ok'，否则返回 no-addr / no-quote / error。
    （不再使用 Decimal，全部用 float 转换）
    """
    # Leg1
    leg1 = await quote_one_leg_leg(app, ci, a_sym, b_sym, base_in_a)
    if leg1.status != LEG_OK or not isinstance(leg1.out_wei, int) or leg1.out_wei <= 0:
        return TwoLegResult(
            from_chain=ci.name, to_chain=cj.name, a=a_sym, b=b_sym,
            base_in_a=base_in_a, leg1=leg1, leg2=None,
            final_a=None, pnl=None, pnl_pct=None,
            status=leg1.status if leg1.status != LEG_OK else LEG_NO_QUOTE,
            note=leg1.note or "leg1 failed"
        )

    # 将 leg1 的 B 输出（wei）转成人类单位，作为第二腿的 base_in
    try:
        B = app.assets[b_sym]
        dB = int(getattr(B, "decimals", 18))
        base_in_b = leg1.out_wei / float(10 ** dB)
    except Exception as e:
        return TwoLegResult(
            from_chain=ci.name, to_chain=cj.name, a=a_sym, b=b_sym,
            base_in_a=base_in_a, leg1=leg1, leg2=None,
            final_a=None, pnl=None, pnl_pct=None,
            status=LEG_ERROR, note=f"bad B meta: {e}"
        )

    # Leg2：加 soft-timeout 防卡
    soft = float(app.settings.get("soft_timeout_sec", 5))
    try:
        leg2 = await asyncio.wait_for(
            quote_one_leg_leg(app, cj, b_sym, a_sym, base_in_b),
            timeout=soft
        )
    except asyncio.TimeoutError:
        leg2 = LegQuote(
            chain=cj.name, a=b_sym, b=a_sym, base_in=base_in_b,
            in_wei=None, out_b=None, out_wei=0, adapter="-",
            status=LEG_NO_QUOTE, note=f"soft-timeout {soft}s"
        )
    except Exception as e:
        leg2 = LegQuote(
            chain=cj.name, a=b_sym, b=a_sym, base_in=base_in_b,
            in_wei=None, out_b=None, out_wei=0, adapter="-",
            status=LEG_ERROR, note=str(e)
        )

    # 计算最终 A、PnL（float）
    try:
        A = app.assets[a_sym]
        dA = int(getattr(A, "decimals", 18))
        final_a = (leg2.out_wei / float(10 ** dA)) if leg2 and leg2.out_wei else None
        pnl = (final_a - base_in_a) if (final_a is not None) else None
        pnl_pct = (pnl / base_in_a * 100.0) if (pnl is not None and base_in_a) else None
        status = LEG_OK if (leg2.status == LEG_OK and final_a is not None) else leg2.status
        note = "" if status == LEG_OK else (leg2.note or "")
        return TwoLegResult(
            from_chain=ci.name, to_chain=cj.name, a=a_sym, b=b_sym,
            base_in_a=base_in_a, leg1=leg1, leg2=leg2,
            final_a=final_a, pnl=pnl, pnl_pct=pnl_pct,
            status=status, note=note
        )
    except Exception as e:
        return TwoLegResult(
            from_chain=ci.name, to_chain=cj.name, a=a_sym, b=b_sym,
            base_in_a=base_in_a, leg1=leg1, leg2=leg2,
            final_a=None, pnl=None, pnl_pct=None,
            status=LEG_ERROR, note=str(e)
        )


# === Single-leg merged mode (render with LegQuote) ===========================
def _derive_pair_any(pair_any: str | None, pair_spec: str | None) -> str:
    """优先用 --pair-any；否则从 --pair-spec 提取第一条腿的 A>B（忽略链前缀与第二腿）。"""
    if pair_any and ">" in pair_any:
        return pair_any.strip()
    if pair_spec:
        first = pair_spec.split(",")[0].strip()        # 取第一条腿
        if ":" in first:                               # 去掉可选链前缀 chain:
            first = first.split(":", 1)[1].strip()
        if ">" in first:
            return first
    raise ValueError(f"需要提供 --pair-any A>B，或 --pair-spec chain:A>B（当前 pair_any={pair_any!r}, pair_spec={pair_spec!r}）")

async def run_merged_mode(cfg_path: str,
                          pair_spec: str,     # <- 兼容 main 的第2个参数（可为空字符串）
                          out_html: str,
                          refresh: int,
                          verbose: bool,
                          once: bool,
                          pair_any: str):     # <- main 的第7个参数
    """
    单腿模式：对 (A>B) 在所有可用链上做一次最优单腿报价，渲染为统一表格。
    依赖：load_config, _init_debug, quote_one_leg_leg, render_legs_page, _write_html
    """
    app = load_config(cfg_path)
    _init_debug(app, out_html, verbose)

    # 解析资产对 A>B
    pair = _derive_pair_any(pair_any, pair_spec)
    a_sym, b_sym = [x.strip().upper() for x in pair.split(">", 1)]

    base_in = float(app.settings.get("default_base_in", 10_000))
    title_base = f"{a_sym}>{b_sym}  (base={base_in:g})"

    loop_idx = 0
    while True:
        loop_idx += 1
        t0 = time.time()

        legs: list[LegQuote] = []
        for ci_name, ci in app.chains.items():
            try:
                leg = await quote_one_leg_leg(app, ci, a_sym, b_sym, base_in)
            except Exception as e:
                leg = LegQuote(
                    chain=getattr(ci, "name", ci_name),
                    a=a_sym, b=b_sym, base_in=base_in,
                    in_wei=None, out_b=None, out_wei=0,
                    adapter="-", status=LEG_ERROR, note=str(e)
                )
            legs.append(leg)

        dur_ms = int((time.time() - t0) * 1000)
        DBG.info("[merged] loop=%d legs=%d dur=%dms", loop_idx, len(legs), dur_ms)

        title = f"Single-leg: {title_base}"
        head_extra = DBG.js_auto_refresh(refresh) if hasattr(DBG, "js_auto_refresh") else ""
        html = render_legs_page(title, refresh, legs, head_extra=head_extra)

        try:
            _write_html(out_html, html)
        except Exception as e:
            DBG.error("write html failed: %s", e)

        try:
            DBG.info("[merged] %s", {
                "mode": "merged",
                "loop": loop_idx,
                "legs_total": len(legs),
                "duration_ms": dur_ms,
                "now": time.strftime("%Y-%m-%d %H:%M:%S"),
                "html_path": out_html,
            })
        except Exception:
            pass

        if once:
            break
        try:
            await asyncio.sleep(int(refresh))
        except asyncio.CancelledError:
            break

#兼容twoleg mode 和watchlist mode
def render_watchlist_page(title: str, refresh: int, rows, head_extra: str = "") -> str:
    css = (
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,'PingFang SC','Microsoft YaHei';"
        "margin:20px;color:#0f172a}"
        ".t{border-collapse:collapse;width:100%}.t th,.t td{border:1px solid #e2e8f0;padding:8px}"
        ".t th{text-align:left;background:#f8fafc}"
        "h2{margin:0 0 12px 0}.sub{color:#475569;margin:0 0 16px 0}"
        ".ok{color:#059669}.no-quote{color:#b45309}.no-addr{color:#b45309}.error{color:#b91c1c}.num{text-align:right}"
    )
    head_auto = head_extra or (DBG.js_auto_refresh(refresh) if hasattr(DBG, "js_auto_refresh") else "")

    body_rows = "".join([r.as_html_row() for r in (rows or [])])
    table = (
        "<table class='t'>"
        "<thead><tr>"
        "<th>From</th><th>To</th><th>A</th><th>B</th><th>Base(A)</th>"
        "<th>Leg1 Adapter</th><th>A→B Out</th>"
        "<th>Leg2 Adapter</th><th>Final A</th><th>PnL(A)</th><th>PnL%</th>"
        "<th>Best Size (A)</th><th>Best PnL (A)</th><th>Best PnL%</th>"
        "<th>Status</th><th>Note</th>"
        "</tr></thead>"
        f"<tbody>{body_rows}</tbody></table>"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"{head_auto}"
        f"<title>{title}</title><style>{css}</style></head>"
        f"<body><h2>{title}</h2><div class='sub'>Auto-refresh: {int(refresh)}s</div>{table}</body></html>"
    )

async def run_two_leg_mode(cfg_path: str,
                           pair_spec: str,
                           out_html: str,
                           refresh: int,
                           verbose: bool,
                           once: bool):
    """
    两腿模式：解析 'chainA:A>B,chainB:B>A'，计算一次两腿并渲染。
    依赖：load_config, _init_debug, _write_html, quote_two_leg_once, render_two_leg_page
    """
    app = load_config(cfg_path)
    _init_debug(app, out_html, verbose)

    ci_name, a_sym, b_sym, cj_name = _parse_two_leg_spec(pair_spec)
    try:
        ci = app.chains[ci_name]; cj = app.chains[cj_name]
    except KeyError as e:
        raise ValueError(f"未知链: {e}")

    base_in_a = float(app.settings.get("default_base_in", 10_000))
    title_base = f"{ci.name}:{a_sym}>{b_sym}, {cj.name}:{b_sym}>{a_sym} (base={base_in_a:g})"

    loop_idx = 0
    while True:
        loop_idx += 1
        t0 = time.time()

        row = await quote_two_leg_once(app, ci, cj, a_sym, b_sym, base_in_a)
        dur_ms = int((time.time() - t0) * 1000)
        DBG.info("[two-leg] loop=%d dur=%dms status=%s pnl=%s", loop_idx, dur_ms, row.status, row.pnl)

        html = render_watchlist_page(f"Two-leg: {title_base}", refresh, [row],
                                   head_extra=(DBG.js_auto_refresh(refresh) if hasattr(DBG, "js_auto_refresh") else ""))

        try:
            _write_html(out_html, html)
        except Exception as e:
            DBG.error("write html failed: %s", e)

        if once:
            break
        try:
            await asyncio.sleep(int(refresh))
        except asyncio.CancelledError:
            break

def _load_watchlist_json(path: str) -> list[dict]:
    """读取 watchlist.json，返回条目列表。允许简写键名，并做基础校验。"""
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
            DBG.warn("[watchlist] item %d not a dict, skip", i); continue

        # 兼容键名：from/to 或 from_chain/to_chain；a/b 或 asset_a/asset_b
        from_chain = it.get("from") or it.get("from_chain")
        to_chain   = it.get("to")   or it.get("to_chain")
        a_sym      = it.get("a")    or it.get("asset_a")
        b_sym      = it.get("b")    or it.get("asset_b")
        base       = it.get("base") or it.get("base_in")  # 可选

        if not (from_chain and to_chain and a_sym and b_sym):
            DBG.warn("[watchlist] item %d missing fields, skip: %s", i, it)
            continue

        norm_list.append({
            "from_chain": str(from_chain),
            "to_chain":   str(to_chain),
            "a":          str(a_sym).upper(),
            "b":          str(b_sym).upper(),
            "base":       float(base) if base is not None else None,
        })

    return norm_list

async def run_watchlist_mode(cfg_path: str,
                             watchlist_path: str,
                             out_html: str,
                             refresh: int,
                             verbose: bool,
                             once: bool):
    """
    Watchlist：并发执行每条两腿报价，逐条渲染为 TwoLegResult。
    并发度与超时从 config.settings 读取（有默认值），不增加 CLI 参数。
    """
    app = load_config(cfg_path)
    _init_debug(app, out_html, verbose)

    default_base   = float(app.settings.get("default_base_in", 10_000))
    max_conc       = int(app.settings.get("watchlist_max_concurrency", 6))
    item_timeout_s = float(app.settings.get("watchlist_item_timeout_sec", 15.0))
    sem = asyncio.Semaphore(max_conc)

    loop_idx = 0
    while True:
        loop_idx += 1
        t0 = time.time()

        try:
            items = _load_watchlist_json(watchlist_path)
        except Exception as e:
            DBG.error("load watchlist failed: %s", e)
            html = f"<!doctype html><meta charset='utf-8'><pre>watchlist error: {e}</pre>"
            try: _write_html(out_html, html)
            except: pass
            return

        async def _handle(idx: int, it: dict):
            ci_name = it["from_chain"]; cj_name = it["to_chain"]
            a_sym   = it["a"];          b_sym   = it["b"]
            base_in = float(it["base"]) if it.get("base") is not None else default_base

            # 先解析链，链名错误直接返回一条错误行
            try:
                ci = app.chains[ci_name]; cj = app.chains[cj_name]
            except KeyError as e:
                return TwoLegResult(
                    from_chain=ci_name, to_chain=cj_name, a=a_sym, b=b_sym,
                    base_in_a=base_in,
                    leg1=LegQuote(ci_name, a_sym, b_sym, base_in, None, None, 0, "-", LEG_ERROR, f"unknown chain: {e}"),
                    leg2=None, final_a=None, pnl=None, pnl_pct=None,
                    status=LEG_ERROR, note=f"unknown chain: {e}"
                )

            # 用信号量限制并发；每个 item 再包一层总超时，避免卡住
            async with sem:
                try:
                    return await asyncio.wait_for(
                        quote_two_leg_once(app, ci, cj, a_sym, b_sym, base_in),
                        timeout=item_timeout_s
                    )
                except asyncio.TimeoutError:
                    return TwoLegResult(
                        from_chain=ci.name, to_chain=cj.name, a=a_sym, b=b_sym,
                        base_in_a=base_in,
                        leg1=LegQuote(ci.name, a_sym, b_sym, base_in, None, None, 0, "-", LEG_NO_QUOTE, f"item-timeout {item_timeout_s}s"),
                        leg2=None, final_a=None, pnl=None, pnl_pct=None,
                        status=LEG_NO_QUOTE, note=f"item-timeout {item_timeout_s}s"
                    )
                except Exception as e:
                    return TwoLegResult(
                        from_chain=ci.name, to_chain=cj.name, a=a_sym, b=b_sym,
                        base_in_a=base_in,
                        leg1=LegQuote(ci.name, a_sym, b_sym, base_in, None, None, 0, "-", LEG_ERROR, "run error"),
                        leg2=None, final_a=None, pnl=None, pnl_pct=None,
                        status=LEG_ERROR, note=str(e)
                    )

        # 保持顺序：gather 会按 tasks 顺序返回
        tasks = [ _handle(i, it) for i, it in enumerate(items) ]
        rows  = await asyncio.gather(*tasks)

        dur_ms = int((time.time() - t0) * 1000)
        DBG.info("[watchlist] loop=%d items=%d dur=%dms conc=%d timeout=%.1fs",
                 loop_idx, len(items), dur_ms, max_conc, item_timeout_s)

        title = "Watchlist (two-leg)"
        head_extra = DBG.js_auto_refresh(refresh) if hasattr(DBG, "js_auto_refresh") else ""
        html = render_watchlist_page(title, refresh, rows, head_extra=head_extra)

        try:
            _write_html(out_html, html)
        except Exception as e:
            DBG.error("write html failed: %s", e)

        if once:
            break
        try:
            await asyncio.sleep(int(refresh))
        except asyncio.CancelledError:
            break

# =============== CLI ===============
# ---- 基础资产：配置与 CLI 覆盖 ----
CLI_BASE_ASSETS: set[str] = set()

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

def main():
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

    # ★ 保存 CLI 覆盖集合
    global CLI_BASE_ASSETS
    CLI_BASE_ASSETS = _parse_base_assets_arg(args.base_assets)

    #让 --watchlist 的优先级高于 --two-leg/--pair-any（避免用户同时传导致歧义）
    if args.watchlist:
        asyncio.run(
            run_watchlist_mode(
                args.config,
                args.watchlist,
                args.pair_html,
                args.refresh,
                args.verbose,
                args.once,
            )
        )
        return

    if args.two_leg:
    # 两腿（watchlist 之前就是手动 pair-spec；我们后面会逐步改成 watchlist 驱动）
        asyncio.run(run_two_leg_mode(args.config, args.pair_spec, args.pair_html, args.refresh, args.verbose, args.once))
    elif args.pair_spec or args.pair_any:
        # 单腿：指定资产对，在所有可用链对比最优单腿兑换
        asyncio.run(run_merged_mode(args.config, args.pair_spec, args.pair_html, args.refresh, args.verbose, args.once, args.pair_any))
    else:
        ap.error("请选择 --two-leg 或 --pair-spec/--pair-any 之一")


if __name__ == "__main__":
    main()
