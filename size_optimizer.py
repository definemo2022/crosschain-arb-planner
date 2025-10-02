# size_optimizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional, Tuple, Union
import math
import inspect
import asyncio

# --------- 类型与结果容器 ---------

Number = Union[int, float]
EvalResult = Union[float, int, Dict[str, Any], Tuple[Any, ...]]

@dataclass
class EvalPoint:
    size: float
    pnl: float

@dataclass
class Bracket:
    a: EvalPoint
    b: EvalPoint
    c: EvalPoint  # 需满足 b.pnl >= a.pnl 且 b.pnl >= c.pnl

@dataclass
class OptimizeResult:
    method: str
    best_size: float
    best_pnl: float
    iterations: int
    evaluations: int
    converged: bool
    reason: str
    history: List[EvalPoint]
    bracket: Optional[Bracket] = None

@dataclass
class OptimizeConfig:
    initial_size: float = 10_000.0
    min_size: float = 1.0
    max_size: float = 10_000_000.0
    grow_factor: float = 2.0               # 翻倍步长
    tol_rel: float = 1e-3                  # 相对容差（区间长度/最佳size）
    tol_abs: float = 1e-6                  # 绝对容差
    max_grows: int = 12                    # 最大翻倍次数
    max_iter: int = 32                     # 细化阶段最大迭代
    max_evals: int = 200                   # 全过程最大评估次数
    prefer_integer_size: bool = False      # 若要 size 取整（比如最小单位）
    method: str = "double_then_golden"     # double_then_golden / double_then_binary / double_then_brent
    # ★ 新增：控制 size 精度/步长
    size_step: Optional[float] = None        # 例如 100；None 表示不限
    round_mode: str = "round"                # "round" | "floor" | "ceil"
# --------- 工具：抽取 PnL / 统一调用同步&异步回调 ---------

def _extract_pnl(x: EvalResult) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        # 常见键位
        for k in ("pnl", "pnl_usd", "profit", "net"):
            if k in x and isinstance(x[k], (int, float)):
                return float(x[k])
        raise ValueError("dict 回调结果缺少可识别的 PnL 字段（期待 'pnl' 等）")
    if isinstance(x, tuple) and x:
        if isinstance(x[0], (int, float)):
            return float(x[0])
        if isinstance(x[0], dict):
            return _extract_pnl(x[0])
    raise ValueError(f"无法从回调结果中提取 PnL: type={type(x)}")

def _quantize_size(x: float, cfg: OptimizeConfig) -> float:
    """将 size 按步长/整型要求量化，并裁剪到 [min_size, max_size]。"""
    q = abs(cfg.size_step) if cfg.size_step else None
    if cfg.prefer_integer_size and not q:
        q = 1.0  # 兼容旧开关：未设步长且要求整数 -> 步长=1
    if not q:
        return max(cfg.min_size, min(cfg.max_size, x))

    inv = x / q
    if cfg.round_mode == "floor":
        k = math.floor(inv)
    elif cfg.round_mode == "ceil":
        k = math.ceil(inv)
    else:
        k = round(inv)
    xq = k * q
    return max(cfg.min_size, min(cfg.max_size, xq))

async def _call_f(func: Callable[[float], EvalResult], size: float) -> float:
    res = func(size)
    if inspect.isawaitable(res):
        res = await res
    return _extract_pnl(res)

# 简单缓存，避免重复评估
class _EvalCache:
    def __init__(self):
        self.memo: Dict[float, float] = {}
        self.order: List[float] = []

    def get(self, size: float) -> Optional[float]:
        return self.memo.get(size)

    def put(self, size: float, pnl: float):
        if size not in self.memo:
            self.order.append(size)
        self.memo[size] = pnl

# --------- 阶段1：翻倍找包围区间（a < b < c, 且 b 为局部峰值） ---------

async def _bracket_max(
    func: Callable[[float], EvalResult],
    cfg: OptimizeConfig,
) -> Tuple[Optional[Bracket], List[EvalPoint], int]:
    """从 initial_size 出发按 grow_factor 扩大，直到出现下降，从而包围住局部最大值。"""
    history: List[EvalPoint] = []
    evals = 0

    def _push(size: float, pnl: float):
        nonlocal evals
        history.append(EvalPoint(size=size, pnl=pnl))
        evals += 1

    # 起点
    x = _quantize_size(max(cfg.min_size, min(cfg.initial_size, cfg.max_size)), cfg)
    y = await _call_f(func, x); _push(x, y)

    # 下探更小 size
    if len(history) == 1 and history[-1].pnl <= 0:
        x2 = _quantize_size(max(cfg.min_size, x / cfg.grow_factor), cfg)
        if x2 < x:
            y2 = await _call_f(func, x2); _push(x2, y2)
            if y2 > y:
                x, y = x2, y2

    # 向上扩张
    prev_x, prev_y = x, y
    for _ in range(cfg.max_grows):
        nxt = _quantize_size(min(cfg.max_size, prev_x * cfg.grow_factor), cfg)
        if cfg.prefer_integer_size:
            nxt = math.floor(nxt)
            if nxt <= prev_x: break
        if nxt <= prev_x + cfg.tol_abs:
            break
        yn = await _call_f(func, nxt); _push(nxt, yn)

        if yn <= prev_y:
            # 找到下降，构造 bracket: (a, b, c) with b=prev
            a = history[-3] if len(history) >= 3 else EvalPoint(size=prev_x / cfg.grow_factor, pnl=prev_y)
            b = EvalPoint(size=prev_x, pnl=prev_y)
            c = EvalPoint(size=nxt, pnl=yn)
            # 兜底：确保 a < b < c
            if a.size >= b.size:
                a = EvalPoint(size=max(cfg.min_size, b.size/cfg.grow_factor), pnl=await _call_f(func, max(cfg.min_size, b.size/cfg.grow_factor)))
                _push(a.size, a.pnl)
            return Bracket(a=a, b=b, c=c), history, evals

        prev_x, prev_y = nxt, yn
        if evals >= cfg.max_evals:
            break

    # 没有出现下降：要么单调增直到上限，要么噪声导致未能观测到峰值
    return None, history, evals

# --------- 阶段2：区间内细化（黄金分割 / 二分最大 / 轻量 Brent） ---------

async def _golden_section_max(
    func: Callable[[float], EvalResult],
    cfg: OptimizeConfig,
    a: float, c: float,      # 搜索区间
    cache: _EvalCache,
) -> Tuple[EvalPoint, List[EvalPoint], int, bool, str]:
    """在 [a, c] 内最大化，返回 (最佳点, 新history, 评估次数, 是否收敛, 原因)"""
    phi = (1 + 5 ** 0.5) / 2
    resphi = 2 - phi  # 1/phi^2 ≈ 0.381966
    b = a + resphi * (c - a)

    history: List[EvalPoint] = []
    evals = 0

    async def eval_at(x: float) -> float:
        x = _quantize_size(x, cfg)
        nonlocal evals
        if cfg.prefer_integer_size: x = round(x)
        x = max(cfg.min_size, min(cfg.max_size, x))
        y = cache.get(x)
        if y is None:
            y = await _call_f(func, x)
            cache.put(x, y)
            history.append(EvalPoint(size=x, pnl=y))
            evals += 1
        return y

    ya = await eval_at(a)
    yb = await eval_at(b)
    yc = await eval_at(c)

    best = max([EvalPoint(a, ya), EvalPoint(b, yb), EvalPoint(c, yc)], key=lambda p: p.pnl)

    it = 0
    while it < cfg.max_iter and evals < cfg.max_evals:
        it += 1
        # 两端与中点的距离
        if abs(c - a) <= max(cfg.tol_abs, cfg.tol_rel * abs(best.size)):
            return best, history, evals, True, "tolerance_reached"

        # 维持 b 为内部点；比较 yb 与两端，收缩区间
        # 为简化：再引入一个内部点 d
        if (best.size - a) > (c - best.size):
            # 左侧更宽，在左侧放一个新点
            d = a + resphi * (best.size - a)
            yd = await eval_at(d)
            if yd > ya:
                c, yc = best.size, best.pnl
                best = EvalPoint(d, yd) if yd > best.pnl else best
                b, yb = d, yd
            else:
                a, ya = d, yd
        else:
            # 右侧更宽，在右侧放一个新点
            d = best.size + resphi * (c - best.size)
            yd = await eval_at(d)
            if yd > yc:
                a, ya = best.size, best.pnl
                best = EvalPoint(d, yd) if yd > best.pnl else best
                b, yb = d, yd
            else:
                c, yc = d, yd

    return best, history, evals, False, "max_iter_or_evals"

async def _binary_refine_max(
    func: Callable[[float], EvalResult],
    cfg: OptimizeConfig,
    a: float, c: float,
    cache: _EvalCache,
) -> Tuple[EvalPoint, List[EvalPoint], int, bool, str]:
    """朴素二分：比较中点左右邻域，向更优一边收缩。"""
    history: List[EvalPoint] = []
    evals = 0

    async def eval_at(x: float) -> float:
        x = _quantize_size(x, cfg)
        nonlocal evals
        if cfg.prefer_integer_size: x = round(x)
        x = max(cfg.min_size, min(cfg.max_size, x))
        y = cache.get(x)
        if y is None:
            y = await _call_f(func, x)
            cache.put(x, y)
            history.append(EvalPoint(size=x, pnl=y))
            evals += 1
        return y

    left, right = a, c
    best = EvalPoint(size=a, pnl=await eval_at(a))
    _ = await eval_at(c)  # 评估右端，录入缓存
    if cache.memo[c] > best.pnl:
        best = EvalPoint(c, cache.memo[c])

    it = 0
    while it < cfg.max_iter and evals < cfg.max_evals:
        it += 1
        mid = 0.5 * (left + right)
        y_mid = await eval_at(mid)

        # 在 mid 左右各取一个点（步长为当前区间的 1/8）
        step = max(cfg.tol_abs, 0.125 * (right - left))
        lpt = max(cfg.min_size, mid - step)
        rpt = min(cfg.max_size, mid + step)
        y_l = await eval_at(lpt)
        y_r = await eval_at(rpt)

        # 更新最优
        for (sx, sy) in ((mid, y_mid), (lpt, y_l), (rpt, y_r)):
            if sy > best.pnl:
                best = EvalPoint(sx, sy)

        # 收缩方向
        if y_l > y_r:
            right = rpt
        else:
            left = lpt

        if abs(right - left) <= max(cfg.tol_abs, cfg.tol_rel * abs(best.size)):
            return best, history, evals, True, "tolerance_reached"

    return best, history, evals, False, "max_iter_or_evals"

async def _brent_like_max(
    func: Callable[[float], EvalResult],
    cfg: OptimizeConfig,
    a: float, c: float,
    cache: _EvalCache,
) -> Tuple[EvalPoint, List[EvalPoint], int, bool, str]:
    """
    轻量版 Brent：在 [a,c] 内混合抛物线插值与黄金分割。
    注：为保持简洁与稳健性，这里实现的是简化版，实务上 golden 已足够。
    """
    history: List[EvalPoint] = []
    evals = 0

    async def eval_at(x: float) -> float:
        x = _quantize_size(x, cfg)
        nonlocal evals
        if cfg.prefer_integer_size: x = round(x)
        x = max(cfg.min_size, min(cfg.max_size, x))
        y = cache.get(x)
        if y is None:
            y = await _call_f(func, x)
            cache.put(x, y)
            history.append(EvalPoint(size=x, pnl=y))
            evals += 1
        return y

    # 初始点
    x = 0.5 * (a + c)
    w = v = x
    fx = await eval_at(x)
    fw = fv = fx

    phi = (3 - 5 ** 0.5) / 2  # ≈ 0.381966
    tol = lambda s: max(cfg.tol_abs, cfg.tol_rel * abs(s))
    it = 0

    while it < cfg.max_iter and evals < cfg.max_evals:
        it += 1
        m = 0.5 * (a + c)
        if abs(x - m) <= (2 * tol(x) - 0.5 * (c - a)):
            return EvalPoint(x, fx), history, evals, True, "tolerance_reached"

        # 抛物线插值尝试
        u = None
        d1 = (w - x) * (fw - fv)
        d2 = (v - x) * (fw - fx)
        denom = 2 * (d1 - d2) if (d1 - d2) != 0 else None
        if denom:
            u_tmp = x - ((w - x) ** 2 * (fw - fv) - (v - x) ** 2 * (fw - fx)) / denom
            if a + tol(x) <= u_tmp <= c - tol(x):
                u = u_tmp

        if u is None:
            # 回退黄金分割步
            if x < m:
                u = x + phi * (c - x)
            else:
                u = x - phi * (x - a)

        fu = await eval_at(u)

        # 最大化：若 fu > fx，则移动三点
        if fu > fx:
            if u < x: c = x
            else: a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x: a = u
            else: c = u
            if fu >= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu >= fv or v == x or v == w:
                v, fv = u, fu

    return EvalPoint(x, fx), history, evals, False, "max_iter_or_evals"

# --------- 对外：异步 / 同步接口 ---------

async def optimize_size_async(
    func: Callable[[float], EvalResult],
    cfg: OptimizeConfig,
) -> OptimizeResult:
    """
    func(size) -> pnl（可返回 float/int，或 dict 含 'pnl'，或 (pnl, ...) tuple）。
    异步友好：func 可为 async def。
    """
    cache = _EvalCache()
    full_hist: List[EvalPoint] = []

    # 阶段1：翻倍找到包围区间
    bracket, hist, n_eval1 = await _bracket_max(func, cfg)
    full_hist.extend(hist)

    # 若没能形成有效 bracket（单调增/减），直接在已评估点里挑最大，或在边界做一次探测
    if bracket is None:
        # 尝试边界（若还没评估过）
        for sz in (cfg.min_size, cfg.max_size):
            if cache.get(sz) is None and len(full_hist) < cfg.max_evals:
                y = await _call_f(func, sz)
                cache.put(sz, y)
                full_hist.append(EvalPoint(sz, y))
        best = max(full_hist, key=lambda p: p.pnl)
        return OptimizeResult(
            method=cfg.method,
            best_size=best.size,
            best_pnl=best.pnl,
            iterations=0,
            evaluations=len(full_hist),
            converged=False,
            reason="no_bracket_monotone_or_cap",
            history=full_hist,
            bracket=None,
        )

    a, b, c = bracket.a.size, bracket.b.size, bracket.c.size

    # 阶段2：细化
    if cfg.method == "double_then_binary":
        best, h2, ne2, ok, why = await _binary_refine_max(func, cfg, a, c, cache)
    elif cfg.method == "double_then_brent":
        best, h2, ne2, ok, why = await _brent_like_max(func, cfg, a, c, cache)
    else:
        best, h2, ne2, ok, why = await _golden_section_max(func, cfg, a, c, cache)

    full_hist.extend(h2)
    return OptimizeResult(
        method=cfg.method,
        best_size=best.size,
        best_pnl=best.pnl,
        iterations=len(h2),
        evaluations=len(full_hist),
        converged=ok,
        reason=why,
        history=full_hist,
        bracket=bracket,
    )

def optimize_size(
    func: Callable[[float], EvalResult],
    cfg: OptimizeConfig,
) -> OptimizeResult:
    """同步版本：如 func 为异步，这里会自动跑一个事件循环来等待。"""
    if inspect.iscoroutinefunction(func):
        return asyncio.run(optimize_size_async(func, cfg))
    # 同步 func
    async def _wrap(x: float):
        return func(x)
    return asyncio.run(optimize_size_async(_wrap, cfg))

# --------- 直接运行：一个小示例（可选） ---------
if __name__ == "__main__":
    # 构造一个假想 PnL 曲线（先增后减），最大值在 ~ 120k
    import random
    def pnl_fn(size: float) -> float:
        peak = 120_000.0
        # 上升段 + 饱和衰减：只是示意
        val = (1 - math.exp(-size / 30_000.0)) - 0.5 * (size / peak)
        # 加点噪声
        return 10_000 * val + random.uniform(-50, 50)

    cfg = OptimizeConfig(
        initial_size=10_000, min_size=1_000, max_size=500_000,
        method="double_then_golden", prefer_integer_size=False
    )
    res = optimize_size(pnl_fn, cfg)
    print(res)
