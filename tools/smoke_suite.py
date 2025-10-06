import subprocess, sys, os, datetime, argparse, shlex

def run(name, py, script, cfg, args):
    cmd = [py, script, "--config", cfg] + args
    print(f"==> {name}\n$ {' '.join(shlex.quote(x) for x in cmd)}")
    code = subprocess.call(cmd)
    print(("[PASS]" if code==0 else f"[FAIL] exit={code}") + f" {name}\n")
    return code

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="python")
    ap.add_argument("--script", default="./crosschain_arbitrage_planner.py")
    ap.add_argument("--config", default="./config.json")
    ap.add_argument("--once", action="store_true", default=True)
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--refresh", default="8")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = "./smoke_out"
    os.makedirs(outdir, exist_ok=True)

    base = ["--refresh", args.refresh]
    if args.once: base += ["--once"]
    if args.verbose: base += ["--verbose"]

    # 1) single
    run("single: ethereum USDT>USDC", args.python, args.script, args.config,
        ["--pair-spec","ethereum:USDT>USDC","--pair-html",f"{outdir}/single_eth_usdt_usdc_{ts}.html"] + base)

    # 2) two-leg
    run("two-leg: eth→arb→eth USDT/USDC", args.python, args.script, args.config,
        ["--two-leg","--pair-spec","ethereum:USDT>USDC,arbitrum:USDC>USDT","--pair-html",f"{outdir}/twoleg_eth_arb_usdt_usdc_{ts}.html"] + base)

    # 3) watchlist
    watch = "./watchlist.json"
    if not os.path.exists(watch):
        open(watch,"w",encoding="utf-8").write('[{"A_chain":"ethereum","B_chain":"arbitrum","A":"USDT","B":"USDC","base":10000}]')
    run("watchlist: sample", args.python, args.script, args.config,
        ["--watchlist",watch,"--pair-html",f"{outdir}/watchlist_{ts}.html"] + base)

if __name__=="__main__":
    sys.exit(main())
