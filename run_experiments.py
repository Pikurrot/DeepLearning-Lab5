#!/usr/bin/env python
"""Double-descent experiments (Belkin et al. 2018, App. C.3) -- resumable background runner.

Usage:
    uv run python run_experiments.py                # all jobs, 2 worker processes x 2 threads
    uv run python run_experiments.py --workers 1    # even lighter
    uv run python run_experiments.py --status       # show progress and exit
    uv run python run_experiments.py --quick        # tiny smoke test (writes to results_quick/)

Resumable: every finished width is appended to results/<job>/rows.jsonl, and the last
under-parametrised model (needed for weight reuse) is checkpointed. Just run it again
after any interruption (Ctrl-C, reboot, sleep...). Finished jobs are skipped.
Runs on CPU with few threads and a lowered priority so the machine stays usable.
"""
import argparse, json, os, sys, time, math
from pathlib import Path
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent
K, D, N_TRAIN = 10, 784, 4000
H_STAR = (N_TRAIN * K - K) / (D + 1 + K)

# ----------------------------------------------------------------------------- experiment design
WIDTHS_FULL = [3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100, 130, 170, 220,
               300, 400, 550, 700, 800]
WIDTHS_ABL = [3, 8, 16, 25, 35, 45, 50, 55, 65, 80, 110, 170, 300, 550, 800]
PAPER_EPOCHS = 6000     # paper-faithful runs (main experiment + label noise)
ABL_EPOCHS = 2000       # ablations are run with a reduced budget, all compared against baseline @2000

BASE = dict(epochs=PAPER_EPOCHS, noise=0.0, reuse=True, loss="mse", lr=0.01, momentum=0.95,
            bs=128, act="relu", init="glorot", seed=0, subset_seed=0)


def make_jobs():
    J = []
    def add(name, group, widths, **kw):
        c = dict(BASE); c.update(kw)
        J.append(dict(name=name, group=group, widths=widths, **c))
    # 1c/1d: paper-faithful main experiment
    add("main", "main", WIDTHS_FULL)
    # 1f: label noise (paper-faithful budget)
    for p in (0.1, 0.2):
        add(f"noise_{p}", "noise", WIDTHS_FULL, noise=p)
    # 1e: ablations (reduced budget, own baseline)
    A = dict(epochs=ABL_EPOCHS)
    add("abl_baseline", "ablation", WIDTHS_ABL, **A)
    add("abl_no_reuse", "ablation", WIDTHS_ABL, **{**A, "reuse": False})
    add("abl_cross_entropy", "ablation", WIDTHS_ABL, **{**A, "loss": "ce"})
    add("abl_sigmoid", "ablation", WIDTHS_ABL, **{**A, "act": "sigmoid"})
    add("abl_tanh", "ablation", WIDTHS_ABL, **{**A, "act": "tanh"})
    add("abl_epochs_400", "ablation", WIDTHS_ABL, **{**A, "epochs": 400})
    add("abl_epochs_100", "ablation", WIDTHS_ABL, **{**A, "epochs": 100})
    add("abl_no_momentum", "ablation", WIDTHS_ABL, **{**A, "momentum": 0.0, "lr": 0.2})
    add("abl_lr_small", "ablation", WIDTHS_ABL, **{**A, "lr": 0.002})
    add("abl_lr_large", "ablation", WIDTHS_ABL, **{**A, "lr": 0.04})
    add("abl_kaiming", "ablation", WIDTHS_ABL, **{**A, "init": "kaiming"})
    add("abl_bs32", "ablation", WIDTHS_ABL, **{**A, "bs": 32, "epochs": 1000})
    add("abl_bs1000", "ablation", WIDTHS_ABL, **{**A, "bs": 1000})
    for s in (1, 2, 3):
        add(f"abl_seed{s}", "seeds", WIDTHS_ABL, **{**A, "seed": s})
    for s in (1, 2):
        add(f"abl_subset{s}", "seeds", WIDTHS_ABL, **{**A, "subset_seed": s})
    return J


def job_cost(j):   # rough relative cost for load balancing
    return sum(j["epochs"] * (0.4 + h / 800) for h in j["widths"])


def n_params(h):
    return (D + 1) * h + (h + 1) * K


def log(msg, wid=None):
    tag = f"[w{wid}] " if wid is not None else ""
    print(f"{time.strftime('%H:%M:%S')} {tag}{msg}", flush=True)


# ----------------------------------------------------------------------------- worker
def worker(wid, job_names, results_dir, threads, quick):
    import torch, torch.nn as nn, torch.nn.functional as F
    os.nice(10)
    torch.set_num_threads(threads)
    jobs = {j["name"]: j for j in make_jobs()}

    from torchvision import datasets
    tr = datasets.MNIST(ROOT / "data", train=True, download=True)
    te = datasets.MNIST(ROOT / "data", train=False, download=True)
    Xall, yall = tr.data.float().div(255).flatten(1), tr.targets
    Xte, yte = te.data.float().div(255).flatten(1), te.targets
    ACTS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}

    class MLP(nn.Module):
        def __init__(self, hidden, act="relu", init="glorot"):
            super().__init__()
            self.hidden, self.act_name = hidden, act
            self.fc1, self.act, self.fc2 = nn.Linear(D, hidden), ACTS[act](), nn.Linear(hidden, K)
            for m in (self.fc1, self.fc2):
                if init == "glorot": nn.init.xavier_uniform_(m.weight)
                elif init == "normal": nn.init.normal_(m.weight, 0, 0.1)   # variance 0.01
                elif init == "kaiming": nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        def forward(self, x): return self.fc2(self.act(self.fc1(x)))

    def grow(small, hidden):
        """weight reuse: first units from the trained smaller net, new weights ~ N(0, 0.01)"""
        big, h = MLP(hidden, small.act_name, "normal"), small.hidden
        with torch.no_grad():
            big.fc1.weight[:h] = small.fc1.weight; big.fc1.bias[:h] = small.fc1.bias
            big.fc2.weight[:, :h] = small.fc2.weight; big.fc2.bias.copy_(small.fc2.bias)
        return big

    @torch.no_grad()
    def evaluate(model, X, y):
        out = model(X)
        return (out.argmax(1) != y).float().mean().item(), ((out - F.one_hot(y, K).float()) ** 2).sum(1).mean().item()

    def corrupt(y, p, seed):
        if p == 0: return y.clone()
        g = torch.Generator().manual_seed(1234 + seed)
        m = int(round(p * len(y)))
        sel = torch.randperm(len(y), generator=g)[:m]
        shift = torch.randint(1, K, (m,), generator=g)
        y2 = y.clone(); y2[sel] = (y[sel] + shift) % K
        return y2

    def train(model, X, y, c, under, H, tag):
        epochs = c["epochs"]
        opt = torch.optim.SGD(model.parameters(), lr=c["lr"], momentum=c["momentum"])
        Y1 = F.one_hot(y, K).float()
        decay_every = max(1, epochs // 12)          # paper: every 500 of 6000 epochs
        tlast = time.time()
        for ep in range(epochs):
            if under and ep > 0 and ep % decay_every == 0:
                for gp in opt.param_groups: gp["lr"] *= 0.9
            perm = torch.randperm(len(X))
            for i in range(0, len(X), c["bs"]):
                b = perm[i:i + c["bs"]]
                out = model(X[b])
                l = F.mse_loss(out, Y1[b]) if c["loss"] == "mse" else F.cross_entropy(out, y[b])
                opt.zero_grad(); l.backward(); opt.step()
            if (ep + 1) % 10 == 0:
                err, sq = evaluate(model, X, y)
                if not math.isfinite(sq): return ep + 1, True
                if under and err == 0: return ep + 1, False
                if time.time() - tlast > 120:
                    tlast = time.time()
                    log(f"  {tag} H={H} epoch {ep+1}/{epochs} train_err={err:.4f} sq={sq:.4f}", wid)
        return epochs, False

    for name in job_names:
        j = jobs[name]
        if quick:
            j = dict(j, epochs=min(j["epochs"], 30), widths=[3, 25, 50, 80])
        d = Path(results_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        if (d / "DONE").exists():
            continue
        rows_f = d / "rows.jsonl"
        rows = [json.loads(l) for l in rows_f.read_text().splitlines()] if rows_f.exists() else []
        done_w = {r["hidden"] for r in rows}
        (d / "config.json").write_text(json.dumps(j, indent=1))

        g = torch.Generator().manual_seed(j["subset_seed"])
        idx = torch.randperm(len(Xall), generator=g)[:N_TRAIN]
        Xtr, ytr_clean = Xall[idx], yall[idx]
        ytr = corrupt(ytr_clean, j["noise"], j["seed"])

        prev = None
        ck = d / "prev.pt"
        if ck.exists():
            s = torch.load(ck)
            prev = MLP(s["hidden"], j["act"]); prev.load_state_dict(s["sd"])
        log(f"START job {name} ({len(done_w)}/{len(j['widths'])} widths already done)", wid)
        t0 = time.time()
        for H in j["widths"]:
            if H in done_w: continue
            torch.manual_seed(j["seed"] * 100003 + H)
            under = n_params(H) < N_TRAIN * K
            model = grow(prev, H) if (j["reuse"] and under and prev is not None) else MLP(H, j["act"], j["init"])
            ts = time.time()
            ep, diverged = train(model, Xtr, ytr, j, under, H, name)
            tr_err, tr_sq = evaluate(model, Xtr, ytr)
            trc_err, _ = evaluate(model, Xtr, ytr_clean)
            te_err, te_sq = evaluate(model, Xte, yte)
            row = dict(name=name, seed=j["seed"], noise=j["noise"], hidden=H, n_params=n_params(H),
                       epochs_run=ep, diverged=diverged, train_err=tr_err, train_sq=tr_sq,
                       train_err_clean=trc_err, test_err=te_err, test_sq=te_sq,
                       secs=time.time() - ts)
            with open(rows_f, "a") as f: f.write(json.dumps(row) + "\n")
            if under and j["reuse"]:
                torch.save(dict(hidden=H, sd=model.state_dict()), ck)
            prev = model if under else prev
            log(f"{name} H={H:4d} params={n_params(H):6d} ep={ep:5d} | train {tr_err:.4f}/{tr_sq:.4f} "
                f"test {te_err:.4f}/{te_sq:.4f} | {time.time()-ts:.0f}s{' DIVERGED' if diverged else ''}", wid)
        (d / "DONE").write_text(time.strftime("%c"))
        if ck.exists(): ck.unlink()
        log(f"FINISHED job {name} in {(time.time()-t0)/60:.1f} min", wid)


# ----------------------------------------------------------------------------- orchestration
def status(results_dir, jobs):
    tot_w = done_w = 0
    for j in jobs:
        d = Path(results_dir) / j["name"]
        n = len((d / "rows.jsonl").read_text().splitlines()) if (d / "rows.jsonl").exists() else 0
        fin = (d / "DONE").exists()
        tot_w += len(j["widths"]); done_w += len(j["widths"]) if fin else n
        print(f"  {'[x]' if fin else '[ ]'} {j['name']:20s} {n:2d}/{len(j['widths'])} widths", flush=True)
    print(f"overall: {done_w}/{tot_w} (job,width) points", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    results_dir = ROOT / ("results_quick" if a.quick else "results")
    jobs = make_jobs()
    if a.status:
        status(results_dir, jobs); return
    pending = [j for j in jobs if not (results_dir / j["name"] / "DONE").exists()]
    log(f"{len(pending)} pending jobs of {len(jobs)}; {a.workers} workers x {a.threads} threads; results -> {results_dir}")
    # greedy balance by cost (longest first)
    queues, load = [[] for _ in range(a.workers)], [0.0] * a.workers
    for j in sorted(pending, key=job_cost, reverse=True):
        i = load.index(min(load)); queues[i].append(j["name"]); load[i] += job_cost(j)
    for i, q in enumerate(queues): log(f"worker {i}: {q}")
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=(i, q, results_dir, a.threads, a.quick)) for i, q in enumerate(queues) if q]
    for p in ps: p.start()
    try:
        last = time.time()
        while any(p.is_alive() for p in ps):
            time.sleep(5)
            if time.time() - last > 600:
                last = time.time(); log("---- status ----"); status(results_dir, jobs)
    except KeyboardInterrupt:
        log("interrupted; terminating workers (re-run to resume)")
        for p in ps: p.terminate()
    for p in ps: p.join()
    log("all done" if all(p.exitcode == 0 for p in ps) else "a worker exited abnormally; re-run to resume")
    status(results_dir, jobs)


if __name__ == "__main__":
    main()
