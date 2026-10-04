"""Reproduce the paper's correction baselines on frozen score matrices.

Self-contained re-implementation of the oracle user with three modes: joint (whole-path
re-decode), frozen (only the touched event moves; an out-of-order choice is applied anyway,
as in correction_sim), and direct (no DP: each event starts at its own best frame).
joint/frozen match run_correction_eval exactly; direct is the no-DP row of Table 3.

Usage: PYTHONPATH=src:. aic/bin/python -m scripts.evaluation.correction_baselines runs/correction-eval-v1
"""
import collections, json, sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "runs/correction-eval-v1"
LAM, TOL, SEP, BMAX = 1e-5, 1000, 5000, 5
META = [json.loads(l) for l in open(f"{R}/meta.jsonl")]

def dp(S, t, fixed, banned):
    M, F = S.shape
    A = S.copy()
    for i in range(M):
        A[i, list(banned[i])] = -np.inf
        if i in fixed:
            keep = A[i, fixed[i]]; A[i] = -np.inf; A[i, fixed[i]] = keep
    best = A[0].copy(); back = []
    for i in range(1, M):
        v = best + LAM * t
        pm = np.maximum.accumulate(v); pa = np.zeros(F, int)
        idx = np.arange(F); arg = np.where(v == pm, idx, 0); arg = np.maximum.accumulate(arg)
        prev = np.concatenate([[-np.inf], pm[:-1]]); parg = np.concatenate([[0], arg[:-1]])
        best = A[i] + prev - LAM * t; back.append(parg)
    f = [int(np.argmax(best))]
    for b in reversed(back): f.append(int(b[f[-1]]))
    return f[::-1]

def ok(t, gt, i, f): return gt[i, 0] - TOL <= t[f] <= gt[i, 1] + TOL

def local_alts(S, t, i, cur, banned, k):
    out = []
    for f in np.argsort(-S[i]):
        if f == cur or f in banned[i]: continue
        if all(abs(t[f] - t[g]) >= SEP for g in out + [cur]): out.append(int(f))
        if len(out) == k: break
    return out

def run(S, t, gt, k, mode):
    M = S.shape[0]; fixed = {}; banned = [set() for _ in range(M)]
    path = dp(S, t, fixed, banned) if mode != "direct" else [int(np.argmax(S[i])) for i in range(M)]
    frames = 0
    for a in range(BMAX + 1):
        if all(ok(t, gt, i, path[i]) for i in range(M)): return a, frames
        if a == BMAX: return None, frames
        i = next(j for j in range(M) if j not in fixed)
        alts = local_alts(S, t, i, path[i], banned, k); frames += 1 + len(alts)
        if ok(t, gt, i, path[i]): fixed[i] = path[i]
        else:
            good = [f for f in alts if ok(t, gt, i, f)]
            if good: fixed[i] = good[0]
            else: banned[i].add(path[i])
        if mode == "joint": path = dp(S, t, fixed, banned)
        elif mode == "frozen":
            others = {j: path[j] for j in range(M) if j != i}
            path = dp(S, t, {**others, **fixed}, banned) if i not in fixed else [fixed.get(j, path[j]) for j in range(M)]
        else:
            path[i] = fixed[i] if i in fixed else max((f for f in range(S.shape[1]) if f not in banned[i]), key=lambda f: S[i, f])
    return None, frames

def load(q):
    z = np.load(f"{R}/{q}.npz"); return z["scores"], z["timestamps_ms"].astype(float), z["gt_ms"]


D = {m["query_id"]: (m, load(m["query_id"])) for m in META}
def hb(res, B): return np.mean([a is not None and a <= B for a in res])
out = {}
for k in (4, 8):
    for mode in ("joint", "frozen", "direct"):
        res = {q: run(S, t, gt, k, mode) for q, (m, (S, t, gt)) in D.items()}
        out[(k, mode)] = res
        a = [r[0] for r in res.values()]
        fr = np.mean([r[1] for r in res.values()])
        print(f"k={k} {mode:6s} H@0..5", [round(hb(a, b), 3) for b in range(6)], f"mean frames inspected {fr:.1f}")
# breakdown, k=4 joint
init_wrong = {}
for q, (m, (S, t, gt)) in D.items():
    p = dp(S, t, {}, [set() for _ in range(S.shape[0])])
    init_wrong[q] = sum(not ok(t, gt, i, p[i]) for i in range(S.shape[0]))
for k in (4, 8):
  for mode in ("joint", "direct"):
    res = out[(k, mode)]
    for key, f in [("n_events", lambda q: D[q][0]["n_events"]), ("init_wrong", lambda q: init_wrong[q])]:
        g = collections.defaultdict(list)
        for q, r in res.items(): g[f(q)].append(r[0])
        print(k, mode, key, {v: (len(a), round(hb(a, 0), 2), round(hb(a, 5), 2)) for v, a in sorted(g.items())})
# end-to-end: only top-1 video counts
for k in (4, 8):
    for mode in ("joint", "direct"):
        a = [r[0] if D[q][0]["video_rank"] == 1 else None for q, r in out[(k, mode)].items()]
        print("e2e", k, mode, "rank1", sum(D[q][0]["video_rank"] == 1 for q in D), [round(hb(a, b), 3) for b in (0, 5)])
# paired diff joint vs direct H@5 k=4, bootstrap
for k in (4, 8):
    j = np.array([out[(k, "joint")][q][0] is not None for q in D]); d = np.array([out[(k, "direct")][q][0] is not None for q in D])
    fz = np.array([out[(k, "frozen")][q][0] is not None for q in D])
    rng = np.random.default_rng(0); n = len(j)
    for name, x in (("joint-direct", j.astype(int) - d), ("joint-frozen", j.astype(int) - fz)):
        bs = [x[rng.integers(0, n, n)].mean() for _ in range(5000)]
        print(k, name, x.mean(), np.percentile(bs, [2.5, 97.5]), "wins/losses", (x > 0).sum(), (x < 0).sum())
# tolerance sensitivity
for tol in (0, 1000, 2000, 3000):
    TOL = tol
    res = [run(S, t, gt, 4, "joint")[0] for q, (m, (S, t, gt)) in D.items()]
    print("tol", tol, "H@0", hb(res, 0), "H@5", hb(res, 5))
TOL = 1000
L = [gt[:, 1] - gt[:, 0] for q, (m, (S, t, gt)) in D.items()]
L = np.concatenate(L) / 1000; print("interval s: median", np.median(L), "p25/p75", np.percentile(L, [25, 75]), "n", len(L))
