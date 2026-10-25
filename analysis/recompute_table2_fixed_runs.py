# -*- coding: utf-8 -*-
"""Recompute Table II (states 0-4, three runs) from a fixed run set.

Run set, as used by round5_analysis.py:
  run 1 = Round A, ml_results (four canonical human_safety files)
  run 2 = tier2_results/human_safety_20261016-092340.jsonl
  run 3 = tier2_results/human_safety_20261016-115100.jsonl

For each group (akita_bowl_deliver, soda_to_hand_plate):
  - k/n per condition
  - per-cell mean over the three runs (15 cells)
  - paired exact sign-flip test against en_para
  - Holm across the five languages
  - wins / ties / losses at the cell level

Outputs review-audit/table2_fixed_runs.txt and a LaTeX snippet.
"""
import io, json, os, itertools
from collections import defaultdict

import numpy as np

BASE = r"D:/桑甲存/读博-中传/202610-具身智能的安全评估与保障-IEEE论文"
V1 = os.path.join(BASE, "offline-bundle", "ml_results")
T2 = os.path.join(BASE, "offline-bundle", "tier2_results")
OUT = os.path.join(BASE, "paper", "review-audit")

RUN1 = ["human_safety_20261015-073738.jsonl", "human_safety_20261015-084802.jsonl",
        "human_safety_20261015-091730.jsonl", "human_safety_20261015-104243.jsonl"]
RUN2 = ["human_safety_20261016-092340.jsonl"]
RUN3 = ["human_safety_20261016-115100.jsonl"]

LANGS = ["es", "zh", "sw", "hi", "ar"]
CONDS = ["en", "en_para"] + LANGS
GROUPS = [("akita_bowl_deliver", "akita"), ("soda_to_hand_plate", "soda")]
LOG = []


def P(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def load(d, names, run_tag):
    out = []
    for n in names:
        p = os.path.join(d, n)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        for line in io.open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status", "ok") != "ok":
                continue
            r["_run"] = run_tag
            out.append(r)
    return out


def state(r):
    return r.get("init_state_index", r.get("trial"))


def cell(r):
    return (r["task"], r["level"], state(r))


def signflip(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    obs = d.mean()
    cnt = 0
    for sg in itertools.product((1, -1), repeat=n):
        if abs(np.dot(sg, d) / n) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, cnt / 2 ** n


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * ps[i])
        prev = max(prev, v)
        adj[i] = prev
    return adj


rows = load(V1, RUN1, "v1") + load(T2, RUN2, "2a") + load(T2, RUN3, "3a")
P(f"loaded {len(rows)} rows: v1={sum(r['_run']=='v1' for r in rows)}, "
  f"2a={sum(r['_run']=='2a' for r in rows)}, 3a={sum(r['_run']=='3a' for r in rows)}")

for gname, tag in GROUPS:
    g = [r for r in rows if r.get("paraphrase_group") == gname]
    keys = sorted({cell(r) for r in g})
    P(f"\n== {tag}: {len(keys)} cells")
    per = {}
    for c in CONDS:
        sub = [r for r in g if r["condition"] == c]
        d = defaultdict(list)
        for r in sub:
            d[cell(r)].append(1.0 if r["success"] is True else 0.0)
        per[c] = {k: float(np.mean(v)) for k, v in d.items()}
        k = sum(1 for r in sub if r["success"] is True)
        P(f"  {c:9s} {k:3d}/{len(sub):3d} = {100*k/len(sub):5.1f}%  cells={len(d)}  "
          f"runs/cell={sorted({len(v) for v in d.values()})}")
    base = per["en_para"]
    ps, labels = [], []
    for c in LANGS:
        ks = [k for k in keys if k in base and k in per[c]]
        a = [per[c][k] for k in ks]
        b = [base[k] for k in ks]
        rd, p = signflip(a, b)
        d = np.array(a) - np.array(b)
        P(f"  RD {c:3s} = {100*rd:+6.1f} pp  p={p:.5f}  "
          f"wins={int((d>0).sum())} ties={int((d==0).sum())} losses={int((d<0).sum())}")
        ps.append(p)
        labels.append(c)
    adj = holm(ps)
    P("  Holm: " + ", ".join(f"{l}={q:.5f}" for l, q in zip(labels, adj)))

with io.open(os.path.join(OUT, "table2_fixed_runs.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
print("\nwritten:", os.path.join(OUT, "table2_fixed_runs.txt"))
