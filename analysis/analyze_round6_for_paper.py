# -*- coding: utf-8 -*-
"""Round 6 results -> numbers for the paper.

Sources
  new  : second_model_offline_batch/round6/round6_results/*.jsonl
  hist : offline-bundle/round4_results/human_safety_20261018-041809.jsonl (4b2, Spanish soda)

Computes, for soda:
  6a  en / zh / cs_en_noun_zh, new batch only
  6b  en / es / cs_en_noun_es / cs_es_noun_en, new batch and pooled with 4b2
Paired cell-level sign-flip over the 15 cells, bootstrap CI, W/T/L.
"""
import glob, io, itertools, json, os
from collections import defaultdict

import numpy as np

BASE = r"D:/桑甲存/读博-中传/202610-具身智能的安全评估与保障-IEEE论文"
NEW = os.path.join(BASE, "second_model_offline_batch", "round6", "round6_results")
HIST = os.path.join(BASE, "offline-bundle", "round4_results", "human_safety_20261018-041809.jsonl")
OUT = os.path.join(BASE, "paper", "review-audit")

LOG = []
def P(*a):
    s = " ".join(str(x) for x in a); LOG.append(s); print(s)

def load_new():
    rows = []
    for p in sorted(glob.glob(os.path.join(NEW, "*.jsonl"))):
        for line in io.open(p, encoding="utf-8"):
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("status", "ok") == "ok" and r.get("success") is not None:
                rows.append(r)
    return rows

def load_hist():
    out = []
    for line in io.open(HIST, encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("status", "ok") == "ok" and r.get("success") is not None:
            out.append(r)
    return out

def cell(r):
    return (r["task"], r["level"], r.get("init_state_index", r.get("trial")))

def cm(rows, cond):
    d = defaultdict(list)
    for r in rows:
        if r["condition"] == cond and r.get("paraphrase_group") == "soda_to_hand_plate":
            d[cell(r)].append(1.0 if r["success"] is True else 0.0)
    return {k: float(np.mean(v)) for k, v in d.items()}

def kn(rows, cond):
    s = [r for r in rows if r["condition"] == cond and r.get("paraphrase_group") == "soda_to_hand_plate"]
    return sum(1 for r in s if r["success"] is True), len(s)

def sf(a, b):
    d = np.asarray(a) - np.asarray(b); n = len(d); obs = d.mean()
    cnt = sum(1 for sg in itertools.product((1, -1), repeat=n)
              if abs(np.dot(sg, d) / n) >= abs(obs) - 1e-12)
    return obs, cnt / 2 ** n

def boot(a, b, B=20000, seed=7):
    rng = np.random.default_rng(seed); d = np.asarray(a) - np.asarray(b); n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])

def contrast(rows_a, rows_b, la, lb):
    ma = cm(rows_a, la)
    mb = cm(rows_b, lb)
    ks = sorted(set(ma) & set(mb))
    a = [ma[k] for k in ks]; b = [mb[k] for k in ks]
    rd, p = sf(a, b); lo, hi = boot(a, b)
    d = np.array(a) - np.array(b)
    return rd, p, lo, hi, int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum()), len(ks)

new = load_new(); hist = load_hist()
P(f"new rows={len(new)}  hist 4b2 rows={len(hist)}")

P("\n== 6a: soda, new batch only (states 0-4, k=2)")
for c in ["en", "zh", "cs_en_noun_zh"]:
    k, n = kn(new, c); P(f"  {c:16s} {k:2d}/{n:2d} = {100*k/n:5.1f}%")
for la, lb in [("en", "cs_en_noun_zh"), ("zh", "cs_en_noun_zh"), ("en", "zh")]:
    rd, p, lo, hi, w, t, l, n = contrast(new, new, la, lb)
    P(f"  {la:16s} vs {lb:16s} RD={100*rd:+6.1f} [{100*lo:+.1f},{100*hi:+.1f}] p={p:.5f} W/T/L={w}/{t}/{l} cells={n}")

P("\n== 6b: soda Spanish, new batch only")
for c in ["en", "es", "cs_en_noun_es", "cs_es_noun_en"]:
    k, n = kn(new, c); P(f"  {c:16s} {k:2d}/{n:2d} = {100*k/n:5.1f}%")
for la, lb in [("en", "cs_en_noun_es"), ("es", "cs_en_noun_es"),
               ("cs_es_noun_en", "cs_en_noun_es"), ("en", "es")]:
    rd, p, lo, hi, w, t, l, n = contrast(new, new, la, lb)
    P(f"  {la:16s} vs {lb:16s} RD={100*rd:+6.1f} [{100*lo:+.1f},{100*hi:+.1f}] p={p:.5f} W/T/L={w}/{t}/{l} cells={n}")

P("\n== 6b: pooled (new + 4b2), cell means over 4 runs")
pool = new + hist
for c in ["en", "es", "cs_en_noun_es", "cs_es_noun_en"]:
    k, n = kn(pool, c); P(f"  {c:16s} {k:2d}/{n:2d} = {100*k/n:5.1f}%")
for la, lb in [("en", "cs_en_noun_es"), ("es", "cs_en_noun_es"),
               ("cs_es_noun_en", "cs_en_noun_es"), ("en", "es")]:
    rd, p, lo, hi, w, t, l, n = contrast(pool, pool, la, lb)
    P(f"  {la:16s} vs {lb:16s} RD={100*rd:+6.1f} [{100*lo:+.1f},{100*hi:+.1f}] p={p:.5f} W/T/L={w}/{t}/{l} cells={n}")

P("\n== 6b: per batch, same contrast")
for tag, rows in [("new", new), ("4b2", hist)]:
    for la, lb in [("en", "cs_en_noun_es"), ("cs_es_noun_en", "cs_en_noun_es")]:
        rd, p, lo, hi, w, t, l, n = contrast(rows, rows, la, lb)
        P(f"  [{tag}] {la:16s} vs {lb:16s} RD={100*rd:+6.1f} p={p:.5f} W/T/L={w}/{t}/{l}")

io.open(os.path.join(OUT, "round6_for_paper.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
print("\nwritten:", os.path.join(OUT, "round6_for_paper.txt"))
