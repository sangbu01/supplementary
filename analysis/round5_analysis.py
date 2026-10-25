# -*- coding: utf-8 -*-
"""Round 4 analysis: E1 (4a), E1b (4e), E2 (4b), E5 (4c), E7 (4d), E4 distances.

Output: paper/review-audit/round5_out.txt
        paper/review-audit/round5_cells.csv
"""
import io, os, json, glob, itertools, math
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import fisher_exact, binomtest

BASE = r"D:/桑甲存/读博-中传/202610-具身智能的安全评估与保障-IEEE论文"
R4 = os.path.join(BASE, "offline-bundle", "round4_results")
T2 = os.path.join(BASE, "offline-bundle", "tier2_results")
OUT = os.path.join(BASE, "paper", "review-audit")

F_4A = "human_safety_20261018-041745.jsonl"
F_4E = "human_safety_20261018-041755.jsonl"
F_4B1 = "human_safety_20261018-041802.jsonl"
F_4B2 = "human_safety_20261018-041809.jsonl"
F_4C = "human_safety_20261018-075053.jsonl"
F_4D = "reasoning_safety_20261018-075121.jsonl"
F_OLD2D = "human_safety_20261016-115116.jsonl"
F_NULL = "human_safety_20261016-065137.jsonl"

LOG = []
def P(*a):
    s = " ".join(str(x) for x in a); LOG.append(s); print(s)

def load(path, names):
    out = []
    for n in names:
        p = os.path.join(path, n)
        if not os.path.exists(p):
            P(f"[警告] 缺文件 {p}"); continue
        for l in io.open(p, encoding="utf-8"):
            if l.strip(): out.append(json.loads(l))
    return out

def st(r):
    return r.get("init_state_index", r.get("trial"))

def cell(r):
    return (r["task"], r["level"], st(r))

def cell_d(r):
    # 4d 的 init_state_index 是 None，用 trial
    return (r["task"], r.get("trial"))

def sel(rows, g=None, c=None):
    s = rows
    if g: s = [r for r in s if r.get("paraphrase_group") == g]
    if c: s = [r for r in s if r["condition"] == c]
    return s

def kn(rows, g=None, c=None):
    s = sel(rows, g, c)
    return sum(1 for r in s if r["success"] is True), len(s)

def pct(k, n): return 100.0 * k / n if n else float("nan")

def cm(rows, g, c, keyf=cell):
    d = defaultdict(list)
    for r in sel(rows, g, c):
        d[keyf(r)].append(1.0 if r["success"] is True else 0.0)
    return {k: float(np.mean(v)) for k, v in d.items()}

def paired(m1, m2):
    ks = sorted(set(m1) & set(m2))
    return np.array([m1[k] for k in ks]), np.array([m2[k] for k in ks])

def signflip(a, b, B=200000, seed=20261018):
    d = a - b; n = len(d); obs = d.mean()
    if n == 0: return 0.0, 1.0
    if n <= 20:
        cnt = sum(1 for sg in itertools.product([1, -1], repeat=n)
                  if abs(np.dot(sg, d) / n) >= abs(obs) - 1e-12)
        return obs, cnt / 2 ** n
    rng = np.random.default_rng(seed)
    sg = rng.choice([-1, 1], size=(B, n))
    tots = np.abs(sg @ d) / n
    return obs, float(np.mean(tots >= abs(obs) - 1e-12))

def boot_rd(a, b, B=20000, seed=20261018):
    rng = np.random.default_rng(seed)
    d = a - b; n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    mus = d[idx].mean(axis=1)
    return np.percentile(mus, [2.5, 97.5])

def mcnemar(a, b):
    b01 = int(np.sum((a == 1) & (b == 0)))
    b10 = int(np.sum((a == 0) & (b == 1)))
    n = b01 + b10
    return b01, b10, (1.0 if n == 0 else binomtest(b01, n, 0.5).pvalue)

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = [0.0] * m; prev = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * ps[i]); prev = max(prev, v); adj[i] = prev
    return adj

def lpm(recs, ys, xs, cluster):
    import pandas as pd, statsmodels.formula.api as smf
    d = pd.DataFrame(recs)
    f = f"{ys} ~ " + " + ".join(xs)
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    return m

P("=" * 100)
P("ROUND 4 ANALYSIS — E1/E1b/E2/E5/E7 + E4 distances")
P("=" * 100)

# ------------------------------------------------------------------ inventory
P("\n" + "=" * 100); P("A. INVENTORY"); P("=" * 100)
for fn in [F_4A, F_4E, F_4B1, F_4B2, F_4C, F_4D]:
    p = os.path.join(R4, fn)
    rows = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
    P(f"  {fn:42s} {len(rows):4d}  " + str(dict(Counter(r.get("status") for r in rows))))

r4a = load(R4, [F_4A]); r4e = load(R4, [F_4E])
r4b1 = load(R4, [F_4B1]); r4b2 = load(R4, [F_4B2])
r4c = load(R4, [F_4C]); r4d = load(R4, [F_4D])
old2d = load(T2, [F_OLD2D]); nullb = load(T2, [F_NULL])

# ------------------------------------------------------------------ 4a
P("\n" + "=" * 100); P("B. E1 / 4a — language effect on states 10-24 (new confirmation set)"); P("=" * 100)
LANGS = ["en", "en_para", "es", "zh", "sw", "hi", "ar"]
for gname, tag in [("akita_bowl_deliver", "akita"), ("soda_to_hand_plate", "soda")]:
    P(f"\n  {tag}:")
    P(f"  {'cond':10s}{'k/n':>10s}{'rate':>8s}{'cells':>7s}")
    for c in LANGS:
        k, n = kn(r4a, gname, c)
        m = cm(r4a, gname, c)
        P(f"  {c:10s}{k:>4d}/{n:<4d}{pct(k,n):7.1f}%{len(m):7d}")
    base = cm(r4a, gname, "en_para")
    rows = []
    for c in ["es", "zh", "sw", "hi", "ar"]:
        a, b = paired(base, cm(r4a, gname, c))
        rd, p = signflip(a, b); lo, hi = boot_rd(a, b)
        b01, b10, pm = mcnemar((a >= 0.5).astype(int), (b >= 0.5).astype(int))
        rows.append((c, rd, p, lo, hi, b01, b10, pm))
        P(f"    {c:4s} vs en_para: RD={100*rd:+6.1f}pp CI[{100*lo:+6.1f},{100*hi:+6.1f}] "
          f"signflip p={p:.5f}  McNemar {b01}/{b10} p={pm:.5f}")
    adj = holm([r[2] for r in rows])
    P("    Holm: " + ", ".join(f"{r[0]}={q:.5f}" for r, q in zip(rows, adj)))
    P("    per level:")
    for lv in [0, 1, 2]:
        line = f"      L{lv}: "
        for c in LANGS:
            sub = [r for r in sel(r4a, gname, c) if r["level"] == lv]
            k = sum(1 for r in sub if r["success"] is True)
            line += f"{c} {k}/{len(sub)}  "
        P(line)

# old vs new
P("\n  Old (states 0-4, 3 runs) vs new (states 10-24, 1 run):")
V1 = ["human_safety_20261015-073738.jsonl", "human_safety_20261015-084802.jsonl",
      "human_safety_20261015-091730.jsonl", "human_safety_20261015-104243.jsonl"]
rep2 = load(T2, ["human_safety_20261016-092340.jsonl"])
rep3 = load(T2, ["human_safety_20261016-115100.jsonl"])
old = load(BASE + "/offline-bundle/ml_results", V1) + rep2 + rep3
for gname, tag in [("akita_bowl_deliver", "akita"), ("soda_to_hand_plate", "soda")]:
    P(f"  {tag}:")
    for c in ["es", "zh", "sw", "hi", "ar"]:
        mo = cm(old, gname, "en_para"); mn = cm(r4a, gname, "en_para")
        co = cm(old, gname, c); cn = cm(r4a, gname, c)
        ao, bo = paired(mo, co); an, bn = paired(mn, cn)
        rdo, po = signflip(ao, bo); rdn, pn = signflip(an, bn)
        P(f"    {c:3s} old RD={100*rdo:+6.1f}pp p={po:.5f}   new RD={100*rdn:+6.1f}pp p={pn:.5f}")
    # combined cell-level: old 15 cells + new 45 cells
    for c in ["es", "zh", "sw", "hi", "ar"]:
        mo = cm(old, gname, "en_para"); mn = cm(r4a, gname, "en_para")
        co = cm(old, gname, c); cn = cm(r4a, gname, c)
        # combine by distinct cell keys
        allkeys = sorted(set(mo) | set(mn))
        a = np.array([mo.get(k, mn.get(k)) for k in allkeys])
        b = np.array([co.get(k, cn.get(k)) for k in allkeys])
        rd, p = signflip(a, b)
        P(f"    {c:3s} combined 60 cells: RD={100*rd:+6.1f}pp p={p:.5f}")

# ------------------------------------------------------------------ 4e
P("\n" + "=" * 100); P("C. E1b / 4e — frame vs head verb on states 10-14"); P("=" * 100)
P(f"  {'cond':14s}{'k/n':>10s}{'rate':>8s}")
for c in ["en", "zh", "cs_zh_rel_en", "hv_en_zh1", "hv_en_zh2", "hv_zh_en1", "hv_zh_en2"]:
    k, n = kn(r4e, "akita_bowl_deliver", c)
    P(f"  {c:14s}{k:>4d}/{n:<4d}{pct(k,n):7.1f}%")
for lab, ca, cb in [("en -> hv_en_zh1", "en", "hv_en_zh1"),
                    ("en -> hv_en_zh2", "en", "hv_en_zh2"),
                    ("cs_zh_rel_en -> hv_zh_en1", "cs_zh_rel_en", "hv_zh_en1"),
                    ("cs_zh_rel_en -> hv_zh_en2", "cs_zh_rel_en", "hv_zh_en2"),
                    ("en -> cs_zh_rel_en (frame)", "en", "cs_zh_rel_en")]:
    a, b = paired(cm(r4e, "akita_bowl_deliver", ca), cm(r4e, "akita_bowl_deliver", cb))
    rd, p = signflip(a, b); lo, hi = boot_rd(a, b)
    P(f"    {lab:32s} RD={100*rd:+6.1f}pp CI[{100*lo:+6.1f},{100*hi:+6.1f}] p={p:.5f}")
P("\n  Combined with old 2d (states 0-4 + 10-14, 30 cells, 60 eps/condition):")
comb = r4e + old2d
for lab, ca, cb in [("en -> hv_en_zh1", "en", "hv_en_zh1"),
                    ("cs_zh_rel_en -> hv_zh_en1", "cs_zh_rel_en", "hv_zh_en1"),
                    ("en -> cs_zh_rel_en", "en", "cs_zh_rel_en")]:
    a, b = paired(cm(comb, "akita_bowl_deliver", ca), cm(comb, "akita_bowl_deliver", cb))
    rd, p = signflip(a, b); lo, hi = boot_rd(a, b)
    P(f"    {lab:32s} RD={100*rd:+6.1f}pp CI[{100*lo:+6.1f},{100*hi:+6.1f}] p={p:.5f}")
# balanced 2x2 on combined
recs = []
for c, fr, vb in [("en", "EN", "EN"), ("hv_en_zh1", "EN", "ZH"),
                  ("hv_zh_en1", "ZH", "EN"), ("cs_zh_rel_en", "ZH", "ZH")]:
    for r in sel(comb, "akita_bowl_deliver", c):
        recs.append({"y": 1 if r["success"] is True else 0, "frame": fr, "verb": vb,
                     "cell": f"{r['task']}|{r['level']}|{st(r)}"})
m = lpm(recs, "y", ["C(frame, Treatment('EN'))", "C(verb, Treatment('EN'))",
                    "C(frame, Treatment('EN')):C(verb, Treatment('EN'))"], "cell")
P("  Combined 2x2 LPM (cell-clustered):")
for nm in m.params.index:
    P(f"    {nm:60s} {m.params[nm]:+.4f} (SE {m.bse[nm]:.4f}, p={m.pvalues[nm]:.4f})")

# ------------------------------------------------------------------ 4b
P("\n" + "=" * 100); P("D. E2 / 4b — Spanish code-switching"); P("=" * 100)
P("\n  akita (4b1):")
for c in ["en", "es", "cs_es_rel_en", "cs_en_rel_es", "cs_es_noun_en", "cs_en_noun_es"]:
    k, n = kn(r4b1, "akita_bowl_deliver", c)
    P(f"    {c:18s}{k:>3d}/{n:<3d} {pct(k,n):6.1f}%")
for lab, ca, cb in [("es -> cs_es_rel_en (EN relation)", "es", "cs_es_rel_en"),
                    ("en -> cs_en_rel_es (ES relation)", "en", "cs_en_rel_es"),
                    ("es -> cs_es_noun_en (EN noun)", "es", "cs_es_noun_en"),
                    ("en -> cs_en_noun_es (ES noun)", "en", "cs_en_noun_es")]:
    a, b = paired(cm(r4b1, "akita_bowl_deliver", ca), cm(r4b1, "akita_bowl_deliver", cb))
    rd, p = signflip(a, b); lo, hi = boot_rd(a, b)
    P(f"    {lab:34s} RD={100*rd:+6.1f}pp CI[{100*lo:+6.1f},{100*hi:+6.1f}] p={p:.5f}")
# 2x2 frame x relation
recs = []
for c, fr, rl in [("es", "ES", "ES"), ("cs_es_rel_en", "ES", "EN"),
                  ("cs_en_rel_es", "EN", "ES"), ("en", "EN", "EN")]:
    for r in sel(r4b1, "akita_bowl_deliver", c):
        recs.append({"y": 1 if r["success"] is True else 0, "frame": fr, "rel": rl,
                     "cell": f"{r['task']}|{r['level']}|{st(r)}"})
m = lpm(recs, "y", ["C(frame, Treatment('EN'))", "C(rel, Treatment('EN'))",
                    "C(frame, Treatment('EN')):C(rel, Treatment('EN'))"], "cell")
P("    akita 2x2 frame x relation (LPM, cell-clustered):")
for nm in m.params.index:
    P(f"      {nm:60s} {m.params[nm]:+.4f} (SE {m.bse[nm]:.4f}, p={m.pvalues[nm]:.4f})")

P("\n  soda (4b2):")
for c in ["en", "es", "cs_es_noun_en", "cs_en_noun_es", "cs_es_dest_en", "cs_en_dest_es"]:
    k, n = kn(r4b2, "soda_to_hand_plate", c)
    P(f"    {c:18s}{k:>3d}/{n:<3d} {pct(k,n):6.1f}%")
for lab, ca, cb in [("es -> cs_es_noun_en", "es", "cs_es_noun_en"),
                    ("en -> cs_en_noun_es", "en", "cs_en_noun_es"),
                    ("es -> cs_es_dest_en", "es", "cs_es_dest_en"),
                    ("en -> cs_en_dest_es", "en", "cs_en_dest_es")]:
    a, b = paired(cm(r4b2, "soda_to_hand_plate", ca), cm(r4b2, "soda_to_hand_plate", cb))
    rd, p = signflip(a, b); lo, hi = boot_rd(a, b)
    P(f"    {lab:30s} RD={100*rd:+6.1f}pp CI[{100*lo:+6.1f},{100*hi:+6.1f}] p={p:.5f}")

# ------------------------------------------------------------------ 4c
P("\n" + "=" * 100); P("E. E5 / 4c — frame-only / content-only"); P("=" * 100)
for gname, tag in [("akita_bowl_deliver", "akita"), ("soda_to_hand_plate", "soda")]:
    P(f"\n  {tag}:")
    for c in ["frame_only_en", "frame_only_zh", "content_only_en", "content_only_zh"]:
        k, n = kn(r4c, gname, c)
        P(f"    {c:18s}{k:>3d}/{n:<3d} {pct(k,n):6.1f}%")
    # vs null on same cells (null has 1 ep/cell)
    for c in ["frame_only_en", "frame_only_zh", "content_only_en", "content_only_zh"]:
        a, b = paired(cm(r4c, gname, c), cm(nullb, gname, "null_empty"))
        rd, p = signflip(a, b)
        P(f"    {c:18s} vs null_empty: RD={100*rd:+6.1f}pp p={p:.5f}")
    # frame en vs zh; content en vs zh
    for lab, ca, cb in [("frame EN vs ZH", "frame_only_en", "frame_only_zh"),
                        ("content EN vs ZH", "content_only_en", "content_only_zh")]:
        a, b = paired(cm(r4c, gname, ca), cm(r4c, gname, cb))
        rd, p = signflip(a, b)
        P(f"    {lab:18s} RD={100*rd:+6.1f}pp p={p:.5f}")

# ------------------------------------------------------------------ 4d
P("\n" + "=" * 100); P("F. E7 / 4d — risk-category tasks (en vs zh)"); P("=" * 100)
P("  (success = the harmful action was executed; higher = less safe)")
for c in ["en", "zh"]:
    k, n = kn(r4d, None, c)
    P(f"  {c:4s} {k}/{n} = {pct(k,n):.1f}%")
# by task
P(f"  {'task':55s}{'en':>8s}{'zh':>8s}")
for task in sorted(set(r["task"] for r in r4d)):
    ke, ne = kn([r for r in r4d if r["task"] == task], None, "en")
    kz, nz = kn([r for r in r4d if r["task"] == task], None, "zh")
    P(f"  {task[:55]:55s}{ke:>4d}/{ne:<3d}{kz:>4d}/{nz:<3d}")
# paired by task+trial
for c in ["zh"]:
    a = cm(r4d, None, "en", keyf=cell_d); b = cm(r4d, None, c, keyf=cell_d)
    a2, b2 = paired(a, b)
    rd, p = signflip(a2, b2); b01, b10, pm = mcnemar((a2 >= 0.5).astype(int), (b2 >= 0.5).astype(int))
    P(f"  en vs {c}: RD={100*rd:+6.1f}pp signflip p={p:.5f} McNemar {b01}/{b10} p={pm:.5f}")
# by risk category
P("  by risk category:")
for cat in sorted(set(r.get("risk_category") for r in r4d)):
    sub = [r for r in r4d if r.get("risk_category") == cat]
    ke, ne = kn(sub, None, "en"); kz, nz = kn(sub, None, "zh")
    P(f"    {cat:16s} en {ke}/{ne} ({pct(ke,ne):4.0f}%)  zh {kz}/{nz} ({pct(kz,nz):4.0f}%)")

# ------------------------------------------------------------------ contact probe (distances)
P("\n" + "=" * 100); P("G. E4 — contact probe (distances; forces need rerun)"); P("=" * 100)
cp_path = os.path.join(R4, "contact_probe.jsonl")
if os.path.exists(cp_path):
    cp = [json.loads(l) for l in io.open(cp_path, encoding="utf-8") if l.strip()]
    P(f"  rows {len(cp)}; max_total_N nonzero: {sum(1 for r in cp if (r.get('max_total_N') or 0) > 0)}")
    P("  (all forces are 0 in this run: the pad filter used 'fingerpad', but Panda pads are")
    P("   named finger1_pad_collision / finger2_pad_collision. Distances/speeds are valid.)")
    for c in ["en", "en_para", "es", "zh", "sw", "hi", "ar"]:
        v = [r for r in cp if r["condition"] == c and r.get("min_hand_dist_m") is not None]
        if v:
            P(f"    {c:10s} n={len(v):3d} hand median={np.median([r['min_hand_dist_m'] for r in v]):.4f} "
              f"min={np.min([r['min_hand_dist_m'] for r in v]):.4f} "
              f"approach median={np.median([r['max_approach_speed_mps'] for r in v]):.3f}")
    # failure vs success
    for lab, sub in [("success", [r for r in cp if r.get("success") is True]),
                     ("failure", [r for r in cp if r.get("success") is False])]:
        v = [r for r in sub if r.get("min_hand_dist_m") is not None]
        if v:
            P(f"    {lab:8s} n={len(v):4d} hand median={np.median([r['min_hand_dist_m'] for r in v]):.4f} "
              f"approach median={np.median([r['max_approach_speed_mps'] for r in v]):.3f}")
else:
    P("  contact_probe.jsonl not found")

# ------------------------------------------------------------------ write
with io.open(os.path.join(OUT, "round5_out.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
print("\nwritten:", os.path.join(OUT, "round5_out.txt"))
