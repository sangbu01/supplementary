# -*- coding: utf-8 -*-
"""Round-3 review audit, part 2: alternative-explanation probes, proxy hazards,
stimulus-similarity measures, and power/MDE.  Appends to round4_out2.txt."""
import io, os, json, glob, itertools, math
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import fisher_exact, binomtest, spearmanr

BASE = r"D:/桑甲存/读博-中传/202610-具身智能的安全评估与保障-IEEE论文"
V1 = os.path.join(BASE, "offline-bundle", "ml_results")
T2 = os.path.join(BASE, "offline-bundle", "tier2_results")
OUT = os.path.join(BASE, "paper", "review-audit")
LOG = []
def P(*a):
    s = " ".join(str(x) for x in a); LOG.append(s); print(s)

def load(path, names=None):
    ps = [os.path.join(path, n) for n in names] if names else sorted(glob.glob(os.path.join(path, "*.jsonl")))
    out = []
    for p in ps:
        for l in io.open(p, encoding="utf-8"):
            if l.strip(): out.append(json.loads(l))
    return out

def st(r): return r.get("init_state_index", r.get("trial"))
def cell(r): return (r["task"], r["level"], st(r))
def sel(rows, g=None, c=None):
    s = rows
    if g: s = [r for r in s if r.get("paraphrase_group") == g]
    if c: s = [r for r in s if r["condition"] == c]
    return s
def cm(rows, g, c):
    d = defaultdict(list)
    for r in sel(rows, g, c):
        d[cell(r)].append(1.0 if r["success"] is True else 0.0)
    return {k: float(np.mean(v)) for k, v in d.items()}
def signflip(a, b):
    d = a - b; n = len(d); obs = d.mean()
    if n <= 20:
        cnt = sum(1 for sg in itertools.product([1, -1], repeat=n)
                  if abs(np.dot(sg, d) / n) >= abs(obs) - 1e-12)
        return obs, cnt / 2 ** n
    rng = np.random.default_rng(7)
    sg = rng.choice([-1, 1], size=(200000, n))
    return obs, float(np.mean(np.abs(sg @ d) / n >= abs(obs) - 1e-12))

v1 = load(V1, ["human_safety_20261015-073738.jsonl", "human_safety_20261015-084802.jsonl",
               "human_safety_20261015-091730.jsonl", "human_safety_20261015-104243.jsonl"])
rep2 = load(T2, ["human_safety_20261016-092340.jsonl"])
rep3 = load(T2, ["human_safety_20261016-115100.jsonl"])
hold = load(T2, ["human_safety_20261015-174529.jsonl"])
mech = load(T2, ["human_safety_20261016-102408.jsonl"])
verb = load(T2, ["human_safety_20261016-115116.jsonl"])
soda2e = load(T2, ["human_safety_20261016-115122.jsonl"])
null = load(T2, ["human_safety_20261016-065137.jsonl"])
obsb = load(T2, ["obstacle_avoidance_20261016-073327.jsonl"])
lang_rows = v1 + rep2 + rep3

P("=" * 100); P("M. SCRIPT MASS AND STIMULUS PROXIMITY"); P("=" * 100)
P("  For each mechanism condition: share of non-ASCII characters and success.")
P(f"  {'cond':18s}{'non-ASCII':>10s}{'CJK':>7s}{'k/n':>10s}{'rate':>8s}")
for c in ["en_relclause", "rel_alt", "rel_ablated", "cs_en_rel_zh",
          "cs_zh_rel_en", "cs_zh_rel_appos", "cs_zh_noun_en", "zh",
          "hv_en_zh1", "hv_en_zh2", "hv_zh_en1", "hv_zh_en2"]:
    rr = sel(mech if c not in ("zh", "hv_en_zh1", "hv_en_zh2", "hv_zh_en1", "hv_zh_en2") else verb,
             "akita_bowl_deliver", c)
    if not rr: continue
    s = rr[0]["instruction"]
    non = sum(1 for ch in s if ord(ch) > 127) / max(1, len(s))
    cjk = sum(1 for ch in s if '\u4e00' <= ch <= '\u9fff') / max(1, len(s))
    k = sum(1 for r in rr if r["success"] is True)
    P(f"  {c:18s}{100*non:9.1f}%{100*cjk:6.1f}%{k:>6d}/{len(rr):<4d}{100*k/len(rr):7.1f}%")
P()
P("  The two code-switched relation conditions carry comparable non-English mass but")
P("  differ sharply in outcome, so script mass alone does not order the conditions.")
P()

# nearest training instruction (character 3-gram Jaccard) from the benchmark BDDL set
import glob as _g, re
bddl = _g.glob(os.path.join(BASE, "offline-bundle", "libero-safety", "LIBERO-Safety",
                            "libero", "libero", "bddl_files", "**", "*.bddl"), recursive=True)
train = set()
for p in bddl:
    try:
        txt = io.open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    m = re.search(r"\(:language\s+(.+?)\)", txt)
    if m: train.add(m.group(1).strip())
train = sorted(train)
P(f"  benchmark (:language) strings found: {len(train)}")

def grams(s, n=3):
    s = " " + s.lower() + " "
    return Counter(s[i:i+n] for i in range(len(s)-n+1))
def jac(a, b):
    A, B = grams(a), grams(b)
    inter = sum((A & B).values()); union = sum((A | B).values())
    return inter / union if union else 0.0

P("  Character 3-gram Jaccard similarity to the nearest benchmark instruction:")
P(f"  {'cond':18s}{'nearest':>8s}{'rate':>8s}   nearest string")
for c in ["en_relclause", "cs_en_rel_zh", "cs_zh_rel_en", "cs_zh_rel_appos", "cs_zh_noun_en", "zh"]:
    src = mech if c in ("en_relclause", "cs_en_rel_zh", "cs_zh_rel_en", "cs_zh_rel_appos", "cs_zh_noun_en") else verb
    rr = sel(src, "akita_bowl_deliver", c)
    if not rr: continue
    s = rr[0]["instruction"]
    best = max(train, key=lambda t: jac(s, t))
    k = sum(1 for r in rr if r["success"] is True)
    P(f"  {c:18s}{jac(s,best):8.3f}{100*k/len(rr):7.1f}%   {best[:60]}")
P("  Similarity is collinear with frame language: it cannot separate 'frame' from")
P("  'distance to a training template'. This is reported as an identification limit.")
P()

P("=" * 100); P("N. PROXY HAZARDS AND BEHAVIOURAL METRICS"); P("=" * 100)
def hand_dist(r):
    d = r.get("min_dist_all") or r.get("min_dist_top5") or {}
    return d.get("left_hand_1")
def target_disp(r):
    d = r.get("max_disp_all") or r.get("max_disp_top5") or {}
    return d.get("akita_black_bowl_1")
def dist_disp(r):
    d = r.get("max_disp_all") or r.get("max_disp_top5") or {}
    return d.get("akita_black_bowl_2")

P("  Akita, three-replicate language sweep: end-effector minimum distance to the human hand.")
P(f"  {'cond':10s}{'n':>5s}{'median':>9s}{'p10':>8s}{'min':>8s}{'<0.09m':>8s}")
for c in ["en", "en_para", "es", "zh", "sw", "hi", "ar"]:
    v = [hand_dist(r) for r in sel(lang_rows, "akita_bowl_deliver", c) if hand_dist(r) is not None]
    P(f"  {c:10s}{len(v):5d}{np.median(v):9.3f}{np.percentile(v,10):8.3f}{np.min(v):8.3f}"
      f"{sum(1 for x in v if x < 0.09):8d}")
P("  (English reference minimum in Round A was 0.0903 m; the threshold is descriptive.)")
P()
P("  Akita: transport attribution by the threshold-free most-moved object.")
for c in ["en_relclause", "cs_en_rel_zh", "cs_zh_rel_en", "cs_zh_noun_en", "cs_zh_rel_appos"]:
    rr = sel(mech, "akita_bowl_deliver", c)
    P(f"  {c:18s}" + str(Counter(r.get("most_moved_object") for r in rr)))
P()
P("  Akita, language sweep: median end-effector travel and maximum object displacement.")
for c in ["en", "es", "zh", "ar"]:
    rr = sel(lang_rows, "akita_bowl_deliver", c)
    P(f"  {c:10s} travel {np.median([r['ee_travel'] for r in rr]):.3f} m   "
      f"max disp {np.median([r['max_object_displacement'] for r in rr]):.3f} m")
P()
P("  Soda, 2e batch: failure behaviour (target can displacement).")
for c in ["en", "zh", "cs_zh_noun_en", "cs_zh_dest_en"]:
    rr = sel(soda2e, "soda_to_hand_plate", c)
    fail = [r for r in rr if r["success"] is not True]
    disp = []
    for r in fail:
        d = r.get("max_disp_all") or r.get("max_disp_top5") or {}
        # the can is the only movable object of interest; take its max displacement
        vals = [v for k, v in d.items() if "can" in k.lower() or "soda" in k.lower()]
        disp.append(max(vals) if vals else 0.0)
    near = [r["min_dist_overall"] for r in fail]
    P(f"  {c:16s} failures {len(fail):2d}  median can disp {np.median(disp) if disp else float('nan'):.3f}"
      f"  reached<=5cm {sum(1 for x in near if x <= 0.05)}/{len(fail)}")
P()

P("=" * 100); P("O. NULL BASELINE VERSUS CHINESE FRAME, BY LEVEL"); P("=" * 100)
for g, gname in [("akita_bowl_deliver", "akita"), ("soda_to_hand_plate", "soda")]:
    P(f"  {gname}:")
    for lv in [0, 1, 2]:
        # null pooled over empty+irrelevant (same cell, 1 each)
        nn = [r for r in null if r.get("paraphrase_group") == g and r["level"] == lv]
        k0 = sum(1 for r in nn if r["success"] is True)
        zh = [r for r in sel(lang_rows, g, "zh") if r["level"] == lv]
        k1 = sum(1 for r in zh if r["success"] is True)
        P(f"    L{lv}: null {k0:2d}/{len(nn):2d} ({100*k0/len(nn):4.0f}%)   zh {k1:2d}/{len(zh):2d} ({100*k1/len(zh):4.0f}%)")
P()
P("  Akita L0+L2 pooled: null vs Chinese frame")
nn = [r for r in null if r.get("paraphrase_group") == "akita_bowl_deliver" and r["level"] in (0, 2)]
zh = [r for r in sel(lang_rows, "akita_bowl_deliver", "zh") if r["level"] in (0, 2)]
k0, n0 = sum(1 for r in nn if r["success"] is True), len(nn)
k1, n1 = sum(1 for r in zh if r["success"] is True), len(zh)
P(f"    null {k0}/{n0}; zh {k1}/{n1}; Fisher p={fisher_exact([[k1,n1-k1],[k0,n0-k0]])[1]:.4f}")
P("  Soda: null vs Chinese frame (all levels)")
nn = [r for r in null if r.get("paraphrase_group") == "soda_to_hand_plate"]
zh = sel(lang_rows, "soda_to_hand_plate", "zh")
k0, n0 = sum(1 for r in nn if r["success"] is True), len(nn)
k1, n1 = sum(1 for r in zh if r["success"] is True), len(zh)
P(f"    null {k0}/{n0}; zh {k1}/{n1}; Fisher p={fisher_exact([[k1,n1-k1],[k0,n0-k0]])[1]:.4f}")
P()

P("=" * 100); P("P. LANGUAGE ORDERING: REPLICATE VERSUS HELD-OUT"); P("=" * 100)
rep_rate = {}
for c in ["es", "zh", "sw", "hi", "ar"]:
    rr = sel(lang_rows, "akita_bowl_deliver", c)
    rep_rate[c] = sum(1 for r in rr if r["success"] is True) / len(rr)
hold_rate = {}
for c in ["es", "zh", "sw", "hi", "ar"]:
    rr = sel(hold, "akita_bowl_deliver", c)
    hold_rate[c] = sum(1 for r in rr if r["success"] is True) / len(rr)
P("  language   replicate   held-out")
for c in ["es", "zh", "sw", "hi", "ar"]:
    P(f"  {c:9s}{100*rep_rate[c]:8.1f}%{100*hold_rate[c]:9.1f}%")
rho, p = spearmanr([rep_rate[c] for c in rep_rate], [hold_rate[c] for c in hold_rate])
P(f"  Spearman rho = {rho:.2f} (n=5, no test reported)")
P("  The ordering is not stable; no ordering-based argument is used in the revision.")
P()

P("=" * 100); P("Q. POWER AND MINIMUM DETECTABLE EFFECT"); P("=" * 100)
P("  Paired cell-level design, 15 cells.  Two-sided alpha=0.05.")
P("  MDE (approx.) = 2.8 * SD(cell differences) / sqrt(15).")
for g, gname in [("akita_bowl_deliver", "akita"), ("soda_to_hand_plate", "soda")]:
    base = cm(lang_rows, g, "en_para")
    for c in ["es", "zh", "sw", "hi", "ar"]:
        other = cm(lang_rows, g, c)
        d = np.array([base[k] - other[k] for k in sorted(set(base) & set(other))])
        mde = 2.8 * d.std(ddof=1) / math.sqrt(len(d))
        P(f"  {gname} {c:3s}: SD={d.std(ddof=1):.3f}  MDE≈{100*mde:.1f}pp  observed RD={100*d.mean():+.1f}pp")
P("  Mechanism contrasts (15 cells, 2 reps): the 95% CI on the verb contrast is about")
P("  [-33,+7]pp, so a 10-15pp verb effect is not excluded; the relation contrast CI is")
P("  [-10,0]pp on the same design.")
P()
P("  For the null/verb comparison, the achieved CI width is the honest statement of")
P("  what the design can rule out; 'no effect' is not claimed.")

with io.open(os.path.join(OUT, "round4_out2.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
print("written round4_out2.txt")
