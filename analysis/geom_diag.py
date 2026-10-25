# -*- coding: utf-8 -*-
"""诊断：为什么指垫-人手接触力全为 0。

在一个 episode 上重建环境、逐步执行记录的动作，打印：
  · 所有含 hand / finger / pad / gripper 的 geom 名称；
  · hand / pad / finger 的 contype / conaffinity / group；
  · gripper.important_geoms；
  · 每一步的接触对、hand / pad / hand+pad / finger-collision 计数；
  · 全 episode 接触对 Top 12。

用法（容器内）：
  python /work/geom_diag.py \
      --jsonl /work/results_tier2/human_safety_*.jsonl \
      --condition cs_zh_rel_en --max-steps 120
"""
import argparse, glob, io, json, os, sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multilingual_eval_v2 import (  # noqa: E402
    LIBERO_DUMMY_ACTION, NUM_STEPS_WAIT, SEED, resolve_bddl,
)

try:
    import mujoco
except Exception:
    mujoco = None


def _raw(env):
    m = getattr(env.sim.model, "_model", env.sim.model)
    d = getattr(env.sim.data, "_data", env.sim.data)
    return m, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--condition", default="cs_zh_rel_en")
    ap.add_argument("--task", default="")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--resolution", type=int, default=64)
    args = ap.parse_args()

    files = sorted(glob.glob(args.jsonl))
    rec = None
    for f in files:
        for l in io.open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("condition") != args.condition:
                continue
            if args.task and args.task not in r["task"]:
                continue
            if not r.get("actions") or r.get("status") != "ok":
                continue
            rec = r
            break
        if rec:
            break
    if not rec:
        print("[错误] 没找到符合条件的 episode", file=sys.stderr)
        return 2

    print("episode:", rec["task"], "L", rec["level"], "init", rec.get("init_state_index"),
          "cond", rec["condition"], "success", rec["success"], "steps", rec["steps"])

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[rec["suite"]]()
    task = next((suite.get_task(i) for i in range(suite.n_tasks)
                 if suite.get_task(i).name == rec["task"]), None)
    bddl = resolve_bddl(suite, task)
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                             camera_heights=args.resolution,
                             camera_widths=args.resolution)
    env.seed(SEED); env.reset()
    inits = suite.get_task_init_states(task.level, task.level_id)
    idx = rec.get("init_state_index")
    init = inits[idx % len(inits)] if (inits is not None and idx is not None) else None
    if init is not None:
        env.set_init_state(init)

    m, d = _raw(env)
    names = {}
    for gid in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if nm:
            names[gid] = nm if isinstance(nm, str) else nm.decode()

    print("\n--- geom names containing hand/finger/pad/gripper ---")
    for gid, nm in sorted(names.items()):
        l = nm.lower()
        if any(k in l for k in ("hand", "finger", "pad", "gripper")):
            print(f"  {gid:4d}  {nm}")

    print("\n--- gripper important_geoms ---")
    try:
        g = env.robots[0].gripper
        for k, v in g.important_geoms.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print("  gripper 读取失败:", type(e).__name__, e)

    hand_ids = {g for g, n in names.items() if "left_hand" in n.lower()}
    pad_ids = set()
    try:
        g = env.robots[0].gripper
        for key in ("left_fingerpad", "right_fingerpad"):
            for gn in g.important_geoms.get(key, []):
                for gid, nm in names.items():
                    if nm == gn or nm.endswith(gn) or gn.endswith(nm):
                        pad_ids.add(gid)
    except Exception:
        pass
    if not pad_ids:
        pad_ids = {g for g, n in names.items()
                   if any(k in n.lower() for k in ("fingerpad", "pad_collision",
                                                   "finger1_pad", "finger2_pad"))}
    finger_ids = {g for g, n in names.items()
                  if "finger" in n.lower() and "collision" in n.lower() and "pad" not in n.lower()}
    gripper_ids = {g for g, n in names.items() if n.lower().startswith("gripper0_")}

    print(f"\nhand_ids={sorted(hand_ids)}  pad_ids={sorted(pad_ids)}  finger_ids={sorted(finger_ids)}")
    print("hand geoms:", [names[i] for i in sorted(hand_ids)])
    print("pad  geoms:", [names[i] for i in sorted(pad_ids)])
    print("finger-collision geoms:", [names[i] for i in sorted(finger_ids)])

    print("\n--- collision properties ---")
    for gid in sorted(hand_ids | pad_ids | finger_ids | gripper_ids):
        print(f"  {gid:4d} {names[gid]:36s} contype={int(m.geom_contype[gid])} "
              f"conaffinity={int(m.geom_conaffinity[gid])} group={int(m.geom_group[gid])}")

    for _ in range(NUM_STEPS_WAIT):
        env.step(LIBERO_DUMMY_ACTION)

    print("\n--- stepping actions ---")
    n_hand = n_pad = n_both = n_any = n_finger = n_grip = 0
    pair_count = Counter()
    grip_pairs = Counter()
    for t, a in enumerate(rec["actions"][:args.max_steps], 1):
        env.step(list(a))
        m, d = _raw(env)
        if d.ncon == 0:
            continue
        n_any += d.ncon
        pairs = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            h = (g1 in hand_ids) or (g2 in hand_ids)
            p = (g1 in pad_ids) or (g2 in pad_ids)
            f = (g1 in finger_ids) or (g2 in finger_ids)
            gr = (g1 in gripper_ids) or (g2 in gripper_ids)
            n_hand += int(h); n_pad += int(p); n_both += int(h and p); n_finger += int(f)
            n_grip += int(gr)
            pair_count[(names.get(g1, g1), names.get(g2, g2))] += 1
            if gr:
                grip_pairs[(names.get(g1, g1), names.get(g2, g2))] += 1
            if h or p:
                pairs.append((names.get(g1, g1), names.get(g2, g2), h, p))
        if pairs and t <= 30:
            print(f"  step {t:3d} ncon={d.ncon}: " +
                  "; ".join(f"{a}--{b}{' [hand]' if h else ''}{' [pad]' if p else ''}"
                            for a, b, h, p in pairs[:6]))
    print(f"\ncontacts: total={n_any} hand={n_hand} pad={n_pad} hand+pad={n_both} "
          f"finger-collision={n_finger}")
    print("\n--- top 12 contact pairs over the episode ---")
    for (a, b), n in pair_count.most_common(12):
        print(f"  {n:6d}  {a}  <->  {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
