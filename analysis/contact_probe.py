# -*- coding: utf-8 -*-
"""E4：离线回放 + 接触力量测（不连策略服务器）。

对每条记录的 episode，用 jsonl 里的 action 序列在 MuJoCo 里重放，逐步记录：
  · 左右指垫与 left_hand_1 的最大法向力 / 切向力 / 合力（N）
  · 末端到人手的最小距离（m）与最大接近速度（m/s）
  · 接触发生的步数、持续步数、力冲量（N·s）

输出：
  contact_probe.jsonl   一行一个 episode（**每条跑完立即写入**，Ctrl-C 不丢）
  contact_probe.txt     汇总

用法（容器内）：
  python /work/contact_probe.py \
      --jsonl /work/results_tier2/human_safety_*.jsonl \
      --out /work/results_round4/contact_probe

注意：
  · 只用回放取轨迹上的力/距离，**不用回放重判成功**（已知谓词会翻转）。
  · MuJoCo 的接触力是仿真量，不能直接等同实测；与 ISO/TS 15066 的比较只做量级/排序。
  · 默认 64×64 渲染：接触力/位姿不受影响，比 256×256 快很多。
  · 中途断了直接重跑同一条命令即可：已写入的行会被跳过（--no-resume 可强制重跑）。
"""
import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multilingual_eval_v2 import (  # noqa: E402
    BehaviorTracker,
    LIBERO_DUMMY_ACTION,
    NUM_STEPS_WAIT,
    SEED,
    bddl_has_goal,
    resolve_bddl,
    _object_positions,
)

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None


def _raw(env):
    """robosuite 的 MjSim 包装在不同版本里字段不同，取底层 MjModel/MjData。"""
    m = getattr(env.sim.model, "_model", env.sim.model)
    d = getattr(env.sim.data, "_data", env.sim.data)
    return m, d


def _geom_names(m):
    out = {}
    for gid in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if nm:
            out[gid] = nm if isinstance(nm, str) else nm.decode()
    return out


def _contact_stats(env, hand_ids, pad_ids):
    """返回 (max_normal, max_tangential, max_total, n_contacts)。"""
    if mujoco is None:
        return 0.0, 0.0, 0.0, 0
    m, d = _raw(env)
    fmax_n = fmax_t = fmax = 0.0
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ((g1 in hand_ids and g2 in pad_ids) or (g2 in hand_ids and g1 in pad_ids)):
            continue
        f = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(m, d, i, f)
        fn = abs(float(f[0]))
        ft = float(np.hypot(f[1], f[2]))
        ftot = float(np.linalg.norm(f[:3]))
        fmax_n = max(fmax_n, fn)
        fmax_t = max(fmax_t, ft)
        fmax = max(fmax, ftot)
        n += 1
    return fmax_n, fmax_t, fmax, n


def _pad_geom_ids(env, names):
    """取左右指垫的 geom id。

    Panda 夹爪的指垫 geom 叫 finger1_pad_collision / finger2_pad_collision，
    名字里没有 "fingerpad"；直接用字符串过滤会漏掉，力全为 0。
    优先从 gripper.important_geoms 取，失败再按多种命名回退。
    """
    ids = set()
    try:
        gripper = env.robots[0].gripper
        for key in ("left_fingerpad", "right_fingerpad"):
            for gn in gripper.important_geoms.get(key, []):
                for gid, nm in names.items():
                    if nm == gn or nm.endswith(gn) or gn.endswith(nm):
                        ids.add(gid)
    except Exception:
        pass
    if not ids:
        for gid, nm in names.items():
            l = nm.lower()
            if ("fingerpad" in l or "pad_collision" in l or
                    "finger1_pad" in l or "finger2_pad" in l):
                ids.add(gid)
    return ids


def replay_one(env, init_state, actions, has_goal, control_dt):
    if init_state is not None:
        env.reset()
        obs = env.set_init_state(init_state)
    else:
        env.seed(SEED)
        obs = env.reset()

    m, _ = _raw(env)
    names = _geom_names(m)
    hand_ids = {g for g, n in names.items() if "left_hand" in n.lower()}
    pad_ids = _pad_geom_ids(env, names)
    if not pad_ids:
        print("[警告] 没找到指垫 geom；接触力会全为 0。请检查 geom 名称。",
              file=sys.stderr, flush=True)

    beh = BehaviorTracker(obs)
    t = 0
    while t < NUM_STEPS_WAIT:
        obs, _, _, info = env.step(LIBERO_DUMMY_ACTION)
        beh.update(obs, action=None, info=info, step_idx=t)
        t += 1

    eef_prev = np.asarray(obs["robot0_eef_pos"], dtype=float)
    rec = {
        "max_normal_N": 0.0, "max_tangential_N": 0.0, "max_total_N": 0.0,
        "force_step": None, "contact_steps": 0, "contact_first_step": None,
        "min_hand_dist_m": None, "max_approach_speed_mps": 0.0,
        "force_impulse_Ns": 0.0, "n_contacts_max": 0,
    }
    for a in actions:
        obs, _, done, info = env.step(list(a))
        beh.update(obs, action=a, info=info, step_idx=t)
        t += 1

        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        speed = float(np.linalg.norm(eef - eef_prev) / max(control_dt, 1e-6))
        eef_prev = eef
        pos = _object_positions(obs)
        hd = float(np.linalg.norm(eef - pos["left_hand_1"])) if "left_hand_1" in pos else None
        if hd is not None:
            rec["min_hand_dist_m"] = (hd if rec["min_hand_dist_m"] is None
                                      else min(rec["min_hand_dist_m"], hd))
        rec["max_approach_speed_mps"] = max(rec["max_approach_speed_mps"], speed)

        fn, ft, fmax, nc = _contact_stats(env, hand_ids, pad_ids)
        if nc > 0:
            rec["contact_steps"] += 1
            if rec["contact_first_step"] is None:
                rec["contact_first_step"] = t
            rec["n_contacts_max"] = max(rec["n_contacts_max"], nc)
            rec["force_impulse_Ns"] += fmax * control_dt
            if fmax > rec["max_total_N"]:
                rec["max_total_N"] = fmax
                rec["max_normal_N"] = fn
                rec["max_tangential_N"] = ft
                rec["force_step"] = t
        if done and has_goal:
            break

    return rec, beh.summary()


def _row_key(row):
    return (row.get("file"), row.get("task"), row.get("level"),
            row.get("init_state_index"), row.get("condition"), row.get("repeat"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="输出前缀，不带扩展名")
    ap.add_argument("--limit", type=int, default=0, help="每个文件最多跑几条（0=全部）")
    ap.add_argument("--condition", nargs="*", default=[])
    ap.add_argument("--task", nargs="*", default=[])
    ap.add_argument("--resolution", type=int, default=64,
                    help="渲染分辨率；接触力/位姿不受影响，64 比 256 快很多")
    ap.add_argument("--progress-every", type=int, default=10,
                    help="每多少条打印一次进度（0=只在结束时打印）")
    ap.add_argument("--no-resume", action="store_true",
                    help="忽略已存在的输出文件，从头跑")
    args = ap.parse_args()

    if mujoco is None:
        print("[错误] 没装 mujoco，无法取接触力", file=sys.stderr)
        return 2

    files = []
    for pat in args.jsonl:
        files.extend(sorted(glob.glob(pat)))
    files = [f for f in files if f.endswith(".jsonl")]
    if not files:
        print("[错误] 没找到 jsonl", file=sys.stderr)
        return 2

    # 先把任务摊平，才能给出准确总数与 ETA
    tasks = []
    for f in files:
        recs = [json.loads(l) for l in io.open(f, encoding="utf-8") if l.strip()]
        if args.condition:
            recs = [r for r in recs if r["condition"] in args.condition]
        if args.task:
            recs = [r for r in recs if any(t in r["task"] for t in args.task)]
        if args.limit:
            recs = recs[:args.limit]
        for rec in recs:
            if not rec.get("actions") or rec.get("status") != "ok":
                continue
            tasks.append((f, rec))

    out_jsonl = args.out + ".jsonl"
    out_txt = args.out + ".txt"
    done = set()
    if os.path.exists(out_jsonl) and not args.no_resume:
        for line in io.open(out_jsonl, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_row_key(json.loads(line)))
            except Exception:
                pass
        print(f"[resume] 已有 {len(done)} 条，跳过；要重跑请加 --no-resume", flush=True)

    print("=" * 72, flush=True)
    print(f" contact_probe: {len(files)} 个 jsonl, {len(tasks)} 条 episode", flush=True)
    print(f" 渲染分辨率 {args.resolution}x{args.resolution}；输出 {out_jsonl}", flush=True)
    print(f" 每 {args.progress_every} 条打印一次进度；Ctrl-C 不会丢已写入的行", flush=True)
    print("=" * 72, flush=True)

    from libero.libero import benchmark
    suite_cache, env_cache = {}, {}
    fh = io.open(out_jsonl, "a", encoding="utf-8")
    t0 = time.time()
    try:
        for i, (f, rec) in enumerate(tasks, 1):
            sname, name = rec["suite"], rec["task"]
            lv, lid = rec["level"], rec["level_id"]
            idx = rec.get("init_state_index")
            key = (os.path.basename(f), name, lv, idx, rec["condition"], rec.get("repeat"))
            if key in done:
                continue
            if sname not in suite_cache:
                suite_cache[sname] = benchmark.get_benchmark_dict()[sname]()
            suite = suite_cache[sname]
            task = next((suite.get_task(j) for j in range(suite.n_tasks)
                         if suite.get_task(j).name == name), None)
            if task is None:
                print(f"[跳过] 套件里找不到任务 {name[:50]}", file=sys.stderr, flush=True)
                continue
            ekey = (sname, name, lv, lid)
            if ekey not in env_cache:
                bddl = resolve_bddl(suite, task)
                if not bddl:
                    print(f"[跳过] bddl 不存在 {name[:50]}", file=sys.stderr, flush=True)
                    continue
                from libero.libero.envs import OffScreenRenderEnv
                env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                         camera_heights=args.resolution,
                                         camera_widths=args.resolution)
                env.seed(SEED)
                env.reset()
                env_cache[ekey] = (env, bddl_has_goal(bddl))
            env, has_goal = env_cache[ekey]
            try:
                inits = suite.get_task_init_states(task.level, task.level_id)
            except Exception:
                inits = None
            init_state = (inits[idx % len(inits)]
                          if (inits is not None and idx is not None) else None)
            control_dt = float(getattr(env, "control_timestep", 0.05))
            try:
                forces, beh = replay_one(env, init_state, rec["actions"], has_goal, control_dt)
            except Exception as e:
                print(f"[跳过] {name[:40]} {rec['condition']}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                continue
            row = {
                "file": os.path.basename(f), "suite": sname, "task": name,
                "level": lv, "init_state_index": idx, "condition": rec["condition"],
                "repeat": rec.get("repeat"), "success": rec.get("success"),
                "steps": rec.get("steps"), **forces,
                "min_dist_hand_json": (rec.get("min_dist_all") or {}).get("left_hand_1"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if args.progress_every and (i % args.progress_every == 0 or i == len(tasks)):
                el = time.time() - t0
                rate = i / max(el, 1e-9)
                eta = (len(tasks) - i) / max(rate, 1e-9)
                print(f"  [{i}/{len(tasks)}] {name[:36]:36s} {rec['condition']:18s} "
                      f"elapsed {el/60:5.1f} min  ETA {eta/60:5.1f} min", flush=True)
    finally:
        fh.close()

    # 汇总从这个文件读，包含之前 resume 的行
    logs = []
    if os.path.exists(out_jsonl):
        for line in io.open(out_jsonl, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except Exception:
                    pass

    lines = ["Contact probe summary", f"episodes: {len(logs)}"]
    if logs:
        import statistics as st
        for key in ["max_total_N", "max_normal_N", "max_tangential_N",
                    "min_hand_dist_m", "max_approach_speed_mps", "force_impulse_Ns"]:
            vals = [r[key] for r in logs if r.get(key) is not None]
            if not vals:
                continue
            lines.append(f"  {key:24s} n={len(vals):4d} median={st.median(vals):9.4f} "
                         f"p90={np.percentile(vals,90):9.4f} max={max(vals):9.4f}")
        by = {}
        for r in logs:
            by.setdefault(r["condition"], []).append(r)
        lines.append("\nby condition (max_total_N median / min_hand_dist median):")
        for c in sorted(by):
            v = by[c]
            mt = [r["max_total_N"] for r in v]
            hd = [r["min_hand_dist_m"] for r in v if r.get("min_hand_dist_m") is not None]
            lines.append(f"  {c:18s} n={len(v):3d} force_med={st.median(mt):8.3f} "
                         f"hand_med={st.median(hd) if hd else float('nan'):8.4f}")
    with io.open(out_txt, "w", encoding="utf-8") as fh2:
        fh2.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwritten:", out_jsonl, out_txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
