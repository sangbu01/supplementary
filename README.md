# Task- and Language-Pair-Scoped Frame Effects in π0.5

Data, instruction sets, episode logs and analysis code for the paper

> **Task- and Language-Pair-Scoped Frame Effects in π0.5: A Reproducible Two-Bowl Delivery Failure and a Safety Predicate That Cannot Fire**
> Jiacun Sang and Wen Liu
> IEEE AIHCIR 2026, Session 10 — Safety Evaluation and Assurance for Embodied Intelligence

---

## What this repository contains

The research objects behind every number in the paper:

- the **instruction sets** used in the six languages, with the native-speaker review outcomes;
- the **per-cell ledger** of all 5,717 analysed episodes from Rounds A–C;
- the **raw logs of the 180-episode Round 6 supplementary batch**;
- the **analysis scripts** that regenerate the tables, the figures and the contact-predicate audit from those files.

The paper studies one policy (π0.5 fine-tuned on LIBERO-Safety) under instruction-only manipulations, holding scene, initial states, seeds and weights fixed. The benchmark, its assets and the checkpoint are **not** redistributed here; see [What is not included](#what-is-not-included).

## Key results at a glance

| Experiment | Result |
|---|---|
| Two-bowl delivery, five non-English languages | English and paraphrase 44/45; non-English 19–24/45; RD 44–56 points; discovery Holm 0.0098, confirmation Holm < 10⁻⁴ |
| Frame × content code-switch | English frame with non-English content 89/90; Chinese frame with English content 63/90 |
| Soda, Chinese | frame −33.3 points (p = 0.008); English frame + Chinese noun −10.0 (n.s.) |
| Soda, Spanish | noun −26.7 points pooled over two batches (p = 0.0039); frame n.s. |
| Contact predicate | `CheckRobotContact` compares geom names with geom indices; it cannot return true. 0/2,850 constraint-logged episodes fired. |

Full tables, intervals and test procedures are in the paper. The unit of analysis is the **cell** = (task instance, level, initial state); a task group has 3 task instances × 5 initial states = 15 cells.

## Repository layout

```
.
├── README.md
├── instructions_163.csv              # 163 reviewed strings (first condition set)
├── instructions_v5.json              # final instruction set for Rounds B/C
├── v5_review_sheet.csv               # 32 Round-C strings and their review outcome
├── v5_复核结果与处置.md               # Round-C review decisions (Chinese)
├── per_cell_ledger.csv               # 5,717 episodes, one row each (Rounds A–C)
├── round5_out.txt                    # Round-C analysis output
├── table_S1_frame_only_content.csv   # Table S1 in machine-readable form
├── second_model_gate_failure.md      # π0 second-model attempt and its gate failure
├── review_sheets/                    # six anonymised native-speaker workbooks
├── round6_results/                   # 180-episode supplementary batch + analysis
└── analysis/
    ├── round4_analysis.py
    ├── round4_analysis2.py
    ├── round5_analysis.py
    ├── recompute_table2_fixed_runs.py
    ├── analyze_round6_for_paper.py
    ├── make_figures.py
    ├── contact_probe.py              # needs MuJoCo + LIBERO-Safety
    └── geom_diag.py                  # needs MuJoCo + LIBERO-Safety
```

## Data

### Instruction sets

`instructions_163.csv` is the first reviewed set: one row per string with language, task, condition, the exact string, its English source, and the reviewer's judgement, verdict and comment. `instructions_v5.json` is the final set used for Rounds B and C: the same strings plus 32 Round-C additions (Spanish code-switching and frame-only/content-only conditions). Every string that exists in both files is byte-identical.

`v5_review_sheet.csv` lists the 32 added strings with the reviewer columns and the final, corrected strings. The review outcome was 29 accepted, 3 accepted with a minimal correction, 0 rejected; the corrections were applied before the batch.

### Per-cell ledger

`per_cell_ledger.csv` has one row per analysed episode from Rounds A–C (5,717 rows):

| Column | Meaning |
|---|---|
| `batch`, `suite`, `group`, `task`, `level`, `state` | episode identity |
| `condition`, `repeat` | instruction condition and run index |
| `success` | the benchmark's terminal predicate |
| `ee_travel`, `min_dist_hand`, `max_disp_all` | behavioural metrics |
| `cost_events` | number of steps on which a BDDL constraint was violated |

Round 6 is **not** in the ledger; its raw logs are in `round6_results/`.

### Round 6 supplementary batch

180 episodes on the soda group, initial states 0–4, two runs per cell, collected on 25 October 2026. Three JSONL files plus two analysis outputs:

- `6a` — the missing mirror cell, English frame + Chinese noun (`cs_en_noun_zh`): `en` 26/30, mirror 23/30, `zh` 14/30; `en` − mirror 
  +10.0 points (95% CI [−6.7, +26.7], p = 0.50), `zh` − mirror −30.0 (p = 0.063), `en` − `zh` +40.0 (p = 0.026).
- `6b` — repeat of the Spanish soda cells: `cs_en_noun_es` 19/30 and `cs_es_noun_en` 27/30, exactly reproducing the first batch. Pooled over both batches (four runs per cell), `en` 54/60 vs `cs_en_noun_es` 38/60 (+26.7 points, p = 0.0039).

### What is not included

The following remain under their original licences and are not redistributed:

- the LIBERO-Safety benchmark, its BDDL files and the 10 GB asset library (upstream release);
- the π0.5 `pi05_libero_safety` checkpoint and the π0 weights;
- the Docker image used to serve the policy;
- the action-level logs of Rounds A–C (about 64 MB). The per-cell ledger is the analysis-ready form and reproduces every table except the contact replay; the action logs are available from the corresponding author.

## Reproducing the numbers

### Requirements

Python 3.8+ with `numpy`, `scipy`, `pandas`, `statsmodels` and `matplotlib` for the table and figure scripts. `contact_probe.py` and `geom_diag.py` additionally need MuJoCo, robosuite 1.4 and LIBERO-Safety.

### Commands

```bash
# Table II from an explicitly fixed run set (Round A + two Round B batches)
python analysis/recompute_table2_fixed_runs.py

# Round-C analyses: confirmation set, verb 2x2, Spanish, frame/content-only,
# risk tasks, contact-probe distances
python analysis/round5_analysis.py

# Round 6: mirror cell, Spanish repeat, pooled two-batch contrasts
python analysis/analyze_round6_for_paper.py

# First-round descriptive statistics and the script-mass / similarity analyses
python analysis/round4_analysis.py
python analysis/round4_analysis2.py

# Figures 1 and 2
python analysis/make_figures.py
```

Each script defines a `BASE` path at the top; edit it to point at your checkout. Outputs are written next to the scripts or into `review-audit/`, as noted in each file's docstring.

### Statistics

Comparisons use an exact paired sign-flip permutation test over cell means (all 2^15 sign assignments at 15 cells), exact McNemar for single-run comparisons, and cell-cluster bootstrap intervals. No p-value in the paper is computed on pooled episodes. Holm correction is applied within each stated family.

## Native-speaker review

The first review had six reviewers and judged 156 of the 163 strings; the seven English originals were exempt because they are the benchmark's own instructions. One rater judged each language, so inter-rater reliability cannot be estimated. Code-switched strings were judged for grammaticality and unambiguous parseability, not naturalness. The reviewers are anonymous and did not consent to being named; the completed workbooks contain no names or dates.

The fourth round added 32 strings (16 Spanish, 8 Chinese, 8 English), reviewed by three independent native speakers before the batch: 29 accepted, 3 accepted with a correction, 0 rejected.

## Second-model attempt

We ran the same pre-specified 420-episode protocol on π0. Two LoRA variants (initialised from the π0 base and from the standard-LIBERO π0 checkpoint) and a third configuration using the base checkpoint's normalisation statistics all failed the English competence gate: 0/60 English successes. The unmodified π0 checkpoint scores 7/10 on standard LIBERO spatial, while both fine-tuned variants score 0/10. Per the pre-specified rule, the second-model language contrast is not interpreted and the paper remains a single-policy case study.
Details are in `second_model_gate_failure.md`.

## Licence and citation

No licence has been finalised yet. The authors intend to release the **code
under the MIT Licence** and the **data and instruction strings under CC BY
4.0**; the corresponding `LICENSE` files will be added before public release.
The LIBERO-Safety benchmark, its assets and the π0.5 checkpoint remain under
their original licences.

```bibtex
@inproceedings{sang2026frameeffects,
  title     = {Task- and Language-Pair-Scoped Frame Effects in $\pi_{0.5}$:
               A Reproducible Two-Bowl Delivery Failure and a Safety
               Predicate That Cannot Fire},
  author    = {Sang, Jiacun and Liu, Wen},
  booktitle = {Proc. IEEE International Conference on Artificial Intelligence,
               Human-Computer Interaction and Robotics (AIHCIR), Session 10},
  year      = {2026},
  note      = {To appear}
}
```

---

## 中文说明

本目录是论文《Task- and Language-Pair-Scoped Frame Effects in π0.5》的数据与代码：

- `instructions_163.csv` / `instructions_v5.json`：六语指令集与复核结果；
- `per_cell_ledger.csv`：Rounds A–C 共 5,717 个回合的逐单元台账；
- `round6_results/`：2026-10-25 补跑的 180 回合（汽水镜像格 + 西语重复批）；
- `analysis/`：重算表格、图 1/2 与接触谓词审计的脚本。

基准本体（LIBERO-Safety、10 GB assets）与 π0.5 权重不在本目录，按上游许可获取。
Round A–C 的逐步动作日志约 64 MB，台账已足够复现除接触回放外的全部数字；
需要动作日志可向通讯作者索取。
