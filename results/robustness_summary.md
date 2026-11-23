# Robustness results

Source: `src/robustness.py` → `results/robustness.{txt,json}`. Honest runs, 11 pipelines, task-held-out unless stated. `LinearSVC(random_state=0)`; no paid calls.

| View | 11-way acc | One-vs-rest AUROC, range across pipelines [lowest, 95% task-cluster CI] | Leave-one-environment-out acc: median (min–max) |
|---|---:|---|---|
| J1 tool calls | 0.909 | 0.939–1.000 [Kimi 0.939, 0.927–0.950] | 0.857 (0.772 clinical_trial – 0.914 auto_workflow) |
| J4 600-char skeleton | 0.768 | 0.833–0.997 [Kimi 0.833, 0.804–0.859] | 0.755 (0.598 shared_dotfiles – 0.782 web_scraping) |
| H2 tool-call metadata | 0.819 | 0.900–1.000 [Kimi 0.900, 0.881–0.918] | 0.804 (0.682 media_processing – 0.859 file_indexing) |
| H3 tool names + approximate verbs | 0.791 | 0.841–0.996 [Kimi 0.841, 0.814–0.867] | 0.756 (0.644 shared_dotfiles – 0.782 slack_clone) |

**Honest→attack transfer.** Train on the 2,056 honest runs of tasks outside the attack set; test on the 1,012 attack runs. Calls here are *rebuilt from structured actions*, so planning calls and key order can be edited. The canonical formatter-text J1 transfer is 0.819 (`results/identification.txt`).

| Variant | Accuracy |
|---|---:|
| Rebuilt tool calls as-is | 0.827 |
| Planning-tool calls removed | 0.828 |
| Planning removed + argument keys sorted | 0.820 |
| H3 verbs as-is | 0.647 |
| H3 verbs, planning removed | 0.640 |

**Reading:**
1. Every pipeline is individually distinguishable: one-vs-rest AUROC ≥ 0.83 on every view. Kimi and GLM 5 are the hardest.
2. Attribution holds on held-out environments, with medians within about 5 points of task-held-out accuracy.
3. Removing planning calls or sorting argument keys changes transfer by at most 1 point.

**Caveats:**
- Leave-one-environment-out uses 10 environments, so the ranges are descriptive only.
- Every loop asserts that its environment and tasks are disjoint from training.
