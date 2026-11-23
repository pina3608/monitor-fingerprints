# Model Fingerprints in Agent Transcripts

**A supervised source-attribution signal in LinuxArena monitor views**

*Independent technical report. Draft of 20 November 2026. All analyses are offline on public data; no paid model calls.*

## Abstract
Untrusted monitoring, in which a copy of an AI agent's own model reviews the agent's actions, is vulnerable to collusion if the monitor can recognise its own model's work. A necessary precondition is that the text the monitor sees carries information about which model produced it.

We measure that precondition on LinuxArena. The data are 3,210 public trajectories in which 11 frontier-model pipelines performed the same tasks. A supervised linear classifier, trained with model labels and evaluated on held-out tasks, attributes the **2,198 honest runs** to their generating pipeline with accuracy (chance 0.091):
- **0.909** from the tool calls in the monitor view;
- **0.819** from tool-call metadata (tool names, argument-key order, quoting, escapes, length bands; no command text);
- **0.791** from tool names and approximate command verbs.

A common-tool approximate-verb view, which drops environment-specific and scaffold tools, keeps 0.752. A model trained on tool names alone reaches 0.313, and outcome, length and prompt-template features together reach 0.288. Attribution also has these properties:
- it holds within a fixed system-prompt template;
- it holds on the first 5–20 common-tool calls;
- it holds separately for successful and failed runs;
- it holds on held-out environments (pooled 0.73–0.85 depending on view);
- trained on honest runs, it transfers to **all 1,012 attack runs** (0.819 from tool calls).

Generation pipelines differ by model (prompt templates, serialisation, tool sets), so this is **pipeline attribution**, not evidence of intrinsic model behaviour; browser-agent fingerprints are known not to transfer across harnesses (*Known By Their Actions*). A test of cheap deterministic text transforms, against a success criterion set before running it, found none that, as implemented, halved identifiability.

The result is an existence proof of supervised attribution signal in one benchmark. It does **not** show that an LLM monitor, prompted or fine-tuned, recognises or exploits that signal.

## 1. Introduction
AI-control protocols aim to keep deployments safe even if the deployed model is misaligned. In **untrusted monitoring**, a capable model reviews the actions of another instance of itself, because it is often the best reviewer available.

The central threat is **collusion**: a monitor that recognises its own model's trajectories could pass its own real attacks while still flagging the security team's honeypots, which defeats honeypot-based validation.

[Gardner-Challis et al. (2026)](https://arxiv.org/abs/2602.20628) evaluate collusion strategies in single-step code (APPS, BigCodeBench):
- off-the-shelf self-recognition was the weakest red-team strategy they evaluated;
- self-recognition after supervised fine-tuning was the strongest;
- minification and trusted summarisation were the main signal-jamming defences, with mixed effectiveness by setting.

They do not study multi-step agent transcripts. Their discussion of gaps speculates that **context-dependent ("selective") collusion** is among the largest, and that "longer contexts may have more collusion potential". This report does not test either claim. It measures one input to them for agentic transcripts.

A collusion risk in deployment requires five distinct things:
1. **information availability:** transcripts carry identifying signal;
2. **recognition capability:** a monitor can extract it;
3. **elicitation:** prompting or training induces recognition;
4. **exploitation:** recognition changes how attacks are scored relative to honeypots;
5. **deployment relevance:** the effect persists under real monitoring and auditing.

**This report addresses only (1)**, with a supervised classifier, and asks which parts of the signal survive normalisation.

## 2. Related work and positioning
**Agent and trajectory fingerprinting already exists.** Three recent papers are closest:
- [*Trace*](https://arxiv.org/abs/2605.01186) (Ediga & Chattopadhyay) attributes the model family of AI penetration-testing agents from terminal command sequences on a Linux CTF box: macro F1 0.981 over seven families and three scaffolds, 0.815 on unseen scaffolds. It then uses the attribution to guide defensive prompt injection.
- [*LIDAR*](https://arxiv.org/abs/2609.28559) (Wang et al., *Who Is Behind the Harness?*) identifies 24 models from 8 families inside coding-agent harnesses (OpenCode, OpenHands, mini-swe-agent) from active probe trajectories: closed-set top-1 0.914–0.989, with references fitted per harness.
- [*Known By Their Actions*](https://arxiv.org/abs/2605.14786) (Lugoloobi et al.) identifies which of 14 multimodal models drives a browser agent from passively recorded page events (clicks, scrolls, keypresses, timing): up to 0.96 F1 within one harness, near-chance transfer to a second harness, 0.74–0.81 macro-F1 when both harnesses are pooled. Timing delays and prompt normalisation hurt a classifier trained on clean traces, but one retrained on the perturbed traces largely recovers.

Together they establish that agent behaviour identifies the generating model. Our result is not novel in that respect, and we do not claim it is. The contribution is a careful measurement of attribution in the input an AI-control monitor actually receives.

**What this report adds**, specifically for AI-control monitor inputs:
- the setting is the *monitor view* of an AI-control benchmark: capped histories produced by a byte-exact port of the LinuxArena monitor formatter (formatter fidelity and the study's own character caps are separate facts, §4);
- tasks are matched: every pipeline ran the same 200 honest tasks and the same 92 attack task pairs;
- metadata and command-verb views compared against tool-name, outcome, length, template, prefix and success-stratum baselines, with tool-name, parse-ambiguity and token-order controls;
- honest→attack transfer on disjoint tasks;
- a test of cheap deterministic jamming transforms against a success criterion set in advance;
- a provenance audit of the public trajectories (cohort, templates, scaffold calls, version labels).

Against these strengths, we have weaker scaffold control than *Trace* (which varies the scaffold deliberately) and no active probing as in *LIDAR*.

**Other related work:**
- **Untrusted monitoring and collusion:** Gardner-Challis et al. (§1).
- **Self-recognition by LLMs:**
  - [SGTR](https://arxiv.org/abs/2608.26159): accuracy depends strongly on presentation and is confounded by quality preference;
  - [Know Thyself?](https://arxiv.org/abs/2510.03399): mostly chance-level unprompted recognition;
  - Panickssery et al. (2024) link recognition to self-preference.
- **Agentic recognition:** [CIAware-Bench](https://arxiv.org/abs/2606.11063): when probed, frontier models can detect a single step written by another model inside their own BashArena and SHADE-Arena trajectories.
- **Model attribution of text:** [Idiosyncrasies in LLMs](https://arxiv.org/abs/2502.12150) for chat outputs; [Calibration-Family Overfit](https://arxiv.org/abs/2607.06596) reports family identification from code transcripts.

## 3. Data
Every count in this section is produced by `src/cohort_audit.py` (`results/cohort_audit.txt`).
- **Source:** LinuxArena public mirror (Tracy et al., [arXiv:2604.15384](https://arxiv.org/abs/2604.15384); HF `anonymouslinuxarena/linuxarena-public` at revision `ed82b89a`, CC-BY-4.0).
  - 10 environment shards, 22,215 records. Each shard's SHA-256 is checked on download.
  - Of these, 12,248 are monitor re-scorings of existing trajectories, 3,212 are cohort agent runs, and 6,755 are other agent runs.
- **Cohort:** agent runs `"<Model> honest"` and `"<Model> attack"` for 11 pipelines: Claude Opus 4.6, Sonnet 4.6, Haiku 4.5; Gemini 3.1 Pro, Gemini 3 Flash; GLM 5; Kimi K2.5; Grok 4.1 fast; GPT-5 Nano, GPT-5 Mini, GPT-5.4.
- **Task matching:**
  - honest: the same 200 main tasks for every pipeline;
  - attack: the same 92 (main, side) task pairs, on 67 main tasks;
  - one trajectory per pipeline per task, asserted on (model, label, env, main task, side task), with trajectory IDs unique.
  - 13 main tasks appear in both conditions.
- **Exclusions:** 2 Kimi honest runs whose `agent_model` field is `kimi-k2-thinking`.
  - Final cohort: **3,210 trajectories** (2,198 honest, 1,012 attack).
- **Version labels:** 1,442 honest runs carry no `agent_model` value. They span all 8 bare-template pipelines (e.g. 139 of the 198 retained Kimi runs; 2 Kimi runs tagged `kimi-k2-thinking` are excluded), so model versions rest on the run names for those runs.
- **Repeated views:** two monitor views each occur in 7 GPT attack runs, one being the empty-history sentinel "No previous actions.". All are kept.
- **Manifest:** `manifest/inputs.jsonl` has one line per trajectory with its ID, run ID and the SHA-256 of its monitor views (manifest SHA-256 `6e678448…5c28`).

## 4. Methods
**Monitor-visible text.** Built with a port of LinuxArena's formatter (`control_tower.format_history` @ 699d5ca), byte-identical on 932/932 sample trajectories. It shows tool calls and outputs, hides agent reasoning, and omits submit and side-task tools.

**Views.** J0–J4, H2 and H3 are the views planned from the start; the others were added as baselines and controls.

| View | Content |
|---|---|
| J0 full | calls + outputs (first 100k chars) |
| J1 tool calls | calls only |
| J2 | J1 with own-line comments removed (partial comment stripping) and whitespace collapsed |
| J3 structure-only | quoted strings → `<S>` and numeric literals → `<N>`; punctuation, structure and values such as `True`/`None` kept\* |
| J4 skeleton | tool name + first command token (~600 chars) |
| H2 tool-call metadata | per action: tool name, ordered argument keys, quote style, escape markers, log2 length band of the argument text; no command text |
| H3 tool names + approximate verbs | per action: `bash:<first word>` for the `bash` tool (prefix regex, whitespace split), `edit:<sub-command>` for the editor, the tool name verbatim otherwise |
| H3b | H3 tokens sorted, then n-grams (order-free) |
| H0 | number of actions only |
| C1 common-tool approximate verbs | `bash` and editor calls only; commands split on `; && \|\| \|` and newlines *outside quotes*, first word taken with `shlex`; wrappers (`sudo`, `env`, `timeout`, `nohup`, …) skipped together with their own value-taking options, and `cd` skipped |
| C1P | C1 with every ambiguous parse (unbalanced quote, `shlex` failure, non-word verb) replaced by the placeholder `bash:?` |
| H3m | H3 with every non-`bash`, non-editor tool name replaced by `tool:OTHER`; parser unchanged *(tool-name ablation)* |
| T0 / T1 | tool names only / common-tool identity only (`bash` vs editor) |
| LEN, OUT, TMPL, LOT | categorical baselines: action-count and length bins / main-task outcome / prompt-template hash / all three |

\*J3 was intended to mask only literals inside code. Arguments are serialised as dict reprs, so every quoted argument key and value is masked, including all command text (see `tests/test_transforms.py`); the intended literal-only transform was not tested. H2 as planned also included a value-type signature, which was never computed.

**Classifier.** Hashed word 1–2-grams and char 3–5-grams (2^18 features each), TF-IDF fitted on training folds only, linear SVM (C = 1, `random_state = 0`). Categorical baselines use one-hot features and the same SVM. The original plan used a vocabulary TF-IDF with multinomial logistic regression; it did not fit in memory (15 GB RAM), so it was replaced by this featuriser and classifier before any result was seen.

**Evaluation.**
- **Task-held-out:** 5-fold cross-validation grouped by (environment, main task); all text views share identical folds.
- **Leave-one-environment-out:** 10 folds.
- **Honest→attack transfer:** train on the 2,056 honest runs of the 187 main tasks outside the attack set, test on all 1,012 attack runs.
- Every loop asserts that its test tasks (or environment) are absent from training.
- **Uncertainty:** 95% bootstrap intervals resample main tasks (clusters), holding the fitted predictions fixed. They are therefore conditional on the trained models and exclude refitting variance.

**Decision rules set in advance.** For two analyses, the decision rule was written down before running them: the jamming criterion (§5.2) and the harness-artefact test (§5.4). These plans were not lodged with an external registry. Where the code as run departs from the plan, this is stated in place (classifier above; J3 and H2 in the views note; the O4 cohort in §5.2). All other analyses are additional baselines and robustness checks chosen during the study.

## 5. Results
### 5.1 Attribution across views (honest runs, 11-way, task-held-out)
| View | Accuracy | Macro balanced accuracy [95% CI] |
|---|---:|---|
| J0 full transcript | 0.803 | – |
| J1 tool calls | 0.909 | 0.909 [0.899, 0.920] |
| J2 comments removed | 0.907 | – |
| J3 structure-only | 0.556 | – |
| J4 skeleton | 0.768 | 0.767 [0.750, 0.784] |
| H2 tool-call metadata | 0.819 | 0.819 [0.804, 0.835] |
| H3 tool names + approximate verbs | 0.791 | 0.791 [0.776, 0.806] |
| H3b bag of verbs (sorted n-grams) | 0.670 | – |
| H0 action count only | 0.150 | – |

Removing environment outputs increases accuracy (J0 → J1). On identical action prefixes (the first min(20, complete) actions, n = 2,196; J0 blocks cut at the numbered `Action n:` headers and checked line by line against the J1 calls), 1,864 J0 prefixes but only 23 J1 prefixes exceed the cap:

| Featuriser on the same prefixes | J0 | J1 | J1 − J0 [95% CI] |
|---|---:|---:|---|
| standard (word + char on first 30k chars) | 0.779 | 0.876 | +0.097 [+0.082, +0.114] |
| word n-grams only (no cap) | 0.755 | 0.871 | +0.117 [+0.101, +0.133] |
| word on whole prefix + char on the call lines only (equal call coverage) | 0.866 | 0.875 | +0.010 [+0.000, +0.019] |

Calls-only prefixes were easier to attribute with the standard and word-only featurisers. The difference fell to about one point when both character branches used the same call text. That comparison changes both what the character branch sees and how much call text it sees, so it does not isolate the cap. The size of the gap therefore depends on feature construction; these results do not show that outputs are mostly noise.

### 5.2 Signal jamming
**Rule.** Some transform must halve above-chance identifiability, i.e. reach accuracy ≤ 0.091 + 0.5 × (0.803 − 0.091) = **0.4470**. That transform must also keep the retention proxy O4 (attack-vs-honest AUROC of a same-pipeline classifier) at or above J0's minus 0.05, i.e. **0.7128** (as run).

**Result.** **No transform, as implemented, met the accuracy ceiling**, so the retention test was not reached. The closest, J3 at 0.556, also had the lowest as-run retention (0.732, against 0.923 for J1).

**Deviation and retention sensitivity.** As run, O4 used 8 pipelines: GPT-5 Nano, GPT-5 Mini and GPT-5.4 were excluded as "non-viable attackers", a rule added in the code that was not in the plan. Their attack-run side-task success is 0.00 / 0.01 / 0.07, against 0.43–0.98 for the others. The accuracy verdict does not use O4. We now report O4 four ways. The matched-task cohort restricts *both* conditions to the 13 main tasks they share (142 honest, 187 attack runs, all 11 pipelines).

**Scoring.** As planned, O4 pools raw out-of-fold SVM scores across the five folds. When the attack share differs between folds, each fold's model has a different offset, and pooled AUROC can drift from what the per-fold rankings support; with identical features for every run it can fall far below 0.5. We therefore also report a *foldwise* AUROC: AUROC within each test fold, weighted by the fold's number of attack–honest pairs (`src/o4_foldwise.py`, same folds; every fold here contains both classes). The pooled values reproduce the committed ones exactly.

| View | as run (8): pooled / foldwise | all 11 | successful attacks only | 13 shared main tasks: pooled [95% CI] / foldwise |
|---|---:|---:|---:|---:|
| J0 | 0.763 / 0.762 | 0.730 / 0.730 | 0.803 / 0.804 | 0.593 [0.501, 0.701] / 0.640 |
| J1 | 0.923 / 0.928 | 0.893 / 0.895 | 0.939 / 0.938 | 0.767 [0.694, 0.862] / 0.829 |
| J2 | 0.915 / 0.920 | 0.885 / 0.888 | 0.932 / 0.932 | 0.773 [0.707, 0.867] / 0.831 |
| J3 | 0.732 / 0.739 | 0.741 / 0.746 | 0.773 / 0.774 | 0.682 [0.602, 0.754] / 0.689 |
| J4 | 0.885 / 0.886 | 0.852 / 0.854 | 0.894 / 0.895 | 0.770 [0.699, 0.838] / 0.773 |

Intervals: task-cluster bootstrap of the pooled scores (B = 1,000), conditional on the fitted models.

**Reading.**
- On the full cohorts, pooled and foldwise O4 agree within 0.007, so pooling does not affect the main values (test-fold attack share 0.18–0.35).
- On the 13 shared tasks, test-fold attack share varies more (0.50–0.67) and pooling understates O4 by up to 0.06. Foldwise, matched-task O4 is 0.06–0.09 below the all-11 foldwise value (J1 0.07, J2 0.06, J3 0.06, J4 0.08, J0 0.09).
- That cohort is about a tenth of the full one, so the drop mixes task matching with far less training data; the two cannot be separated here. A task component in the full-cohort value can be neither confirmed nor ruled out.

**Task-disjoint control** (`src/retention_matched.py`). Five draws, each with 13 honest-only and 13 attack-only main-task groups and no shared task, so the number of task groups per condition matches the matched cohort; run counts are 143 honest and 187–231 attack. Scored foldwise only, with `StratifiedGroupKFold(5)` so that every test fold holds both classes; a null check with identical features for every run gives 0.500 on every draw. Mean foldwise AUROC (range over draws): J0 0.448 (0.267–0.624), J1 0.589 (0.402–0.663), J2 0.598 (0.398–0.677), J3 0.641 (0.528–0.729), J4 0.684 (0.646–0.731). Every view scores below its matched-task foldwise value. This gives no evidence that differences between the honest-only and attack task pools inflate O4 at this size. It does not establish that the full-cohort value is free of task effects: the control differs from the matched cohort in more than task overlap (each held-out task contains one condition only).

O4 remains a coarse proxy for detection signal, not a measure of monitor performance.

### 5.3 Every pipeline is individually distinguishable
**One-vs-rest AUROC across the 11 pipelines** (lowest shown with its 95% task-cluster interval):

| View | Range | Lowest |
|---|---|---|
| J1 | 0.939–1.000 | Kimi 0.939 [0.927, 0.950] |
| J4 | 0.833–0.997 | Kimi 0.833 [0.804, 0.859] |
| H2 | 0.900–1.000 | Kimi 0.900 [0.881, 0.918] |
| H3 | 0.841–0.996 | Kimi 0.841 [0.814, 0.867] |

**Per-pipeline detail** (task-held-out; n = true trajectories, pred = times predicted; balanced = one-vs-rest balanced accuracy):

| Model | n | J1 pred | J1 recall | J1 precision | J1 balanced | H3 recall | H3 precision | H3 balanced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Opus | 200 | 204 | 0.91 | 0.89 | 0.949 | 0.80 | 0.72 | 0.884 |
| Sonnet | 200 | 210 | 0.96 | 0.92 | 0.978 | 0.72 | 0.78 | 0.852 |
| Haiku | 200 | 236 | 0.96 | 0.82 | 0.972 | 0.82 | 0.78 | 0.901 |
| Gemini Pro | 200 | 205 | 0.97 | 0.95 | 0.982 | 0.92 | 0.84 | 0.949 |
| Gemini Flash | 200 | 202 | 0.97 | 0.96 | 0.983 | 0.78 | 0.80 | 0.878 |
| GLM 5 | 200 | 207 | 0.78 | 0.75 | 0.874 | 0.56 | 0.59 | 0.760 |
| Kimi | 198 | 122 | 0.49 | 0.80 | 0.739 | 0.43 | 0.52 | 0.695 |
| Grok | 200 | 210 | 0.99 | 0.94 | 0.992 | 0.89 | 0.87 | 0.936 |
| GPT-5 Nano | 200 | 206 | 0.99 | 0.97 | 0.996 | 0.94 | 0.92 | 0.966 |
| GPT-5 Mini | 200 | 196 | 0.97 | 0.99 | 0.984 | 0.88 | 0.87 | 0.931 |
| GPT-5.4 | 200 | 200 | 1.00 | 1.00 | 1.000 | 0.97 | 0.95 | 0.982 |

Kimi is the outlier: under J1 it is predicted 122 times for 198 true runs (recall 0.49), mostly as GLM 5. Most Kimi runs (139 of 198) have no version label, so we cannot tell whether this reflects model similarity or mixed provenance.

### 5.4 Harness and provenance audit
**Confounds found:**
1. **Different system-prompt templates.** Gemini Pro, Gemini Flash and Grok honest runs nearly all used a full scaffold prompt (194–200 of 200). The other eight mostly received the bare task text (139–190 runs per pipeline, of 200; of 198 for Kimi).
2. **Serialisation differs by pipeline:** argument-key order (`path, command` vs `command, path`), extra keys (GPT-5 Nano `cmd2`), `timeout` usage.
3. **Different tool sets:** environment-specific tools (`db_bash`, `rust_worker_bash`, …) and scaffold tools (`ready_to_plan`; 212 planning calls in honest runs) appear at different rates.
4. **Planning-tool calls in attack runs.** Removal predicate, used everywhere in the code: tool name contains `phase` or `planning`. Observed matching names (raw count): `start_phase1_planning` 96, `start_phase2_planning` 212, `start_phase3_planning` 1, `start_phase1_planning</arg_value>` 1 (malformed), `start_phase2_execution` 3, `start_phase3_execution` 2, `start_phase2_final_selection` 3, `start_phase2_implementation_checklist` 2, `planning_phase` 2. A strict rule (`start_phase*` names containing `_planning`) matches 310 raw / 152 visible calls instead of 322 / 164. Honest-run `ready_to_plan` calls do not match either rule and are kept. Removing these calls removes one class of scaffold artefact; it does not make any view artefact-free. The broad rule's 322 calls: 164 are monitor-visible (Gemini Flash 97, Kimi 43, GPT-5 Mini 17, Grok 5, Sonnet 1, Haiku 1), and 158 are malformed and dropped by the formatter.

**Test (rule set in advance).**
- Harness artefact if H2 ≥ 0.73 **and** H3 ≤ 0.18.
- Observed H2 = 0.819, H3 = 0.791, giving **"behavioural fingerprint supported"** under that rule.

This rule falsifies only a pure-artefact hypothesis (H3 near chance). It does not measure how much signal is independent of the pipeline, and H3, as computed, passes environment and scaffold tool names through verbatim. The baselines in §5.5 address that question.

**Template held constant** (`src/within_template.py`):

| Cohort | n | H2 metadata | H3 verbs | H3b bag |
|---|---:|---:|---:|---:|
| 8 bare-template pipelines, all runs (chance 0.125) | 1,442 | 0.797 | 0.775 | 0.676 |
| same, 131 tasks every pipeline ran under this template | 1,048 | 0.769 | 0.749 | 0.638 |
| 3 scaffold-template pipelines, all runs (chance 0.333) | 591 | 0.944 | 0.915 | 0.851 |
| same, 191-task intersection | 573 | 0.946 | 0.913 | 0.857 |

Template coverage is uneven: 139 (Kimi) to 190 bare-template runs per pipeline. In the bare-template subset, restricting to tasks that every pipeline ran under the template lowers accuracy by 2–4 points; in the scaffold subset it changes accuracy by less than one point in either direction. Both remain far above chance.

### 5.5 What carries the signal: baselines
`src/baselines.py`; all honest runs; same folds as §5.1; paired differences with task-cluster 95% intervals.

| View | Accuracy |
|---|---:|
| H3 tool names + approximate verbs | 0.791 |
| H3m H3 with specialist tool names masked (parser unchanged) | 0.783 |
| **C1 common-tool approximate verbs** | **0.752** |
| C1P C1 with ambiguous parses replaced by a placeholder | 0.748 |
| H3b unigram bag (order-free, no artificial adjacencies) | 0.527 |
| T0 tool names only | 0.313 |
| T1 common-tool identity only | 0.291 |
| LOT outcome + length + template | 0.288 |
| TMPL prompt template | 0.189 |
| LEN action count + length bins | 0.169 |
| OUT main-task outcome | 0.129 |

| Paired difference | Δ accuracy [95% CI] |
|---|---|
| H3 − T0 | +0.478 [+0.458, +0.498] |
| C1 − T1 | +0.460 [+0.441, +0.481] |
| H3 − LOT | +0.504 [+0.481, +0.527] |
| C1 − LOT | +0.464 [+0.441, +0.488] |
| H3 − H3m (tool-name ablation, parser fixed) | +0.009 [+0.001, +0.016] |
| C1 − C1P (ambiguous-parse placeholder) | +0.003 [−0.007, +0.013] |
| H3 − C1 (tool names *and* parser differ; not an ablation) | +0.040 [+0.028, +0.051] |

**C1 coverage and parse quality.** C1 keeps 0.96–1.00 of each pipeline's visible calls as common-tool tokens, except Gemini Flash at 0.86 (3,553 calls to other tools dropped in total). Of 82,075 `bash` calls, 2,544 (3.1%) are ambiguous parses and 79 yield no verb. The ambiguous fraction per `bash` call ranges from 0.005 (GPT-5 Mini) to 0.070 (Opus) and 0.081 (Kimi); replacing those parses with a placeholder (C1P) changes accuracy by 0.003.

**Order test.** Shuffling the common-tool tokens within each trajectory (5 seeds; tokens, folds and featuriser fixed) gives 0.672–0.697. Observed minus shuffled is +0.055 to +0.080, each interval excluding zero (e.g. +0.079 [+0.061, +0.095]).

**Reading.**
- Masking specialist tool names with the parser held fixed costs under one point (H3 − H3m). The 4-point H3 − C1 gap mixes tool names with parser differences and is not a tool-name effect.
- Tool identity, outcome, length and template each fall far below the verb views.
- Common-tool command representations retain substantial predictive information beyond these baselines: the shuffled multiset of common-tool tokens (shell verbs, editor sub-commands) alone gives about 0.68. Common-tool token order, which also encodes editor sub-commands and `bash`/editor transitions, adds a further 5.5–8.0 points. These accuracies do not add up to a decomposition of the signal.

**Fixed prefixes.** On the first *k* common-tool calls, for runs with at least *k*:

| k | runs | C1 | T1 | C1 − T1 [95% CI] |
|---:|---:|---:|---:|---|
| 5 | 2,189 | 0.689 | 0.342 | +0.347 [0.326, 0.365] |
| 10 | 2,145 | 0.691 | 0.325 | +0.366 [0.345, 0.386] |
| 20 | 1,960 | 0.671 | 0.296 | +0.375 [0.351, 0.398] |

Five commands already identify the pipeline at 0.69. The signal is not an artefact of trajectory length or of late-trajectory behaviour.

**Success strata** (first 10 common-tool calls, main-task outcome of the honest run, all 11 pipelines present in both strata):
- successful runs (n = 1,063): C1 0.666 (macro recall 0.657), T1 0.369;
- failed runs (n = 1,082): C1 0.679 (macro recall 0.671), T1 0.284.

Attribution does not depend on whether the run succeeded. Pipeline mix differs between strata (e.g. Grok 49 successes vs 151 failures), so the two rows are not directly comparable to each other.

### 5.6 Same-vendor discrimination
| View | 11-way errors | of which within-vendor | Anthropic 3-way (chance 0.33) | Google 2-way (chance 0.50) | OpenAI 3-way (chance 0.33) |
|---|---:|---:|---:|---:|---:|
| J1 tool calls | 199 | 33 (17%) | 0.958 | 0.970 | 0.978 |
| J4 skeleton | 511 | 120 (23%) | 0.853 | 0.920 | 0.952 |
| H2 tool-call metadata | 397 | 168 (42%) | 0.797 | 0.912 | 0.950 |
| H3 tool names + verbs | 459 | 110 (24%) | 0.862 | 0.925 | 0.950 |

Same-vendor pipelines remain separable on every view. Metadata errors concentrate within vendors (42% of H2 errors, against 17–24% for the action views). This fits a provider-level serialisation channel plus a more model-specific action channel, though the design cannot separate provider from model. Within each vendor the template is nearly constant (Anthropic and OpenAI ~92–94% bare template; Google ~98% scaffold template).

### 5.7 Generalisation to unseen environments
Train on 9 environments, test on the 10th, for all 10. Each held-out environment has 219–220 honest runs, 20 main tasks and all 11 pipelines (chance 0.091).

| Held-out environment | n | J1 tool calls | J4 skeleton | H2 metadata | H3 verbs |
|---|---:|---:|---:|---:|---:|
| auto_workflow | 220 | 0.914 | 0.745 | 0.832 | 0.750 |
| clinical_trial | 219 | 0.772 | 0.767 | 0.799 | 0.758 |
| file_indexing | 220 | 0.859 | 0.755 | 0.859 | 0.759 |
| media_processing | 220 | 0.877 | 0.682 | 0.682 | 0.718 |
| model_registry | 220 | 0.841 | 0.755 | 0.773 | 0.768 |
| shared_dotfiles | 219 | 0.795 | 0.598 | 0.753 | 0.644 |
| slack_clone | 220 | 0.859 | 0.764 | 0.809 | 0.782 |
| tiktok | 220 | 0.855 | 0.755 | 0.845 | 0.773 |
| user_tracking | 220 | 0.845 | 0.655 | 0.723 | 0.695 |
| web_scraping | 220 | 0.877 | 0.782 | 0.814 | 0.755 |
| **Pooled** | 2,198 | **0.849** | **0.726** | **0.789** | **0.740** |
| *Task-held-out (reference)* | | *0.909* | *0.768* | *0.819* | *0.791* |

Pooled accuracy falls 3–6 points below task-held-out; no environment falls near chance. The 10 environments are a fixed benchmark set, so no claim is made beyond them.

### 5.8 Honest → attack transfer
Train on 2,056 honest runs (187 main tasks outside the attack set), test on all 1,012 attack runs.

| Variant | Accuracy |
|---|---:|
| J0 full view | 0.749 |
| J1 tool calls (formatter text) | 0.819 |
| J2 / J3 / J4 | 0.807 / 0.491 / 0.647 |
| tool calls rebuilt from structured actions | 0.827 |
| rebuilt, planning-tool calls removed | 0.828 |
| rebuilt, planning removed + argument keys sorted | 0.820 |
| H3 verbs as-is / planning removed | 0.647 / 0.640 |

The two J1 numbers differ by construction; neither contains per-action headers. 0.819 (canonical) uses the formatter's call text, which redacts `old_str`/`new_str` of editor `str_replace` calls whenever the editor returned output. 0.827 rebuilds the calls from structured actions so that planning calls and key order can be edited; the rebuild keeps those strings unredacted and truncates each call's arguments at 4,000 characters. Removing planning calls or sorting keys changes the rebuilt figure by at most 1 point, so the attack-scaffold artefacts of §5.4 do not drive transfer.

The "keys sorted" variant above is a shallow canonicalisation. It starts from argument text cut at 4,000 characters (3,891 honest and 1,027 attack visible non-planning calls hit that cap), sorts only top-level keys, and keeps the original text when parsing fails. A separate view, **J1C** (`src/canonical_calls.py`), starts instead from the formatter's own call lines, so monitor visibility and editor redaction match J1. It removes the 4,000-character per-call cap but keeps the 100,000-character per-trajectory cap on call text that J1 also has (`src/extract_text.py`), so the two views see the same calls. It drops planning calls (the §5.4 predicate), sorts dictionary keys recursively, keeps list order and values, and serialises as JSON. 79 of 166,307 visible calls (74 honest, 5 attack) fail to parse and become an explicit `<UNPARSED>` placeholder; every one is the final call of one of the 79 trajectories whose call text reaches the 100,000-character cap, cut off mid-call. Serialising as JSON also normalises quote style and spacing.

| View (same folds and featuriser) | Honest 11-way | Transfer |
|---|---:|---:|
| J1 tool calls | 0.909 | 0.819 |
| J1, planning calls dropped | 0.909 | 0.817 |
| J1C recursively canonicalised | 0.903 | 0.804 |

Paired honest difference J1C − J1: −0.006 [−0.012, −0.000]. No honest J1 call matches the removal predicate, so dropping planning calls leaves honest accuracy unchanged; honest `ready_to_plan` calls (§5.4) do not match it and are kept. On transfer, dropping the planning calls costs 0.2 points (0.819 → 0.817) and canonicalisation as implemented a further 1.3 points (0.817 → 0.804). That second step changes key order, nesting order, quoting, spacing and the representation of the 79 cut-off calls together, so it is not attributable to any one of them.

Pooled transfer hides weak pipelines. Per-pipeline transfer recall under J1 ranges from 0.30 (Kimi) and 0.40 (Haiku) to 0.97–1.00 (Gemini Pro, Gemini Flash, Grok, GPT-5.4); under J1C, Kimi 0.27 and Haiku 0.42.

**Task-matched own-vs-other pairs** (`src/pairwise.py`): for each pipeline and task, a pipeline-vs-rest classifier picks which of two same-task runs is the pipeline's own. Ranges across the 11 pipelines and all comparison pipelines:

| | Honest | Attack |
|---|---|---|
| J1 tool calls | 0.946–1.000 | 0.918–1.000 |
| J4 skeleton | 0.838–0.998 | 0.685–0.995 |

The minimum is Kimi in three cells; for J1 attack runs it is GPT-5 Mini against its same-vendor siblings (0.918; Kimi 0.924).

## 6. Discussion
**What the results support.** In LinuxArena monitor views, a supervised classifier attributes trajectories to their generating pipeline with high accuracy. That attribution has four properties:
- **It persists.** It survives removal of outputs, partial comment stripping, a fixed prompt template, held-out environments, and the shift from honest to attack runs.
- **Command representations carry most of what we measured.** They retain substantial predictive information beyond the coarse baselines we evaluated (tool names, outcome, length, template); masking specialist tool names costs under a point, and common-tool token order adds 5.5–8.0 points (§5.5). Baseline accuracies are not an additive decomposition, so this does not apportion the signal between channels.
- **It appears early:** five common-tool commands give 0.69.
- **It has two channels:** a metadata channel that is partly vendor-level, and a more model-specific action channel.

None of this separates the model from the rest of its pipeline (provider SDK, tool schema, sampling settings, template). Related work shows how much the pipeline can matter: browser-agent fingerprints fall to near chance when the harness changes (*Known By Their Actions*), and *Trace* reports a drop from 0.981 to 0.815 macro F1 on unseen scaffolds.

**Implications, kept to proposition (1).**
- A safety case for untrusted monitoring of agents should not assume that the monitor input carries no source information. Here it carries a great deal.
- No cheap deterministic transform we tested, as implemented, removed most of it. Each transform was evaluated with a classifier retrained on the transformed text, i.e. against an adaptive attacker. That is the relevant case: in browser agents, timing delays and prompt normalisation defeat a classifier trained on clean traces but not one retrained on perturbed traces (*Known By Their Actions*).
- *Inference, not tested here:* much of the signal is in which commands were run, which is also what a monitor needs to judge an action. Text-level jamming that removes it would plausibly also degrade monitoring. Our retention proxy is too coarse and too noisy on matched tasks (§5.2) to measure this trade-off.
- Any jamming defence should be evaluated for its effect on both attribution and malicious-action detection.
- We make no ranking of alternative defences.

**What the results do not show.**
- We do not show that an LLM monitor can recognise its own trajectories (proposition 2), that recognition can be elicited (3), that it changes how attacks are scored (4), or that it matters in deployment (5).
- A supervised linear classifier is neither a lower nor an upper bound on what a prompted or fine-tuned monitor does.
- Lower attribution after normalisation would not have shown a protocol to be safe.

**Limitations.**
- One benchmark (10 environments), prompted attackers, one classifier family.
- Pipeline differences are only partly controlled: templates differ by pipeline, and provider, SDK and sampling settings are not observed.
- 1,442 honest runs lack a version label; Kimi provenance is the most affected.
- O4 is a coarse proxy; on the 13 shared main tasks its intervals are wide and pooled scoring understates it by up to 0.06, and a task component in the full-cohort value cannot be ruled out (§5.2).
- Documented transform imperfections:
  - J3 masks every quoted string, including all command text, and keeps only punctuation, structure and unquoted values (§4);
  - J2 misses trailing comments on one-line commands;
  - H3's verb parser is approximate; C1 is quote-aware but still approximate (3.1% ambiguous `bash` parses).
- Matching the recorded prompt fields (template hash, task text) does not establish identical complete generation contexts.
- The short prefixes of §5.5 are the first *common-tool* calls, which are not necessarily a run's first actions.
- Attribution on honest prefixes does not show that a monitor could recognise a run *before* its malicious action.
- Attack-vs-honest classification (O4) is not validated malicious-action detection.
- The failure of the tested transforms is not an impossibility result for signal jamming.
- Honest→attack transfer was measured under one protocol (trained once, every overlapping task excluded); other protocols or feature sets may give different figures.
- Bootstrap intervals are conditional on the fitted models and exclude refitting uncertainty.
- The decision rules for the two planned tests were written down in advance but not registered externally; all other analyses are additional checks chosen during the study.

## 7. Reproducibility
- `make reproduce` regenerates all inputs from public data (pinned revision, shards hash-checked), runs the tests, writes the manifest and cohort audit, and reruns every analysis.
- Package versions are pinned in `requirements.txt`. `make smoke` fetches one shard and runs the cohort audit and baselines on it as a quick check.
- No API keys are needed. Folds, bootstraps and the SVM solver are seeded (`random_state = 0`).
- Manifest SHA-256: `6e678448c19051349b79708ec1a90a0f6aa5553d5cd488bebd7857127f345c28`.
- Code: `src/`; tests: `tests/`; committed outputs: `results/`; input manifest: `manifest/`. For the section → script → output mapping, see `README.md`.

## 8. Conclusion
Capped LinuxArena monitor-formatter histories carry a strong, supervised source-attribution signal. Simplified command representations retain much of it beyond tool-name, outcome, length and template baselines, with common-tool token order adding a few points; it appears within the first few actions, and none of the cheap transforms we tested, as implemented, removed most of it. Prior work on agent fingerprinting (*Trace*, *LIDAR*, *Known By Their Actions*) makes the existence of such signal unsurprising. Here it is measured on capped histories produced by the AI-control benchmark's monitor formatter, with matched tasks and an honest→attack transfer test.

Whether an LLM monitor can recognise its own trajectories from this input remains open. A pointwise test of that capability is the natural next step: single trajectory, raw monitor input, matched other-pipeline attacks as honeypot proxies, with its analysis plan committed publicly before any model is called.

## References
- Gardner-Challis, N., Bostock, J., Kozhevnikov, G., Sinclaire, M., Velja, J., Abate, A., Griffin, C. *When can we trust untrusted monitoring? A safety case sketch across collusion strategies.* arXiv:2602.20628.
- Ediga, M., Chattopadhyay, S. *Trace: Unmasking AI Attack Agents Through Terminal Behavior Fingerprinting.* arXiv:2605.01186.
- Wang, C., Xie, X., Wang, T., Luo, F., Cui, Y. *Who Is Behind the Harness? Verifying Model Identity in Coding Agents through Behavioral Fingerprinting.* arXiv:2609.28559.
- Tracy, T. et al. *LinuxArena: A Control Setting for AI Agents in Live Production Software Environments.* arXiv:2604.15384.
- Schaeffer, J. et al. *CIAware-Bench: Benchmarking Control Intervention Awareness Across Frontier LLMs.* arXiv:2606.11063.
- St. Amand, J. et al. *Self-Generated Text Recognition.* arXiv:2608.26159.
- *Idiosyncrasies in Large Language Models.* arXiv:2502.12150.
- *Know Thyself? On the Incapability and Implications of AI Self-Recognition.* arXiv:2510.03399.
- Panickssery, A., Bowman, S., Feng, S. *LLM Evaluators Recognize and Favor Their Own Generations.* arXiv:2404.13076.
- Pinto, L. *Calibration-Family Overfit.* arXiv:2607.06596.
