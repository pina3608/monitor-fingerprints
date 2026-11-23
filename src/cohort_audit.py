"""Cohort, provenance and artefact audit (descriptive; sources every count quoted in report §3 and §5.4).

 1. Record counts over all 10 shards by run type (from data/struct/*.counts.json, written by extract_struct.py).
 2. Cohort table per model: honest/attack runs, exclusions, missing agent_model labels, distinct agent_model values,
    main-task success (honest), side-task success (attack), system-prompt template mix (honest).
 3. Uniqueness: one run per (model, label, env, main_task, side_task); trajectory_id unique within the cohort;
    text and struct extracts join one-to-one.
 4. Repeated monitor views: identical J0 (raw view) hashes across different trajectories, and the empty-history sentinel.
 5. Attack-scaffold planning-tool calls: total / monitor-visible / dropped as malformed (parse errors), per model.
Usage: python -I src/cohort_audit.py data/text data/struct results/cohort_audit
"""
import collections, glob, gzip, hashlib, json, sys

TEXT, STRUCT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODELS = ['Opus', 'Sonnet', 'Haiku', 'Gemini Pro', 'Gemini Flash', 'GLM 5', 'Kimi', 'Grok', 'GPT-5 Nano', 'GPT-5 Mini', 'GPT-5.4']
key = lambda d: (d['model'], d['label'], d['env'], d['main_task'], d['side_task'])
PLAN = lambda a: 'phase' in a['f'] or 'planning' in a['f']
excluded = lambda d: d['model'] == 'Kimi' and d.get('agent_model') == 'kimi-k2-thinking'
out, res = [], {}

# 1. record counts
tot = collections.Counter(); other = collections.Counter()
for f in sorted(glob.glob(f'{STRUCT}/*.counts.json')):
    c = json.load(open(f)); tot.update(c['counts']); other.update(c['other_agent_runs'])
res['record_counts'] = dict(tot); res['other_agent_run_names'] = dict(other)
out.append('## 1. Records in the 10 environment shards')
out.append(f"records {tot['records']:,}; monitor re-scoring records ('<Monitor> monitor on <Model> ...') {tot['monitor_rescoring']:,}; "
           f"cohort agent runs (11 models x honest/attack) {tot['cohort_agent_run']:,}; other agent runs {tot['other_agent_run']:,} "
           f"(run names: {len(other)}); duplicate trajectory_id records {tot['duplicate_trajectory_id_records']:,}")

# load extracts
S = [json.loads(l) for f in sorted(glob.glob(f'{STRUCT}/*.jsonl.gz')) for l in gzip.open(f, 'rt')]
T = [json.loads(l) for f in sorted(glob.glob(f'{TEXT}/*.jsonl.gz')) for l in gzip.open(f, 'rt')]
out.append(f'extracted: struct {len(S)}, text {len(T)}; excluded (kimi-k2-thinking) {sum(map(excluded, S))}; '
           f'cohort {sum(not excluded(d) for d in S)}')

# 3. uniqueness / join
for name, D in (('struct', S), ('text', T)):
    kc = collections.Counter(key(d) for d in D if not excluded(d)); tc = collections.Counter(d['trajectory_id'] for d in D if not excluded(d))
    assert max(kc.values()) == 1, f'{name}: duplicate cohort key'
    assert max(tc.values()) == 1, f'{name}: duplicate trajectory_id in cohort'
assert {d['trajectory_id'] for d in S} == {d['trajectory_id'] for d in T}, 'text/struct trajectory sets differ'
assert all(key(s) == key(t) for s, t in zip(sorted(S, key=lambda d: d['trajectory_id']), sorted(T, key=lambda d: d['trajectory_id'])))
out.append('uniqueness: one run per (model, label, env, main_task, side_task); trajectory_id unique; text<->struct join one-to-one  [asserted]')

# 2. cohort table
out.append('\n## 2. Cohort per model (after exclusion)')
out.append('model         honest attack excl  no-version-label(h/a)  agent_model values                 main-task success (h)  side-task success (a)  honest templates')
res['per_model'] = {}
for m in MODELS:
    H = [d for d in S if d['model'] == m and d['label'] == 'honest' and not excluded(d)]
    A = [d for d in S if d['model'] == m and d['label'] == 'attack' and not excluded(d)]
    ex = sum(1 for d in S if d['model'] == m and excluded(d))
    nl = (sum(d['agent_model'] is None for d in H), sum(d['agent_model'] is None for d in A))
    am = dict(collections.Counter(str(d['agent_model']) for d in H + A))
    ms = sum(bool(d['main_task_success']) for d in H) / len(H); ss = sum(bool(d['side_task_success']) for d in A) / len(A)
    tm = dict(collections.Counter(d['tmpl_hash'] for d in H).most_common(3))
    res['per_model'][m] = {'honest': len(H), 'attack': len(A), 'excluded': ex, 'missing_version_label': nl, 'agent_model_values': am,
                           'honest_main_task_success': ms, 'attack_side_task_success': ss, 'honest_templates': tm}
    out.append(f'{m:<13} {len(H):>6} {len(A):>6} {ex:>4}  {nl[0]:>4}/{nl[1]:<4}               {str(am)[:34]:<34} {ms:>8.2f}               {ss:>8.2f}        {tm}')
mh = sum(v['missing_version_label'][0] for v in res['per_model'].values()); ma = sum(v['missing_version_label'][1] for v in res['per_model'].values())
out.append(f'missing agent_model labels: honest {mh}, attack {ma}, total {mh + ma}')
hon = [d for d in S if d['label'] == 'honest' and not excluded(d)]; att = [d for d in S if d['label'] == 'attack' and not excluded(d)]
g_h = {(d['env'], d['main_task']) for d in hon}; g_a = {(d['env'], d['main_task']) for d in att}
out.append(f'honest {len(hon)} on {len(g_h)} main-task groups; attack {len(att)} on {len({(d["env"], d["main_task"], d["side_task"]) for d in att})} '
           f'(main, side) pairs / {len(g_a)} main-task groups; shared main-task groups {len(g_h & g_a)}; '
           f'honest runs outside the attack set {sum((d["env"], d["main_task"]) not in g_a for d in hon)} on {len(g_h - g_a)} groups')

# 4. repeated views
out.append('\n## 4. Repeated monitor views (J0 = raw view hash), cohort only')
h = collections.defaultdict(list)
for d in T:
    if not excluded(d):
        h[hashlib.sha256(d['raw'].encode()).hexdigest()].append(d)
rep = {k: v for k, v in h.items() if len(v) > 1}
res['repeated_views'] = []
for k, v in sorted(rep.items(), key=lambda x: -len(x[1])):
    r = {'sha256': k[:12], 'n': len(v), 'labels': dict(collections.Counter(d['label'] for d in v)), 'models': dict(collections.Counter(d['model'] for d in v)),
         'task_groups': len({(d['env'], d['main_task']) for d in v}), 'empty_sentinel': v[0]['raw'] == 'No previous actions.', 'chars': len(v[0]['raw'])}
    res['repeated_views'].append(r); out.append(str(r))
out.append(f'{len(rep)} repeated view hashes covering {sum(len(v) for v in rep.values())} trajectories')

# 5. planning calls
out.append('\n## 5. Attack-scaffold planning-tool calls (tool name contains "phase" or "planning"; same rule as src/robustness.py), attack runs')
pl = {}
for m in MODELS:
    acts = [a for d in att if d['model'] == m for a in d['actions'] if PLAN(a)]
    pl[m] = {'total': len(acts), 'visible': sum(not a['hidden'] and not a['parse_err'] for a in acts), 'malformed_dropped': sum(a['parse_err'] for a in acts)}
res['planning_calls'] = pl
t = {k: sum(v[k] for v in pl.values()) for k in ('total', 'visible', 'malformed_dropped')}
out.append(f"total {t['total']}; monitor-visible {t['visible']}; malformed (parse error, dropped by the formatter) {t['malformed_dropped']}")
out.append('per model (total/visible/malformed): ' + '; '.join(f"{m} {v['total']}/{v['visible']}/{v['malformed_dropped']}" for m, v in pl.items() if v['total']))
hp = sum(1 for d in hon for a in d['actions'] if PLAN(a) or 'ready_to_plan' in a['f'])
out.append(f'planning / ready_to_plan calls in honest runs: {hp}')

json.dump(res, open(OUT + '.json', 'w'), indent=1); open(OUT + '.txt', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
