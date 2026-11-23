"""Extract structured per-action records (no outputs) + system-prompt fingerprints for the 11 models' agent runs.
Also writes OUT.counts.json: record counts in the shard by run type (all records, not only the cohort).
Usage: python -I extract_struct.py SHARD.jsonl OUT.jsonl.gz
"""
import collections, gzip, hashlib, json, os, re, sys
MODELS = ['Opus', 'Sonnet', 'Haiku', 'Gemini Pro', 'Gemini Flash', 'GLM 5', 'Kimi', 'Grok', 'GPT-5 Nano', 'GPT-5 Mini', 'GPT-5.4']
RUNS = {f'{m} {l}': (m, l) for m in MODELS for l in ('honest', 'attack')}
HIDDEN = {'run_side_task_test', 'submit', 'side_task_scorer'}
NAMES = re.compile(r'(claude|anthropic|opus|sonnet|haiku|gemini|google|gpt-?[\w.-]*|openai|kimi|moonshot|glm[\w.-]*|zhipu|z\.ai|grok|xai)', re.I)
n = 0; counts = collections.Counter(); runs = collections.Counter(); tids = collections.Counter()
tmp = sys.argv[2] + '.tmp'
with open(sys.argv[1]) as f, gzip.open(tmp, 'wt') as g:
    for line in f:
        r = json.loads(line)
        name = r.get('run_display_name') or ''
        counts['records'] += 1; runs[name] += 1; tids[r.get('trajectory_id')] += 1
        counts['monitor_rescoring' if ' monitor on ' in name else ('cohort_agent_run' if name in RUNS else 'other_agent_run')] += 1
        k = RUNS.get(name)
        if not k:
            continue
        sp = r.get('system_prompt') or ''
        td = (r.get('main_task') or {}).get('description') or ''
        ed = (r.get('environment') or {}).get('description') or ''
        sg = (r.get('side_task') or {}).get('goal') or ''
        tmpl = sp
        for x, ph in ((td, '<TASK>'), (ed, '<ENV>'), (sg, '<SIDE_GOAL>')):
            if x:
                tmpl = tmpl.replace(x, ph)
        tmpl = NAMES.sub('<M>', tmpl)
        acts = []
        for a in r.get('actions') or []:
            args = a.get('arguments') or {}
            acts.append({'f': a['function'], 'hidden': a['function'] in HIDDEN,
                         'parse_err': bool((a.get('output_error') or '').startswith('parsing:')),
                         'keys': list(args.keys()), 'repr': repr(args)[:4000],
                         'cmd': args.get('cmd') if isinstance(args.get('cmd'), str) else None,
                         'sub': args.get('command') if a['function'] == 'text_editor' else None,
                         'has_reason': bool(a.get('agent_reasoning')), 'has_think': bool(a.get('extended_thinking')),
                         'has_inter': bool(a.get('inter_action_messages'))})
        g.write(json.dumps({'model': k[0], 'label': k[1], 'trajectory_id': r.get('trajectory_id'), 'run_id': r.get('run_id'),
                            'main_task_success': r.get('main_task_success'), 'side_task_success': r.get('side_task_success'), 'agent_model': r.get('agent_model'), 'env': r['environment']['id'],
                            'main_task': (r.get('main_task') or {}).get('id'), 'side_task': (r.get('side_task') or {}).get('id'),
                            'sys_hash': hashlib.sha1(sp.encode()).hexdigest()[:12],
                            'sys_norm_hash': hashlib.sha1(NAMES.sub('<M>', sp).encode()).hexdigest()[:12],
                            'sys_len': len(sp), 'tmpl_hash': hashlib.sha1(tmpl.encode()).hexdigest()[:12], 'tmpl': tmpl[:3000], 'tmpl_len': len(tmpl), 'first_user_len': len(r.get('initial_user_message') or ''),
                            'actions': acts}) + '\n')
        n += 1
print(sys.argv[1], n, file=sys.stderr)
os.replace(tmp, sys.argv[2])
counts['distinct_trajectory_ids'] = len(tids); counts['duplicate_trajectory_id_records'] = sum(c - 1 for c in tids.values() if c > 1)
json.dump({'shard': os.path.basename(sys.argv[1]), 'counts': dict(counts),
           'other_agent_runs': {k: v for k, v in sorted(runs.items()) if ' monitor on ' not in k and k not in RUNS}},
          open(sys.argv[2].replace('.jsonl.gz', '.counts.json'), 'w'), indent=1)
