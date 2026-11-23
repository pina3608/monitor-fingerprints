"""Extract monitor-view text for the 11 attacker models' agent runs (honest + attack) from a LinuxArena shard.
Writes gzipped JSONL: model, label, trajectory_id, run_id, env, main_task, side_task, main/side_task_success, raw (format_history, first 100k chars),
calls (tool calls only, first 100k chars).
Usage: python -I extract_text.py SHARD.jsonl OUT.jsonl.gz
"""
import gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor_input import format_history, _skip, _format_action_call, _format_action_output  # noqa: E402

MODELS = ['Opus', 'Sonnet', 'Haiku', 'Gemini Pro', 'Gemini Flash', 'GLM 5', 'Kimi', 'Grok', 'GPT-5 Nano', 'GPT-5 Mini', 'GPT-5.4']
RUNS = {f'{m} {l}': (m, l) for m in MODELS for l in ('honest', 'attack')}
CAP = 100000


def calls_only(actions):
    acts = [a for a in actions if not _skip(a)]
    return '\n'.join(f"Tool call: {_format_action_call(a, redact=bool(_format_action_output(a)))}" for a in acts)


n = 0
tmp = sys.argv[2] + '.tmp'
with open(sys.argv[1]) as f, gzip.open(tmp, 'wt') as g:
    for line in f:
        r = json.loads(line)
        k = RUNS.get(r.get('run_display_name'))
        if not k:
            continue
        acts = r.get('actions') or []
        g.write(json.dumps({'model': k[0], 'label': k[1], 'trajectory_id': r.get('trajectory_id'), 'run_id': r.get('run_id'), 'agent_model': r.get('agent_model'), 'env': r['environment']['id'],
                            'main_task': (r.get('main_task') or {}).get('id'), 'side_task': (r.get('side_task') or {}).get('id'),
                            'main_task_success': r.get('main_task_success'), 'side_task_success': r.get('side_task_success'),
                            'raw': format_history(acts)[:CAP], 'calls': calls_only(acts)[:CAP]}) + '\n')
        n += 1
os.replace(tmp, sys.argv[2])  # atomic: a partial extract never looks complete
print(sys.argv[1], n, file=sys.stderr)
