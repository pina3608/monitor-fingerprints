"""Byte-equality test: our format_history port vs upstream control-tower code, on real trajectories.

Upstream modules (control-tower @ 699d5ca) are loaded from a local clone with their heavy imports
stubbed (only the pure formatting functions are exercised).

Usage: python -I tests/test_formatter_fidelity.py CONTROL_TOWER_DIR SAMPLE.jsonl [N]
"""
import json, os, sys, types
from types import SimpleNamespace

ct, sample = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import monitor_input as ours  # noqa: E402

# stub package skeleton so upstream files import
for name in ['control_tower', 'control_tower.trajectories', 'control_tower.trajectories.trajectory']:
    sys.modules[name] = types.ModuleType(name)
sys.modules['control_tower.trajectories.trajectory'].Action = object
src_dir = os.path.join(ct, 'src', 'control_tower', 'trajectories')
teo = types.ModuleType('control_tower.trajectories.text_editor_output')
exec(open(os.path.join(src_dir, 'text_editor_output.py')).read(), teo.__dict__)
sys.modules['control_tower.trajectories.text_editor_output'] = teo
fh = types.ModuleType('control_tower.trajectories.format_history')
exec(open(os.path.join(src_dir, 'format_history.py')).read(), fh.__dict__)

n = same = 0; first_diff = None; n_diffrec = 0
for line in open(sample):
    r = json.loads(line)
    acts = r.get('actions') or []
    up = fh.format_history([SimpleNamespace(function=a['function'], arguments=a.get('arguments') or {},
                                            output=a.get('output'), output_error=a.get('output_error')) for a in acts])
    mine = ours.format_history(acts)
    n += 1; same += (up == mine)
    n_diffrec += sum(1 for a in acts if a['function'] == 'text_editor' and not a.get('output')
                     and (a.get('arguments') or {}).get('command') == 'str_replace')
    if up != mine and first_diff is None:
        first_diff = r['trajectory_id']
    if n >= N:
        break
print(f'trajectories: {n}; port byte-identical to upstream: {same}/{n}; '
      f'text_editor str_replace actions with null output (diff reconstructed): {n_diffrec}')
print('first mismatch :', first_diff)
sys.exit(0 if same == n else 1)
