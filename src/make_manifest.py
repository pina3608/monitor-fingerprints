"""Write an immutable input manifest for the fingerprint study: one line per trajectory used
(trajectory_id, run_id, model, label, agent_model, env, main_task, side_task, SHA-256 of the monitor view (J0) and of the
tool-calls view (J1)), plus the shard hashes. Asserts one run per cohort key and unique trajectory ids.
Usage: python -I src/make_manifest.py data/text manifest/inputs.jsonl  (run from the repository root)"""
import glob, gzip, hashlib, json, os, sys
src, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out), exist_ok=True)
n = 0; seen_k, seen_t = set(), set()
with open(out, 'w') as g:
    for f in sorted(glob.glob(f'{src}/*.jsonl.gz')):
        for line in gzip.open(f, 'rt'):
            d = json.loads(line)
            if not (d['model'] == 'Kimi' and d.get('agent_model') == 'kimi-k2-thinking'):
                k = (d['model'], d['label'], d['env'], d['main_task'], d['side_task'])
                assert k not in seen_k and d['trajectory_id'] not in seen_t, f'duplicate {k}'
                seen_k.add(k); seen_t.add(d['trajectory_id'])
            g.write(json.dumps({k: d.get(k) for k in ('trajectory_id', 'run_id', 'model', 'label', 'agent_model', 'env', 'main_task', 'side_task')} |
                               {'excluded': d['model'] == 'Kimi' and d.get('agent_model') == 'kimi-k2-thinking',
                                'sha256_raw_view': hashlib.sha256(d['raw'].encode()).hexdigest(),
                                'sha256_calls_view': hashlib.sha256(d['calls'].encode()).hexdigest(),
                                'shard': os.path.basename(f)}) + '\n'); n += 1
open(os.path.join(os.path.dirname(out), 'SHA256SUMS.shards'), 'w').write(open('data/SHA256SUMS.raw').read())
h = hashlib.sha256(open(out, 'rb').read()).hexdigest()
open(os.path.join(os.path.dirname(out), 'MANIFEST_SHA256'), 'w').write(f'{h}  {os.path.basename(out)}\n')
print(f'{n} trajectories; manifest sha256 {h}')
