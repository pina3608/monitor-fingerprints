"""Study 2: harness/provenance audit (decision rule set before running; report §5.4). H1 descriptive provenance; H2 wrapper-only; H3 normalised verbs; H3b bag of verbs.
Classifier/featuriser and CV identical to identification.py (its definitions are loaded, not copied).
Usage: python -I src/harness_audit.py data/struct results/harness_audit
"""
import collections, glob, gzip, json, math, os, re, sys
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'identification.py')).read().split("honest = [d for d in docs")[0]
src = src.replace("TEXT, OUT = sys.argv[1], sys.argv[2]", "TEXT, OUT = '', ''")
src = src.replace("docs = [json.loads(l) for f in sorted(glob.glob(f'{TEXT}/*.jsonl.gz')) for l in gzip.open(f, 'rt')]", "docs = []")
ns = {}; exec(src, ns)
MODELS, featurize = ns['MODELS'], ns['featurize']
STRUCT, OUT = sys.argv[1], sys.argv[2]
docs = [json.loads(l) for f in sorted(glob.glob(f'{STRUCT}/*.jsonl.gz')) for l in gzip.open(f, 'rt')]
docs = [d for d in docs if not (d['model'] == 'Kimi' and d['agent_model'] == 'kimi-k2-thinking')]
log = [f'trajectories: {len(docs)}']

# ---------------- H1 provenance ----------------
log.append('\n## H1 provenance (per model)')
for m in MODELS:
    D = [d for d in docs if d['model'] == m]
    tm = collections.Counter((d['label'], d['tmpl_hash']) for d in D)
    tools = collections.Counter(a['f'] for d in D for a in d['actions'])
    keyorders = collections.Counter((a['f'], tuple(a['keys'])) for d in D for a in d['actions'] if not a['hidden'])
    flags = {k: np.mean([any(a[k] for a in d['actions']) for d in D]) for k in ('has_reason', 'has_think', 'has_inter', 'parse_err')}
    ids = collections.Counter(d['agent_model'] for d in D)
    log.append(f"{m:<13} templates(label,hash)={dict(tm)}  agent_model ids={dict(ids)}")
    log.append(f"{'':<13} tools={dict(tools.most_common(6))}")
    log.append(f"{'':<13} top arg-key orders={[(f'{f}:{list(k)}', n) for (f, k), n in keyorders.most_common(5)]}")
    log.append(f"{'':<13} share of trajectories with agent_reasoning={flags['has_reason']:.2f} extended_thinking={flags['has_think']:.2f} inter_action_msgs={flags['has_inter']:.2f} parse_errors={flags['parse_err']:.2f}")
# templates shared across models?
tmpl_models = collections.defaultdict(set)
for d in docs:
    tmpl_models[(d['label'], d['tmpl_hash'])].add(d['model'])
log.append('distinct (label, template) groups and the models using each: ' + json.dumps(
    {f'{k[0]}:{k[1]}': sorted(v) for k, v in sorted(tmpl_models.items(), key=lambda x: -len(x[1]))}))
examples = {}
for d in docs:
    examples.setdefault((d['label'], d['tmpl_hash']), d['tmpl'][:600])

# ---------------- views ----------------
ESC = [('\\n', 'NL'), ('\\t', 'TAB'), ('\\\\', 'BS'), ('\\u', 'UNI'), ('\\x', 'HEX')]
def vtype(v):
    return type(v).__name__


def wrapper_token(a):
    r = a['repr']
    esc = ''.join(t for e, t in ESC if e in r)
    quotes = ''.join('d' if seg.startswith('"') else 's' for seg in re.findall(r": (['\"])", r))
    lb = int(math.log2(len(r) + 1))
    return f"{a['f']}|{'+'.join(a['keys'])}|q{quotes}|e{esc}|L{lb}"


PREFIX = re.compile(r'^\s*((\w+=\S*\s+)+|sudo\s+|timeout\s+\S+\s+|(cd\s+\S+\s*&&\s*))+')
def verb(a):
    if a['f'] == 'bash' and a['cmd'] is not None:
        c = a['cmd'].strip()
        for _ in range(3):
            c = PREFIX.sub('', c)
        tok = c.split()[0] if c.split() else '?'
        return 'bash:' + os.path.basename(tok.strip('\'"()'))
    if a['f'] == 'text_editor':
        return 'edit:' + (a['sub'] or '?')
    return a['f']


def visible(d):
    return [a for a in d['actions'] if not a['hidden'] and not a['parse_err']]


VIEWS = {'H2_wrapper_only': lambda d: ' '.join(wrapper_token(a) for a in visible(d)),
         'H3_norm_verbs': lambda d: ' '.join(verb(a) for a in visible(d)),
         'H3b_bag_of_verbs': lambda d: ' '.join(sorted(verb(a) for a in visible(d))),
         'H0_n_actions_only': lambda d: ' '.join(['x'] * min(len(visible(d)), 400))}

honest = [d for d in docs if d['label'] == 'honest']
y = np.array([MODELS.index(d['model']) for d in honest]); g = np.array([f"{d['env']}/{d['main_task']}" for d in honest])
res = {}
log.append('\n## 11-way identification, honest runs, task-held-out (chance 0.091; study-1 refs: J1 calls 0.909, J4 skeleton 0.768)')
for name, fn in VIEWS.items():
    T = [fn(d) for d in honest]
    pred = np.empty(len(T), int)
    for tr, te in GroupKFold(5).split(T, y, g):
        assert set(g[tr]).isdisjoint(g[te]), 'task leakage across folds'
        Xtr, Xte = featurize([T[i] for i in tr], [T[i] for i in te])
        pred[te] = LinearSVC(C=1.0, random_state=0).fit(Xtr, y[tr]).predict(Xte)
    acc = accuracy_score(y, pred); res[name] = acc
    rec = {m: round(float((pred[y == i] == i).mean()), 2) for i, m in enumerate(MODELS)}
    log.append(f'{name:<20} acc={acc:.3f} ({acc/ns["CHANCE"]:.1f}x chance)  per-model recall={rec}')
    print(log[-1], flush=True)

h2, h3 = res['H2_wrapper_only'], res['H3_norm_verbs']
verdict = ('HARNESS ARTEFACT' if (h2 >= 0.73 and h3 <= 0.18) else
           'BEHAVIOURAL FINGERPRINT SUPPORTED' if h3 >= 0.27 else 'MIXED')
log.append(f'\nDecision (rule set in advance): {verdict}   [H2={h2:.3f}, H3={h3:.3f}]')
json.dump({'acc': res, 'verdict': verdict, 'template_examples': {f'{k[0]}:{k[1]}': v for k, v in examples.items()}},
          open(OUT + '.json', 'w'), indent=1)
open(OUT + '.txt', 'w').write('\n'.join(log) + '\n'); print('\n'.join(log))
