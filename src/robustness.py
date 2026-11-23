"""Robustness analyses, part 1.

 7. Same-model-vs-rest: per-model one-vs-rest AUROC from task-held-out (GroupKFold by env/main task) decision scores
    on honest runs, with 95% task-cluster bootstrap CIs (B=1000).
 8. Leave-one-environment-out: 11-way accuracy on each held-out environment (10 folds); report median/min/max only.
 9. Honest->attack transfer with attack-scaffold artefacts removed: train on honest runs of tasks not in the attack set,
    test on attack runs; variants: as-is / planning-tool calls removed / removed + argument keys canonicalised (sorted).
Views: J1 tool calls and J4 skeleton (identification.py definitions), H2 formatting-only and H3 normalised verbs (harness-audit definitions).
Usage: python -I robustness.py TEXT_DIR STRUCT_DIR OUT_PREFIX   (e.g. data/text data/struct results/robustness)
"""
import ast, collections, glob, gzip, json, os, sys, time
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.svm import LinearSVC

TEXT, STRUCT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
HA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness_audit.py')
src = open(HA).read().split("\nhonest = [d for d in docs")[0]
ns = {'__file__': HA}
_argv = sys.argv; sys.argv = ['x', STRUCT, '/dev/null']; exec(src, ns); sys.argv = _argv
MODELS, featurize, TR = ns['MODELS'], ns['featurize'], ns['ns']['TRANSFORMS']
wrapper_token, verb, visible = ns['wrapper_token'], ns['verb'], ns['visible']
rng = np.random.default_rng(0); t0 = time.time()

key = lambda d: (d['model'], d['label'], d['env'], d['main_task'], d['side_task'])
text = {key(d): d for d in (json.loads(l) for f in sorted(glob.glob(f'{TEXT}/*.jsonl.gz')) for l in gzip.open(f, 'rt'))}
struct = [d for d in ns['docs']]  # already excludes kimi-k2-thinking
docs = [d for d in struct if key(d) in text]
assert len(docs) == len(struct), 'text/struct join mismatch'
PLAN = lambda a: 'phase' in a['f'] or 'planning' in a['f']


def calls_from_struct(d, strip_plan=False, canon=False):
    out = []
    for a in visible(d):
        if strip_plan and PLAN(a):
            continue
        r = a['repr']
        if canon:
            try:
                r = repr(dict(sorted(ast.literal_eval(r).items())))
            except Exception:
                pass  # truncated repr (>4000 chars): keep as is
        out.append(f"Tool call: {a['f']}: {r}")
    return '\n'.join(out)


VIEWS = {'J1_calls': lambda d: TR['J1_calls'](text[key(d)]),
         'J4_skeleton': lambda d: TR['J4_skeleton'](text[key(d)]),
         'H2_formatting_only': lambda d: ' '.join(wrapper_token(a) for a in visible(d)),
         'H3_norm_verbs': lambda d: ' '.join(verb(a) for a in visible(d))}
honest = [d for d in docs if d['label'] == 'honest']; attack = [d for d in docs if d['label'] == 'attack']
y = np.array([MODELS.index(d['model']) for d in honest])
g_task = np.array([f"{d['env']}/{d['main_task']}" for d in honest]); g_env = np.array([d['env'] for d in honest])
log, res = [f'trajectories honest={len(honest)} attack={len(attack)}'], {}


def auroc(pos, neg):
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float) + 1
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, r) / cnt)[inv]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


tasks = np.unique(g_task); idx_by_task = {t: np.where(g_task == t)[0] for t in tasks}
for vname, fn in VIEWS.items():
    T = [fn(d) for d in honest]
    # ---- 7. same-model-vs-rest ----
    S = np.zeros((len(T), len(MODELS)))
    for tr, te in GroupKFold(5).split(T, y, g_task):
        assert set(g_task[tr]).isdisjoint(g_task[te])
        Xtr, Xte = featurize([T[i] for i in tr], [T[i] for i in te]); S[te] = LinearSVC(C=1.0, random_state=0).fit(Xtr, y[tr]).decision_function(Xte)
    acc = float((S.argmax(1) == y).mean())
    per = {}
    for mi, m in enumerate(MODELS):
        est = auroc(S[y == mi, mi], S[y != mi, mi]); bs = []
        for _ in range(1000):
            ii = np.concatenate([idx_by_task[t] for t in rng.choice(tasks, len(tasks))])
            yy, ss = y[ii], S[ii, mi]
            bs.append(auroc(ss[yy == mi], ss[yy != mi]))
        per[m] = (float(est), *np.percentile(bs, [2.5, 97.5]).tolist())
    # ---- 8. leave-one-environment-out ----
    env_acc = {}
    for tr, te in LeaveOneGroupOut().split(T, y, g_env):
        assert set(g_env[tr]).isdisjoint(g_env[te]) and set(g_task[tr]).isdisjoint(g_task[te]), 'environment leakage'
        Xtr, Xte = featurize([T[i] for i in tr], [T[i] for i in te])
        env_acc[g_env[te][0]] = float((LinearSVC(C=1.0, random_state=0).fit(Xtr, y[tr]).predict(Xte) == y[te]).mean())
    ea = np.array(list(env_acc.values()))
    res[vname] = {'acc_task_heldout': acc, 'one_vs_rest_auroc': per, 'loeo_env_acc': env_acc}
    log.append(f"\n## {vname}  (11-way task-held-out acc {acc:.3f}; t={time.time()-t0:.0f}s)")
    log.append('  same-model-vs-rest AUROC [95% task-cluster CI]: ' + '; '.join(f'{m} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]' for m, v in per.items()))
    log.append(f'  leave-one-environment-out acc: median {np.median(ea):.3f}, min {ea.min():.3f} ({min(env_acc, key=env_acc.get)}), max {ea.max():.3f} ({max(env_acc, key=env_acc.get)})')
    print('\n'.join(log[-3:]), flush=True)

# ---- 9. artefact-stripped transfer (honest -> attack) ----
attack_main = {f"{d['env']}/{d['main_task']}" for d in attack}
tr = [i for i, d in enumerate(honest) if f"{d['env']}/{d['main_task']}" not in attack_main]
assert not ({g_task[i] for i in tr} & attack_main)
ya = np.array([MODELS.index(d['model']) for d in attack])
n_plan = collections.Counter(d['model'] for d in attack for a in visible(d) if PLAN(a))
log.append(f'\n## Honest->attack transfer (train: {len(tr)} honest runs on tasks outside the attack set)')
log.append('  planning-tool calls in attack runs by model: ' + json.dumps(dict(n_plan)))
variants = {'calls as-is': lambda d: calls_from_struct(d),
            'calls, planning removed': lambda d: calls_from_struct(d, strip_plan=True),
            'calls, planning removed + keys sorted': lambda d: calls_from_struct(d, strip_plan=True, canon=True),
            'H3 verbs as-is': lambda d: ' '.join(verb(a) for a in visible(d)),
            'H3 verbs, planning removed': lambda d: ' '.join(verb(a) for a in visible(d) if not PLAN(a))}
res['transfer'] = {}
for name, fn in variants.items():
    Xtr, Xte = featurize([fn(honest[i]) for i in tr], [fn(d) for d in attack])
    a = float((LinearSVC(C=1.0, random_state=0).fit(Xtr, y[tr]).predict(Xte) == ya).mean()); res['transfer'][name] = a
    log.append(f'  {name:<40} transfer acc {a:.3f}'); print(log[-1], flush=True)
json.dump(res, open(OUT + '.json', 'w'), indent=1); open(OUT + '.txt', 'w').write('\n'.join(log) + '\n')
print(f'done in {time.time()-t0:.0f}s')
