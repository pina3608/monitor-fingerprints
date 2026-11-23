"""Robustness analyses, part 2.

 A. Per-model balanced accuracy, precision, recall and class coverage (task-held-out 5-fold), plus macro balanced
    accuracy with a 95% task-cluster bootstrap CI.
 B. Leave-one-environment-out: per-environment table (honest trajectories, distinct tasks, models covered,
    majority-class baseline, uniform chance, accuracy per view) and pooled accuracy over all held-out predictions.
 C. Same-vendor discrimination: (i) within-vendor confusion share of 11-way errors; (ii) dedicated within-vendor
    classifiers (Anthropic 3-way, Google 2-way, OpenAI 3-way), task-held-out, with the system-prompt template mix.
Views: J1 tool calls, J4 skeleton, H2 formatting only, H3 normalised verbs (same definitions as robustness.py).
Usage: python -I robustness2.py TEXT_DIR STRUCT_DIR OUT_PREFIX   (e.g. data/text data/struct results/robustness2)
"""
import collections, glob, gzip, json, os, sys, time
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.svm import LinearSVC

TEXT, STRUCT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
HA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness_audit.py')
ns = {'__file__': HA}; _a = sys.argv; sys.argv = ['x', STRUCT, '/dev/null']
exec(open(HA).read().split("\nhonest = [d for d in docs")[0], ns); sys.argv = _a
MODELS, featurize, TR = ns['MODELS'], ns['featurize'], ns['ns']['TRANSFORMS']
VENDOR = ns['ns']['VENDOR']; wrapper_token, verb, visible = ns['wrapper_token'], ns['verb'], ns['visible']
key = lambda d: (d['model'], d['label'], d['env'], d['main_task'], d['side_task'])
text = {key(d): d for d in (json.loads(l) for f in sorted(glob.glob(f'{TEXT}/*.jsonl.gz')) for l in gzip.open(f, 'rt'))}
honest = [d for d in ns['docs'] if d['label'] == 'honest']
assert all(key(d) in text for d in honest)
VIEWS = {'J1_calls': lambda d: TR['J1_calls'](text[key(d)]), 'J4_skeleton': lambda d: TR['J4_skeleton'](text[key(d)]),
         'H2_formatting_only': lambda d: ' '.join(wrapper_token(a) for a in visible(d)),
         'H3_norm_verbs': lambda d: ' '.join(verb(a) for a in visible(d))}
y = np.array([MODELS.index(d['model']) for d in honest])
g_task = np.array([f"{d['env']}/{d['main_task']}" for d in honest]); g_env = np.array([d['env'] for d in honest])
tmpl = np.array([d['tmpl_hash'] for d in honest])
rng = np.random.default_rng(0); t0 = time.time(); res = {}; log = []
tasks = np.unique(g_task); idx = {t: np.where(g_task == t)[0] for t in tasks}


def fitpred(T, tr, te, yy):
    Xtr, Xte = featurize([T[i] for i in tr], [T[i] for i in te]); return LinearSVC(C=1.0, random_state=0).fit(Xtr, yy[tr]).predict(Xte)


def per_model(yt, yp, labels):
    out = {}
    for c in labels:
        tp = ((yt == c) & (yp == c)).sum(); fn = ((yt == c) & (yp != c)).sum(); fp = ((yt != c) & (yp == c)).sum(); tn = ((yt != c) & (yp != c)).sum()
        rec = tp / (tp + fn) if tp + fn else float('nan'); spec = tn / (tn + fp) if tn + fp else float('nan')
        out[MODELS[c]] = {'n': int(tp + fn), 'predicted': int(tp + fp), 'recall': float(rec),
                          'precision': float(tp / (tp + fp)) if tp + fp else float('nan'), 'balanced_acc_ovr': float((rec + spec) / 2)}
    return out


def macro_bal(yt, yp):
    return float(np.mean([((yt == c) & (yp == c)).sum() / (yt == c).sum() for c in np.unique(yt)]))


for vname, fn in VIEWS.items():
    T = [fn(d) for d in honest]
    # ---- A ----
    pred = np.empty(len(T), int)
    for tr, te in GroupKFold(5).split(T, y, g_task):
        assert set(g_task[tr]).isdisjoint(g_task[te]); pred[te] = fitpred(T, tr, te, y)
    pm = per_model(y, pred, range(len(MODELS)))
    mb = macro_bal(y, pred); bs = []
    for _ in range(1000):
        ii = np.concatenate([idx[t] for t in rng.choice(tasks, len(tasks))]); bs.append(macro_bal(y[ii], pred[ii]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # ---- C(i) within-vendor share of errors ----
    err = pred != y; same_v = np.array([VENDOR[MODELS[a]] == VENDOR[MODELS[b]] for a, b in zip(y, pred)])
    res[vname] = {'acc': float((pred == y).mean()), 'macro_balanced_acc': mb, 'macro_balanced_acc_ci95': [float(lo), float(hi)],
                  'per_model': pm, 'errors': int(err.sum()), 'errors_within_vendor': int((err & same_v).sum())}
    log.append(f"\n## {vname}: acc {res[vname]['acc']:.3f}; macro balanced acc {mb:.3f} [{lo:.3f},{hi:.3f}]; "
               f"errors {err.sum()} of which within-vendor {(err & same_v).sum()} ({(err & same_v).sum()/max(err.sum(),1):.0%})")
    log.append('  per model (n / predicted / recall / precision / one-vs-rest balanced acc): ' + '; '.join(
        f"{m} {v['n']}/{v['predicted']}/{v['recall']:.2f}/{v['precision']:.2f}/{v['balanced_acc_ovr']:.3f}" for m, v in pm.items()))
    # ---- B ----
    rows, allp, ally = [], [], []
    for tr, te in LeaveOneGroupOut().split(T, y, g_env):
        assert set(g_env[tr]).isdisjoint(g_env[te]) and set(g_task[tr]).isdisjoint(g_task[te]), 'environment leakage'
        p = fitpred(T, tr, te, y); yt = y[te]; allp.append(p); ally.append(yt)
        cnt = collections.Counter(yt)
        rows.append({'env': g_env[te][0], 'n': int(len(te)), 'tasks': int(len(set(g_task[te]))), 'models_covered': len(cnt),
                     'majority_baseline': float(max(cnt.values()) / len(te)), 'uniform_chance': 1 / len(cnt), 'acc': float((p == yt).mean())})
    allp, ally = np.concatenate(allp), np.concatenate(ally)
    res[vname]['loeo'] = {'per_env': rows, 'pooled_acc': float((allp == ally).mean()), 'pooled_macro_balanced_acc': macro_bal(ally, allp)}
    log.append(f"  leave-one-environment-out pooled acc {res[vname]['loeo']['pooled_acc']:.3f}, pooled macro balanced {res[vname]['loeo']['pooled_macro_balanced_acc']:.3f}")
    log.append('  per env (n trajectories / tasks / models / majority baseline / acc): ' + '; '.join(
        f"{r['env']} {r['n']}/{r['tasks']}/{r['models_covered']}/{r['majority_baseline']:.2f}/{r['acc']:.3f}" for r in rows))
    # ---- C(ii) within-vendor classifiers ----
    res[vname]['within_vendor'] = {}
    for v in ('anthropic', 'google', 'openai'):
        sel = np.array([VENDOR[MODELS[c]] == v for c in y]); ii = np.where(sel)[0]
        Tv = [T[i] for i in ii]; yv = y[ii]; gv = g_task[ii]; pv = np.empty(len(ii), int)
        for tr, te in GroupKFold(5).split(Tv, yv, gv):
            assert set(gv[tr]).isdisjoint(gv[te]); pv[te] = fitpred(Tv, tr, te, yv)
        k = len(set(yv)); tm = collections.Counter(tmpl[ii])
        res[vname]['within_vendor'][v] = {'models': sorted({MODELS[c] for c in yv}), 'n': int(len(ii)), 'acc': float((pv == yv).mean()),
                                          'chance': 1 / k, 'macro_balanced_acc': macro_bal(yv, pv), 'templates': dict(tm)}
        log.append(f"  within-vendor {v} ({k}-way, chance {1/k:.2f}): acc {(pv == yv).mean():.3f}, macro balanced {macro_bal(yv, pv):.3f}; template mix {dict(tm)}")
    print('\n'.join(log[-6:]) + f'\n  t={time.time()-t0:.0f}s', flush=True)
json.dump(res, open(OUT + '.json', 'w'), indent=1); open(OUT + '.txt', 'w').write('\n'.join(log) + '\n'); print(f'done in {time.time()-t0:.0f}s')
