"""Classifier pairwise own-vs-other accuracy on task-matched pairs.

For each model M and each task, pair M's trajectory with model X's trajectory on the same task; the classifier "picks"
the one with the higher (task-held-out) M-score. Reported per M for: honest pairs (5-fold GroupKFold by task),
and attack pairs (classifier trained on honest runs of tasks not in the attack set). Distractors split into
same-vendor and cross-vendor. Text views: J1 (tool calls) and J4 (action skeleton), as in identification.py.
Usage: python -I src/pairwise.py data/text results/pairwise
"""
import collections, glob, gzip, json, os, sys
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
sys.argv, _argv = sys.argv[:1] + ['x', '/dev/null'], sys.argv  # identification.py reads argv at import; neutralise
import importlib.util
spec = importlib.util.spec_from_file_location('identification', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'identification.py'))
src = open(spec.origin).read().split("honest = [d for d in docs")[0]  # take definitions only (transforms, featurize, clf)
src = src.replace("docs = [json.loads(l) for f in sorted(glob.glob(f'{TEXT}/*.jsonl.gz')) for l in gzip.open(f, 'rt')]", "docs = []")
src = src.replace("docs = [d for d in docs if not (d['model'] == 'Kimi' and d['agent_model'] == 'kimi-k2-thinking')]", "")
ns = {}; exec(src, ns)
sys.argv = _argv
TEXT, OUT = sys.argv[1], sys.argv[2]
MODELS, VENDOR, TR, featurize = ns['MODELS'], ns['VENDOR'], ns['TRANSFORMS'], ns['featurize']
docs = [json.loads(l) for f in sorted(glob.glob(f'{TEXT}/*.jsonl.gz')) for l in gzip.open(f, 'rt')]
docs = [d for d in docs if not (d['model'] == 'Kimi' and d['agent_model'] == 'kimi-k2-thinking')]
honest = [d for d in docs if d['label'] == 'honest']; attack = [d for d in docs if d['label'] == 'attack']
key_h = lambda d: (d['env'], d['main_task']); key_a = lambda d: (d['env'], d['main_task'], d['side_task'])
yh = np.array([MODELS.index(d['model']) for d in honest]); gh = np.array(['/'.join(key_h(d)) for d in honest])
attack_main = {key_h(d) for d in attack}
res, log = {}, []
for view in ('J4_skeleton', 'J1_calls'):
    fn = TR[view]; th = [fn(d) for d in honest]
    S = np.zeros((len(honest), len(MODELS)))
    for tr, te in GroupKFold(5).split(th, yh, gh):
        assert set(gh[tr]).isdisjoint(gh[te]), 'task leakage across folds'
        Xtr, Xte = featurize([th[i] for i in tr], [th[i] for i in te]); S[te] = LinearSVC(C=1.0, random_state=0).fit(Xtr, yh[tr]).decision_function(Xte)
    tr = [i for i, d in enumerate(honest) if key_h(d) not in attack_main]
    Xtr, Xte = featurize([th[i] for i in tr], [fn(d) for d in attack]); SA = LinearSVC(C=1.0, random_state=0).fit(Xtr, yh[tr]).decision_function(Xte)
    for lab, D, SC, kf in (('honest', honest, S, key_h), ('attack', attack, SA, key_a)):
        by = collections.defaultdict(dict)
        for i, d in enumerate(D):
            by[kf(d)][d['model']] = i
        for mi, M in enumerate(MODELS):
            acc = collections.defaultdict(list)
            for k, mm in by.items():
                if M not in mm:
                    continue
                for X, j in mm.items():
                    if X == M:
                        continue
                    kind = 'same_vendor' if VENDOR[X] == VENDOR[M] else 'cross_vendor'
                    acc[kind].append(float(SC[mm[M], mi] > SC[j, mi]) + 0.5 * float(SC[mm[M], mi] == SC[j, mi]))
            res[f'{view}|{lab}|{M}'] = {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}
            log.append(f"{view:<12} {lab:<6} {M:<13} " + '  '.join(f'{k}={np.mean(v):.3f} (n={len(v)})' for k, v in sorted(acc.items())))
        print('\n'.join(log[-len(MODELS):]), flush=True)
json.dump(res, open(OUT + '.json', 'w'), indent=1); open(OUT + '.txt', 'w').write('\n'.join(log) + '\n')
