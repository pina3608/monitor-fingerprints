"""Sensitivity check for the O4 retention proxy (attack-vs-honest AUROC, task-held-out).

src/identification.py computes O4 by pooling raw out-of-fold SVM scores across the five GroupKFold folds. When the
attack share differs between folds, each fold's model has a different offset, and pooled AUROC can then move away
from what the per-fold rankings support (it can fall well below 0.5 for inputs with no signal). This script reruns
exactly the same folds, transforms, featuriser and classifier for the four O4 cohorts of identification.py and
reports, side by side:
  pooled    AUROC of the pooled out-of-fold scores (must reproduce results/identification.json exactly)
  foldwise  AUROC within each test fold that contains both classes, averaged with weights equal to the fold's number
            of attack-honest pairs (folds with one class are skipped and counted)
Usage: python -I src/o4_foldwise.py data/text results/o4_foldwise
"""
import json, os, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

TEXT, OUT = sys.argv[1], sys.argv[2]
ID = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'identification.py')
ns = {'__file__': ID}; _a = sys.argv; sys.argv = ['x', TEXT, '/dev/null']
exec(open(ID).read().split("\nfor name, fn in TRANSFORMS.items():")[0], ns); sys.argv = _a
docs, TRANSFORMS, featurize, clf = ns['docs'], ns['TRANSFORMS'], ns['featurize'], ns['clf']
VIABLE, shared_main = ns['VIABLE'], ns['shared_main']
grp = lambda d: f"{d['env']}/{d['main_task']}"
COHORTS = {
    'as_run_8': lambda d: d['model'] in VIABLE,
    'all_11': lambda d: True,
    'successful_attacks_all_11': lambda d: d['label'] == 'honest' or bool(d.get('side_task_success')),
    'matched_13_tasks_all_11': lambda d: grp(d) in shared_main,
}


def o4_both(fn, keep):
    both = [d for d in docs if keep(d)]
    tb = [fn(d) for d in both]; yb = np.array([d['label'] == 'attack' for d in both]); gb = np.array([grp(d) for d in both])
    sb = np.zeros(len(both)); num = den = 0.0; skipped = 0; shares = []
    for tr, te in GroupKFold(5).split(tb, yb, gb):
        assert set(gb[tr]).isdisjoint(gb[te]), 'task leakage across folds'
        Xtr, Xte = featurize([tb[i] for i in tr], [tb[i] for i in te])
        sb[te] = clf().fit(Xtr, yb[tr]).decision_function(Xte)
        p = int(yb[te].sum()); n = len(te) - p; shares.append(round(p / len(te), 3))
        if p == 0 or n == 0:
            skipped += 1; continue
        num += roc_auc_score(yb[te], sb[te]) * p * n; den += p * n
    return float(roc_auc_score(yb, sb)), float(num / den), skipped, shares


res, log, t0 = {}, [], time.time()
for name, fn in TRANSFORMS.items():
    res[name] = {}
    for c, keep in COHORTS.items():
        pooled, fw, sk, shares = o4_both(fn, keep)
        res[name][c] = {'pooled': pooled, 'foldwise': fw, 'folds_skipped': sk, 'test_fold_attack_share': shares}
        log.append(f"{name:<16} {c:<26} pooled {pooled:.3f}  foldwise {fw:.3f}  (folds skipped {sk}; attack share per test fold {shares})  t={time.time()-t0:.0f}s")
        print(log[-1], flush=True)
json.dump(res, open(OUT + '.json', 'w'), indent=1); open(OUT + '.txt', 'w').write('\n'.join(log) + '\n')
