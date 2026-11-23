"""Is the O4 retention proxy (attack-vs-honest AUROC, task-held-out) inflated by task differences?

O4 as planned compares attack runs (92 task pairs on 67 main tasks) with honest runs (200 main tasks); only 13 main
tasks occur in both conditions, so a classifier can partly separate the conditions by task. Two cohorts, all 11 pipelines:
  MATCHED  honest and attack runs restricted to the 13 shared main tasks (142 honest, 187 attack)
  CONTROL  same number of main-task groups per condition, but with NO overlap: 13 honest groups drawn from honest-only
           tasks and 13 attack groups drawn from attack tasks outside the shared set (5 seeded draws)
Identical transforms, featuriser, classifier and GroupKFold(5) by (env, main task) as src/identification.py.
MATCHED is scored as in src/identification.py (pooled out-of-fold scores) with a task-cluster bootstrap 95% interval
(B = 1000, conditional on the fitted models), and also foldwise. Pooling raw SVM scores across folds with different class
prevalence can give AUROC far from 0.5 even for uninformative inputs, so the CONTROL is scored only foldwise:
StratifiedGroupKFold(5) by (env, main task), so every test fold holds both classes; AUROC within each fold, averaged with
weights equal to the fold's number of positive-negative pairs. A null check with identical features for every run is
reported and must give 0.500.
Usage: python -I src/retention_matched.py data/text results/retention_matched
"""
import json, os, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

TEXT, OUT = sys.argv[1], sys.argv[2]
ID = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'identification.py')
ns = {'__file__': ID}; _a = sys.argv; sys.argv = ['x', TEXT, '/dev/null']
exec(open(ID).read().split("\nhonest = [d for d in docs")[0], ns); sys.argv = _a
docs, TRANSFORMS, featurize, clf = ns['docs'], ns['TRANSFORMS'], ns['featurize'], ns['clf']
grp = lambda d: f"{d['env']}/{d['main_task']}"
H = {grp(d) for d in docs if d['label'] == 'honest'}; A = {grp(d) for d in docs if d['label'] == 'attack'}
S = H & A
rng = np.random.default_rng(0)


def oof_scores(sel, fn, stratified=False):
    t = [fn(d) for d in sel]; yb = np.array([d['label'] == 'attack' for d in sel]); gb = np.array([grp(d) for d in sel])
    sb = np.zeros(len(sel)); fold = np.full(len(sel), -1)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0) if stratified else GroupKFold(5)
    for k, (tr, te) in enumerate(cv.split(t, yb, gb)):
        assert set(gb[tr]).isdisjoint(gb[te]), 'task leakage across folds'
        Xtr, Xte = featurize([t[i] for i in tr], [t[i] for i in te])
        sb[te] = clf().fit(Xtr, yb[tr]).decision_function(Xte); fold[te] = k
    return yb, gb, sb, fold


def foldwise_auc(yb, sb, fold):
    num = den = 0.0; skipped = 0
    for k in np.unique(fold):
        m = fold == k; p = int(yb[m].sum()); n = int(m.sum()) - p
        if p == 0 or n == 0:
            skipped += 1; continue
        num += roc_auc_score(yb[m], sb[m]) * p * n; den += p * n
    return float(num / den), skipped


def auc_ci(yb, gb, sb, fold=None):
    tasks = np.unique(gb); idx = {k: np.where(gb == k)[0] for k in tasks}; bs = []
    for _ in range(1000):
        ii = np.concatenate([idx[k] for k in rng.choice(tasks, len(tasks))])
        if len(set(yb[ii])) == 2:
            bs.append(roc_auc_score(yb[ii], sb[ii]))
    return float(roc_auc_score(yb, sb)), [float(x) for x in np.percentile(bs, [2.5, 97.5])]


matched = [d for d in docs if grp(d) in S]
draws = []
for k in range(5):
    r = np.random.default_rng(k)
    hg = set(r.choice(sorted(H - A), len(S), replace=False)); ag = set(r.choice(sorted(A - S), len(S), replace=False))
    draws.append([d for d in docs if (d['label'] == 'honest' and grp(d) in hg) or (d['label'] == 'attack' and grp(d) in ag)])
res0 = {'shared_groups': len(S), 'matched': {'honest': sum(d['label'] == 'honest' for d in matched), 'attack': sum(d['label'] == 'attack' for d in matched)},
       'control_sizes': [{'honest': sum(d['label'] == 'honest' for d in c), 'attack': sum(d['label'] == 'attack' for d in c)} for c in draws], 'views': {}}
res = res0
log = [f"shared main-task groups {len(S)}; MATCHED cohort {res['matched']}",
       f"CONTROL cohorts (5 draws): {len(S)} honest-only and {len(S)} attack-only main-task groups each (group counts matched; run counts {res['control_sizes']})",
       "CONTROL scoring: StratifiedGroupKFold(5), AUROC within each fold, weighted by positive-negative pairs (pooled scores are not comparable across folds)"]
const = lambda d: 'x'
nulls = [foldwise_auc(*[v for i, v in enumerate(oof_scores(x, const, True)) if i != 1])[0] for x in draws]
res['null_identical_features_foldwise'] = nulls
log.append(f"null check (identical features for every run), CONTROL foldwise AUROC per draw: {[round(v, 3) for v in nulls]}")
print(log[-1], flush=True)
t0 = time.time()
for name, fn_ in TRANSFORMS.items():
    yb, gb, sb, fd = oof_scores(matched, fn_)
    m_auc, m_ci = auc_ci(yb, gb, sb); m_fw, m_sk = foldwise_auc(yb, sb, fd)
    c = []
    for x in draws:
        yc, _, sc, fc = oof_scores(x, fn_, True); a, sk = foldwise_auc(yc, sc, fc); assert sk == 0; c.append(a)
    res['views'][name] = {'matched_auc_pooled': m_auc, 'matched_ci95': m_ci, 'matched_auc_foldwise': m_fw, 'matched_folds_skipped': m_sk,
                          'control_aucs_foldwise': c, 'control_mean_foldwise': float(np.mean(c))}
    log.append(f"{name:<16} MATCHED pooled {m_auc:.3f} [{m_ci[0]:.3f}, {m_ci[1]:.3f}] foldwise {m_fw:.3f}   CONTROL foldwise mean {np.mean(c):.3f} (range {min(c):.3f}-{max(c):.3f})   t={time.time()-t0:.0f}s")
    print(log[-1], flush=True)
json.dump(res, open(OUT + '.json', 'w'), indent=1); open(OUT + '.txt', 'w').write('\n'.join(log) + '\n')
