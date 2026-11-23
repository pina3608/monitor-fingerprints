"""Identification with the system-prompt template held constant.
Reuses harness_audit.py views/featuriser. Usage: python -I src/within_template.py data/struct results/within_template.txt
"""
import gzip, glob, json, os, sys
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness_audit.py')).read().split("\nhonest = [d for d in docs")[0]
src = src.replace("STRUCT, OUT = sys.argv[1], sys.argv[2]", "STRUCT, OUT = sys.argv[1], ''").replace("log = [f'trajectories: {len(docs)}']", "log=[]")
src = src.split("# ---------------- H1 provenance")[0] + "\n" + src.split("# ---------------- views ----------------")[1]
ns = {'__file__': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness_audit.py')}; exec(src, ns)
docs, VIEWS, featurize = ns['docs'], ns['VIEWS'], ns['featurize']
out = []
for tmpl, label in (('b6a3c5340e67', 'bare-task template'), ('043cbe728ba7', 'full scaffold template')):
    H = [d for d in docs if d['label'] == 'honest' and d['tmpl_hash'] == tmpl]
    counts = {m: sum(d['model'] == m for d in H) for m in sorted({d['model'] for d in H})}
    keep = [m for m, c in counts.items() if c >= 100]
    H = [d for d in H if d['model'] in keep]
    if len(keep) < 2:
        out.append(f'{label}: fewer than 2 models with >=100 runs ({counts})'); continue
    out.append(f'\n## {label} ({tmpl}); honest runs with this template per model: {counts}; models with >=100: {keep}; chance={1/len(keep):.3f}')
    # (a) all runs of the included models under this template; (b) only tasks run under this template by EVERY included model
    tasks_by_model = {m: {(d['env'], d['main_task']) for d in H if d['model'] == m} for m in keep}
    common = set.intersection(*tasks_by_model.values())
    for subset, HH in (('all runs', H), (f'common-task intersection ({len(common)} tasks)', [d for d in H if (d['env'], d['main_task']) in common])):
        y = np.array([keep.index(d['model']) for d in HH]); g = np.array([f"{d['env']}/{d['main_task']}" for d in HH])
        out.append(f'  [{subset}] n={len(HH)}')
        for name in ('H2_wrapper_only', 'H3_norm_verbs', 'H3b_bag_of_verbs'):
            T = [VIEWS[name](d) for d in HH]; pred = np.empty(len(T), int)
            for tr, te in GroupKFold(5).split(T, y, g):
                assert set(g[tr]).isdisjoint(g[te]), 'task leakage across folds'
                Xtr, Xte = featurize([T[i] for i in tr], [T[i] for i in te]); pred[te] = LinearSVC(C=1.0, random_state=0).fit(Xtr, y[tr]).predict(Xte)
            acc = (pred == y).mean()
            out.append(f'  {name:<18} acc={acc:.3f} ({acc*len(keep):.1f}x chance) recall=' + str({m: round(float((pred[y == i] == i).mean()), 2) for i, m in enumerate(keep)}))
            print(out[-1], flush=True)
open(sys.argv[2], 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
