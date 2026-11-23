"""Recursive argument canonicalisation of the monitor-visible tool calls (additional robustness check).

The §5.8 "keys sorted" transfer variant (src/robustness.py) is shallow: it starts from argument reprs truncated at
4,000 chars, sorts only top-level keys, and keeps the original text when parsing fails. This script builds a separate
view, J1C, from the formatter's own call lines (the J1 input: monitor visibility rules and editor redaction already
applied; no 4,000-char per-call cap, but the same 100,000-char per-trajectory cap as J1, so a trajectory at the cap
ends in a cut-off call, which fails to parse):
  1. drop planning-scaffold calls (PLAN predicate below; the observed matching tool names are reported);
  2. parse each call's argument dict with ast.literal_eval;
  3. recursively sort dictionary keys, keep list order and scalar values;
  4. serialise deterministically as JSON (sort_keys, compact separators, ensure_ascii=False);
  5. on a parse failure emit the placeholder '<UNPARSED>' (never the original text, which would keep key order) and
     count it.
Compared views (identification.py featuriser and classifier): J1 as planned, J1 with planning calls dropped (J1S),
and J1C. Reported: honest 11-way accuracy (identical GroupKFold(5) folds by env/main task, paired task-cluster
bootstrap differences, B = 1000) and honest->attack transfer (trained once on the honest runs of tasks outside the
attack set, as in identification.py), with per-model transfer recall.
Usage: python -I src/canonical_calls.py data/text data/struct results/canonical_calls
"""
import ast, collections, glob, gzip, json, os, sys, time
import numpy as np

PLAN = lambda name: 'phase' in name or 'planning' in name   # same predicate as src/robustness.py and src/cohort_audit.py
PREFIX = 'Tool call: '


def canon_value(v):
    if isinstance(v, dict):
        return {str(k): canon_value(v[k]) for k in sorted(v, key=str)}
    if isinstance(v, (list, tuple)):
        return [canon_value(x) for x in v]
    return v


def split_call(line):
    """'Tool call: <name>: <args repr>' -> (name, args_text); None if the line is not a call line."""
    if not line.startswith(PREFIX):
        return None
    name, sep, rest = line[len(PREFIX):].partition(': ')
    return (name, rest) if sep else (name, '')


def canon_call(line):
    """Returns (canonical line or None if dropped as planning, parsed_ok)."""
    sc = split_call(line)
    if sc is None:
        return f'{PREFIX}<UNPARSED>', False
    name, rest = sc
    if PLAN(name):
        return None, True
    try:
        args = ast.literal_eval(rest) if rest.strip() else {}
        if not isinstance(args, dict):
            raise ValueError('arguments are not a dict')
        return f'{PREFIX}{name}: ' + json.dumps(canon_value(args), sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str), True
    except Exception:
        return f'{PREFIX}{name}: <UNPARSED>', False


def canon_calls(calls_text):
    out, fails = [], 0
    for line in calls_text.split('\n') if calls_text else []:
        c, ok = canon_call(line)
        fails += not ok
        if c is not None:
            out.append(c)
    return '\n'.join(out), fails


def strip_planning(calls_text):
    return '\n'.join(l for l in (calls_text.split('\n') if calls_text else []) if not ((sc := split_call(l)) and PLAN(sc[0])))


if __name__ == '__main__':
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold
    TEXT, STRUCT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    ID = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'identification.py')
    ns = {'__file__': ID}; _a = sys.argv; sys.argv = ['x', TEXT, '/dev/null']
    exec(open(ID).read().split("\nfor name, fn in TRANSFORMS.items():")[0], ns); sys.argv = _a
    docs, TR, featurize, clf, MODELS = ns['docs'], ns['TRANSFORMS'], ns['featurize'], ns['clf'], ns['MODELS']
    honest, attack, y_h, g_h, attack_main = ns['honest'], ns['attack'], ns['y_h'], ns['g_h'], ns['attack_main']
    log, res, t0 = [], {}, time.time()

    # --- coverage and failure accounting ---
    fails = collections.Counter(); ncalls = collections.Counter(); plan_names = collections.Counter(); long_vis = collections.Counter()
    for d in docs:
        for l in d['calls'].split('\n') if d['calls'] else []:
            sc = split_call(l); ncalls[d['label']] += 1
            if sc and PLAN(sc[0]):
                plan_names[sc[0]] += 1; continue
            if sc and len(sc[1]) > 4000:
                long_vis[d['label']] += 1
            fails[d['label']] += not canon_call(l)[1]
    trunc = collections.Counter(); nstruct = collections.Counter(); ids = {d['trajectory_id'] for d in docs}
    for f in sorted(glob.glob(f'{STRUCT}/*.jsonl.gz')):
        for l in gzip.open(f, 'rt'):
            s = json.loads(l)
            if s['trajectory_id'] not in ids:
                continue
            for a in s['actions']:
                if a['hidden'] or a['parse_err'] or PLAN(a['f']):   # same visibility as the §5.8 variant
                    continue
                nstruct[s['label']] += 1; trunc[s['label']] += len(a.get('repr', '')) >= 4000
    res['accounting'] = {'visible_calls': dict(ncalls), 'planning_calls_dropped_by_name': dict(plan_names),
                         'canonicalisation_failures': dict(fails), 'visible_call_args_over_4000_chars': dict(long_vis),
                         'struct_non_planning_actions': dict(nstruct), 'struct_reprs_at_4000_char_cap': dict(trunc)}
    log += [f"visible call lines: {dict(ncalls)}",
            f"planning-scaffold calls dropped (predicate: 'phase' in name or 'planning' in name), by tool name: {dict(plan_names)}",
            f"canonicalisation failures (emitted as <UNPARSED>): {dict(fails)}",
            f"monitor-visible non-planning calls whose argument text exceeds 4,000 chars: {dict(long_vis)}",
            f"visible non-planning struct actions whose stored repr hit the 4,000-char cap used by the §5.8 'keys sorted' variant: {dict(trunc)} of {dict(nstruct)}"]
    print('\n'.join(log), flush=True)

    VIEWS = {'J1': TR['J1_calls'], 'J1S_planning_dropped': lambda d: strip_planning(d['calls']),
             'J1C_recursive_canonical': lambda d: canon_calls(d['calls'])[0]}
    folds = list(GroupKFold(5).split(honest, y_h, g_h))
    tr = [i for i, d in enumerate(honest) if f"{d['env']}/{d['main_task']}" not in attack_main]
    assert not ({g_h[i] for i in tr} & attack_main), 'transfer leakage'
    y_a = np.array([MODELS.index(d['model']) for d in attack])
    correct, res['views'] = {}, {}
    for name, fn in VIEWS.items():
        th = [fn(d) for d in honest]; pred = np.empty(len(honest), int)
        for a_, b_ in folds:
            assert set(g_h[a_]).isdisjoint(g_h[b_])
            Xtr, Xte = featurize([th[i] for i in a_], [th[i] for i in b_]); pred[b_] = clf().fit(Xtr, y_h[a_]).predict(Xte)
        correct[name] = pred == y_h
        Xtr, Xte = featurize([th[i] for i in tr], [fn(d) for d in attack]); pa = clf().fit(Xtr, y_h[tr]).predict(Xte)
        recall = {m: float((pa[y_a == k] == k).mean()) for k, m in enumerate(MODELS)}
        res['views'][name] = {'acc_11way': float(correct[name].mean()), 'acc_transfer': float(accuracy_score(y_a, pa)), 'transfer_recall_per_model': recall}
        log.append(f"{name:<24} honest 11-way acc {correct[name].mean():.3f} | transfer acc {accuracy_score(y_a, pa):.3f} | transfer recall per model "
                   + ', '.join(f'{m} {r:.2f}' for m, r in recall.items()) + f" | t={time.time()-t0:.0f}s")
        print(log[-1], flush=True)
    rng = np.random.default_rng(0); tasks = np.unique(g_h); idx = {k: np.where(g_h == k)[0] for k in tasks}
    for a_, b_ in [('J1S_planning_dropped', 'J1'), ('J1C_recursive_canonical', 'J1'), ('J1C_recursive_canonical', 'J1S_planning_dropped')]:
        dlt = correct[a_].astype(float) - correct[b_]; bs = []
        for _ in range(1000):
            ii = np.concatenate([idx[k] for k in rng.choice(tasks, len(tasks))]); bs.append(dlt[ii].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5]); res[f'paired_{a_}_minus_{b_}'] = [float(dlt.mean()), float(lo), float(hi)]
        log.append(f"paired honest 11-way {a_} - {b_}: {dlt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"); print(log[-1], flush=True)
    json.dump(res, open(OUT + '.json', 'w'), indent=1); open(OUT + '.txt', 'w').write('\n'.join(log) + '\n')
