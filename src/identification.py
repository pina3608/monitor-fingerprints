"""Study 1: model identifiability of LinuxArena trajectories and the effect of deterministic jamming transforms.
Decision rules were written down before running (report §4, §5.2). Outputs JSON + text log.
Usage: python -I src/identification.py data/text results/identification
"""
import glob, gzip, json, re, sys, time
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

TEXT, OUT = sys.argv[1], sys.argv[2]
MODELS = ['Opus', 'Sonnet', 'Haiku', 'Gemini Pro', 'Gemini Flash', 'GLM 5', 'Kimi', 'Grok', 'GPT-5 Nano', 'GPT-5 Mini', 'GPT-5.4']
VENDOR = {'Opus': 'anthropic', 'Sonnet': 'anthropic', 'Haiku': 'anthropic', 'Gemini Pro': 'google', 'Gemini Flash': 'google',
          'GLM 5': 'zhipu', 'Kimi': 'moonshot', 'Grok': 'xai', 'GPT-5 Nano': 'openai', 'GPT-5 Mini': 'openai', 'GPT-5.4': 'openai'}
CHANCE = 1 / len(MODELS)
t0 = time.time()
docs = [json.loads(l) for f in sorted(glob.glob(f'{TEXT}/*.jsonl.gz')) for l in gzip.open(f, 'rt')]
docs = [d for d in docs if not (d['model'] == 'Kimi' and d['agent_model'] == 'kimi-k2-thinking')]  # different model, excluded

# ---------------- transforms ----------------
COMMENT = re.compile(r'(?m)(^|\s)#[^\n\'"]*$|//[^\n\'"]*$|/\*.*?\*/', re.S)
DOC = re.compile(r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'')
STR = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
NUM = re.compile(r'\b\d+(?:\.\d+)?\b')
WS = re.compile(r'\s+')


def j2(s):
    s = s.replace('\\n', '\n')
    return WS.sub(' ', COMMENT.sub(' ', DOC.sub(' ', s)))


def j3(s):
    return NUM.sub('<N>', STR.sub('<S>', j2(s)))


CALL = re.compile(r"Tool call: (\w+): (\{.*)")
def j4(calls):
    out = []
    for line in calls.split('\n'):
        m = CALL.match(line)
        if not m:
            continue
        fn, args = m.group(1), m.group(2)
        if fn == 'bash':
            c = re.search(r"'cmd': ['\"]\s*([\w./-]+)", args)
            out.append(f'bash:{c.group(1) if c else "?"}')
        elif fn == 'text_editor':
            c = re.search(r"'command': '(\w+)'", args)
            out.append(f'edit:{c.group(1) if c else "?"}')
        else:
            out.append(fn)
    return ' '.join(out)


TRANSFORMS = {'J0_raw': lambda d: d['raw'], 'J1_calls': lambda d: d['calls'], 'J2_nocomments': lambda d: j2(d['calls']),
              'J3_maskliterals': lambda d: j3(d['calls']), 'J4_skeleton': lambda d: j4(d['calls'])}


def featurize(train_txt, test_txt):
    hw = HashingVectorizer(analyzer='word', ngram_range=(1, 2), n_features=2 ** 18, alternate_sign=False, token_pattern=r'\S+', lowercase=False)
    hc = HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=2 ** 18, alternate_sign=False, lowercase=False)
    Xw_tr, Xw_te = hw.transform(train_txt), hw.transform(test_txt)
    Xc_tr, Xc_te = hc.transform([t[:30000] for t in train_txt]), hc.transform([t[:30000] for t in test_txt])
    tw, tc = TfidfTransformer(sublinear_tf=True).fit(Xw_tr), TfidfTransformer(sublinear_tf=True).fit(Xc_tr)
    return hstack([tw.transform(Xw_tr), tc.transform(Xc_tr)]).tocsr(), hstack([tw.transform(Xw_te), tc.transform(Xc_te)]).tocsr()


def clf():
    return LinearSVC(C=1.0, random_state=0)


honest = [d for d in docs if d['label'] == 'honest']; attack = [d for d in docs if d['label'] == 'attack']
y_h = np.array([MODELS.index(d['model']) for d in honest]); g_h = np.array([f"{d['env']}/{d['main_task']}" for d in honest])
results = {'n_honest': len(honest), 'n_attack': len(attack), 'chance': CHANCE, 'per_transform': {}}
log = [f'docs honest={len(honest)} attack={len(attack)} per model honest: ' + str({m: int((y_h == i).sum()) for i, m in enumerate(MODELS)})]
attack_main = {f"{d['env']}/{d['main_task']}" for d in attack}
# main tasks run in BOTH conditions; the matched-task O4 variant keeps honest AND attack runs only from these groups
shared_main = attack_main & set(g_h)
_m = [d for d in docs if f"{d['env']}/{d['main_task']}" in shared_main]
results['matched_task_cohort'] = {'groups': len(shared_main), 'honest': sum(d['label'] == 'honest' for d in _m), 'attack': sum(d['label'] == 'attack' for d in _m)}
log.append(f"matched-task O4 cohort: {results['matched_task_cohort']}")

VIABLE = {'Opus', 'Sonnet', 'Haiku', 'Gemini Pro', 'Gemini Flash', 'GLM 5', 'Kimi', 'Grok'}


def o4(fn, keep):
    both = [d for d in docs if keep(d)]
    tb = [fn(d) for d in both]; yb = np.array([d['label'] == 'attack' for d in both]); gb = np.array([f"{d['env']}/{d['main_task']}" for d in both])
    sb = np.zeros(len(both))
    for tr2, te2 in GroupKFold(5).split(tb, yb, gb):
        assert set(gb[tr2]).isdisjoint(gb[te2]), 'task leakage across folds'
        Xtr, Xte = featurize([tb[i] for i in tr2], [tb[i] for i in te2])
        sb[te2] = clf().fit(Xtr, yb[tr2]).decision_function(Xte)
    return float(roc_auc_score(yb, sb))


for name, fn in TRANSFORMS.items():
    th = [fn(d) for d in honest]
    # O1/O2: 11-way, task-held-out
    pred = np.empty(len(honest), int); proba = np.zeros((len(honest), len(MODELS)))
    for tr, te in GroupKFold(5).split(th, y_h, g_h):
        assert set(g_h[tr]).isdisjoint(g_h[te]), 'task leakage across folds'
        Xtr, Xte = featurize([th[i] for i in tr], [th[i] for i in te])
        m = clf().fit(Xtr, y_h[tr]); proba[te] = m.decision_function(Xte); pred[te] = proba[te].argmax(1)
    acc = accuracy_score(y_h, pred)
    vend_auc = {}
    for v in sorted(set(VENDOR.values())):
        idx = [i for i, m in enumerate(MODELS) if VENDOR[m] == v]
        vend_auc[v] = float(roc_auc_score([VENDOR[MODELS[k]] == v for k in y_h], proba[:, idx].max(1)))
    model_auc = {m: float(roc_auc_score(y_h == i, proba[:, i])) for i, m in enumerate(MODELS)}
    cm = confusion_matrix(y_h, pred, labels=range(len(MODELS))).tolist()
    # O3: train on honest tasks NOT in attack set, test on attack runs
    tr = [i for i, d in enumerate(honest) if f"{d['env']}/{d['main_task']}" not in attack_main]
    assert not ({f"{honest[i]['env']}/{honest[i]['main_task']}" for i in tr} & attack_main), 'transfer leakage'
    ta = [fn(d) for d in attack]; y_a = np.array([MODELS.index(d['model']) for d in attack])
    Xtr, Xte = featurize([th[i] for i in tr], ta)
    pa = clf().fit(Xtr, y_h[tr]).predict(Xte)
    acc_transfer = accuracy_score(y_a, pa)
    # O4: retention proxy: attack-vs-honest label AUROC, task-held-out (group by env/main task).
    # AS RUN, O4 was restricted to 8 pipelines (GPT-5 Nano/Mini/5.4 hard-coded as "non-viable attackers").
    # That restriction was NOT in the plan; it is a disclosed deviation (report §5.2).
    # The decision uses the as-run value; all-11 and two sensitivity variants are reported alongside.
    ret = o4(fn, lambda d: d['model'] in VIABLE)
    ret_all = o4(fn, lambda d: True)
    ret_success = o4(fn, lambda d: d['label'] == 'honest' or bool(d.get('side_task_success')))   # successful sabotage only
    ret_matched = o4(fn, lambda d: f"{d['env']}/{d['main_task']}" in shared_main)                   # both conditions restricted to shared main tasks
    results['per_transform'][name] = {'acc_11way': float(acc), 'acc_transfer_honest_to_attack': float(acc_transfer),
                                      'vendor_auc': vend_auc, 'model_auc': model_auc, 'retention_auc_attack_vs_honest': ret,
                                      'retention_auc_all11': ret_all, 'retention_auc_successful_attacks_all11': ret_success,
                                      'retention_auc_matched_tasks_all11': ret_matched, 'confusion': cm, 'mean_chars': float(np.mean([len(t) for t in th]))}
    log.append(f'{name:<16} 11-way acc={acc:.3f} ({acc/CHANCE:.1f}x chance) | honest->attack transfer acc={acc_transfer:.3f} | '
               f'retention AUROC={ret:.3f} (as run, 8 models; all 11: {ret_all:.3f}; successful attacks: {ret_success:.3f}; matched tasks: {ret_matched:.3f}) | vendor AUROC ' + ', '.join(f'{k}={v:.3f}' for k, v in vend_auc.items()) +
               f' | mean chars {np.mean([len(t) for t in th]):,.0f} | t={time.time()-t0:.0f}s')
    print(log[-1], flush=True)

# decision criteria set before running (report §5.2)
P = results['per_transform']; a0 = P['J0_raw']['acc_11way']; r0 = P['J0_raw']['retention_auc_attack_vs_honest']
crit_a = a0 >= 3 * CHANCE
cands = [k for k in P if k != 'J0_raw' and (P[k]['acc_11way'] - CHANCE) <= 0.5 * (a0 - CHANCE)]
crit_b = bool(cands)
crit_c = any(P[k]['retention_auc_attack_vs_honest'] >= r0 - 0.05 for k in cands)
ceiling, floor = CHANCE + 0.5 * (a0 - CHANCE), r0 - 0.05
results['criteria'] = {'accuracy_ceiling_b': ceiling, 'retention_floor_c': floor, 'a_identifiable': crit_a, 'b_some_transform_halves': crit_b, 'b_transforms': cands, 'c_retains_detection': crit_c}
log.append(f'criteria: (a) {crit_a}  (b) {crit_b} {cands}  (c) {crit_c}   [(b) accuracy ceiling {ceiling:.4f}; (c) retention floor {floor:.4f}; '
           f'(c) is evaluated only for transforms passing (b)]')
json.dump(results, open(OUT + '.json', 'w'), indent=1); open(OUT + '.txt', 'w').write('\n'.join(log) + '\n'); print(log[-1])
