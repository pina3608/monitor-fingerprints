"""Baselines: does the normalised-verb signal exceed harness, length, outcome and template baselines?

Honest runs, 11-way, task-held-out GroupKFold(5) by (env, main task); every view in a comparison uses identical folds.
Views (per visible action; hidden tools and malformed calls removed, as in the monitor view):
  H3          planned "tool names + approximate command verbs" (harness_audit.verb)
  T0          tool names only (all visible tools)
  H3m         H3 with every tool other than bash / text_editor replaced by one placeholder token (positions kept):
              isolates the contribution of environment-specific and scaffold tool names with the H3 parser held fixed
  C1          "common-tool approximate verbs": bash and text_editor calls only; the command is split on ; && || | and
              newlines outside quotes, env assignments and wrapper prefixes (sudo/env/nice/xargs/... with their options,
              timeout) are removed, `cd X` segments are skipped, and the first remaining word's basename is taken;
              text_editor -> edit:<sub-command>. Approximate: no full shell grammar (heredocs, subshells, functions).
  C1P         C1 with every ambiguous bash parse (unbalanced quotes, shlex failure, non-word verb) replaced by bash:?
  T1          common tools only, tool identity only (bash / edit) -- C1 with verbs removed
  H3b_uni     H3 as an unordered bag of unigrams (no n-grams spanning neighbours)
  LEN, OUT, TMPL, LOT   categorical baselines: visible-action count in bins of 10 (>=150 pooled), main-task success,
              system-prompt template hash, and all three combined (one-hot, LinearSVC)
Also: fixed prefixes of the first k common-tool calls (k = 5, 10, 20; runs with >= k such calls), success-stratified
prefixes, J0 vs J1 on identical action prefixes under three featurisers (standard; word-only; char n-grams on call
lines only), and an order test (C1 tokens shuffled within trajectories, everything else fixed).
Paired task-cluster bootstrap (B = 1000) for accuracy differences between views on the same runs and folds.
Usage: python -I src/baselines.py data/text data/struct results/baselines
"""
import collections, glob, gzip, json, os, re, shlex, sys, time
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

TEXT, STRUCT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
HA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness_audit.py')
ns = {'__file__': HA}; _a = sys.argv; sys.argv = ['x', STRUCT, '/dev/null']
exec(open(HA).read().split("\nhonest = [d for d in docs")[0], ns); sys.argv = _a
MODELS, featurize, verb, visible = ns['MODELS'], ns['featurize'], ns['verb'], ns['visible']
honest = [d for d in ns['docs'] if d['label'] == 'honest']
text = {d['trajectory_id']: d for d in (json.loads(l) for f in sorted(glob.glob(f'{TEXT}/*.jsonl.gz')) for l in gzip.open(f, 'rt'))}
assert all(d['trajectory_id'] in text for d in honest)
y = np.array([MODELS.index(d['model']) for d in honest]); g = np.array([f"{d['env']}/{d['main_task']}" for d in honest])
rng = np.random.default_rng(0); t0 = time.time(); res = {'views': {}, 'prefix': {}, 'success_strata': {}, 'j0_j1_prefix': {}, 'paired': {}, 'coverage': {}}
log = [f'honest runs {len(honest)}, task groups {len(set(g))}, chance {1/len(MODELS):.3f}']

# ---------------- common-tool approximate verb ----------------
WRAP_OPTS = {   # wrapper -> options that take a separate value (all other leading -options are skipped alone)
    'sudo': {'-u', '-g', '-C', '-D', '-h', '-p', '-r', '-t', '-U'}, 'env': {'-u', '-C', '-S'}, 'nice': {'-n'},
    'xargs': {'-I', '-n', '-P', '-d', '-L', '-s', '-E', '-a'}, 'stdbuf': {'-i', '-o', '-e'},
    'nohup': set(), 'time': set(), 'command': set(), 'exec': {'-a'}}
ASSIGN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')
stats = collections.Counter()


def split_segments(cmd):
    """Split on ; && || | and newlines OUTSIDE single/double quotes (backslash escapes respected).
    Returns (segments, ambiguous) where ambiguous=True if a quote is left unbalanced."""
    segs, cur, q, i = [], [], None, 0
    while i < len(cmd):
        ch = cmd[i]
        if ch == '\\' and q != "'" and i + 1 < len(cmd):
            cur.append(cmd[i:i + 2]); i += 2; continue
        if q:
            if ch == q:
                q = None
            cur.append(ch); i += 1; continue
        if ch in '\'"':
            q = ch; cur.append(ch); i += 1; continue
        if cmd.startswith('&&', i) or cmd.startswith('||', i):
            segs.append(''.join(cur)); cur = []; i += 2; continue
        if ch in ';|\n':
            segs.append(''.join(cur)); cur = []; i += 1; continue
        cur.append(ch); i += 1
    segs.append(''.join(cur))
    return [x.strip() for x in segs if x.strip()], q is not None


def seg_verb(seg):
    """First command word of one segment, after env assignments and wrapper prefixes. Returns (verb, ambiguous)."""
    try:
        toks = shlex.split(seg, comments=True); amb = False
    except ValueError:
        toks = seg.split(); amb = True
    i = 0
    while i < len(toks):
        t = toks[i]
        if ASSIGN.match(t):
            i += 1
        elif t in WRAP_OPTS:
            i += 1
            while i < len(toks) and toks[i].startswith('-'):
                i += 1 + (toks[i] in WRAP_OPTS[t])
        elif t == 'timeout':
            i += 1
            while i < len(toks) and (toks[i].startswith('-') or re.match(r'^\d', toks[i])):
                i += 1
        else:
            v = os.path.basename(t.strip('()'))
            return v, amb or not re.match(r'^[A-Za-z0-9_.+-]+$', v)
    return None, amb


def c1_tokens(d, placeholder=False):
    """C1 'common-tool approximate verbs'. With placeholder=True, ambiguous parses become bash:? (sensitivity check)."""
    out = []
    for a in visible(d):
        if a['f'] == 'text_editor':
            out.append('edit:' + (a['sub'] or '?')); stats['editor'] += 1 - placeholder
        elif a['f'] == 'bash':
            segs, amb = split_segments((a['cmd'] or '').strip())
            v = None
            for seg in segs:
                v, a2 = seg_verb(seg); amb = amb or a2
                if v and v != 'cd':
                    break
            v = v or ('cd' if any(x.split()[:1] == ['cd'] for x in segs) else None)
            if not placeholder:
                stats['bash'] += 1; stats['bash_ambiguous'] += amb; stats['bash_unmapped'] += v is None
                stats[('amb', d['model'])] += amb; stats[('bash', d['model'])] += 1
            if placeholder and amb:
                out.append('bash:?')
            elif v:
                out.append('bash:' + v)
        elif not placeholder:
            stats['other_tool_dropped'] += 1
    return out


C1 = [c1_tokens(d) for d in honest]
C1P = [c1_tokens(d, placeholder=True) for d in honest]
cov = {}
for m in MODELS:
    D = [d for d in honest if d['model'] == m]
    vis = sum(len(visible(d)) for d in D); kept = sum(len(c) for d, c in zip(honest, C1) if d['model'] == m)
    cov[m] = {'visible_calls': vis, 'common_tool_calls_mapped': kept, 'fraction_kept': kept / vis,
              'bash_calls': stats[('bash', m)], 'bash_ambiguous_fraction': stats[('amb', m)] / max(1, stats[('bash', m)])}
glob_counts = {k: v for k, v in stats.items() if isinstance(k, str)}
res['coverage'] = {'per_model': cov, 'counts': glob_counts}
log.append('C1 coverage (fraction of visible calls kept as common-tool tokens): ' + '; '.join(f"{m} {v['fraction_kept']:.2f}" for m, v in cov.items()))
log.append('C1 ambiguous bash parses (unbalanced quotes, shlex failure, or non-word verb) per bash call: ' + '; '.join(f"{m} {v['bash_ambiguous_fraction']:.3f}" for m, v in cov.items()))
log.append(f'C1 counts: {glob_counts}')


def lenbin(d):
    return f'L{min(len(visible(d)) // 10, 15)}'


CAT = {'LEN': lambda d: {'len': lenbin(d)}, 'OUT': lambda d: {'out': str(bool(d['main_task_success']))},
       'TMPL': lambda d: {'tmpl': d['tmpl_hash']},
       'LOT': lambda d: {'len': lenbin(d), 'out': str(bool(d['main_task_success'])), 'tmpl': d['tmpl_hash']}}
TXT = {'H3': lambda i: ' '.join(verb(a) for a in visible(honest[i])), 'T0': lambda i: ' '.join(a['f'] for a in visible(honest[i])),
       'C1': lambda i: ' '.join(C1[i]), 'T1': lambda i: ' '.join(t.split(':')[0] for t in C1[i]),
       'H3b_uni': lambda i: ' '.join(verb(a) for a in visible(honest[i])),
       # H3 parser unchanged; every tool other than bash / text_editor replaced by one placeholder, positions kept
       'H3m': lambda i: ' '.join(verb(a) if a['f'] in ('bash', 'text_editor') else 'tool:OTHER' for a in visible(honest[i])),
       'C1P': lambda i: ' '.join(C1P[i])}


def uni_featurize(tr, te):
    hv = HashingVectorizer(analyzer='word', ngram_range=(1, 1), n_features=2 ** 18, alternate_sign=False, token_pattern=r'\S+', lowercase=False)
    a, b = hv.transform(tr), hv.transform(te); t = TfidfTransformer(sublinear_tf=True).fit(a)
    return t.transform(a), t.transform(b)


class InsufficientData(Exception):
    pass


def cv(ix, make, kind='text'):
    """Out-of-fold predictions on the runs `ix`, GroupKFold(5) by task within ix (identical folds for identical ix)."""
    ix = np.asarray(ix, dtype=int); yy, gg = y[ix], g[ix]; pred = np.empty(len(ix), int)
    if len(set(yy)) < 2 or len(set(gg)) < 5:
        raise InsufficientData(f'{len(ix)} runs, {len(set(yy))} classes, {len(set(gg))} task groups')
    docs = [make(i) for i in ix]
    for tr, te in GroupKFold(5).split(ix, yy, gg):
        assert set(gg[tr]).isdisjoint(gg[te]), 'task leakage across folds'
        if kind == 'cat':
            dv = DictVectorizer(sparse=False); Xtr = dv.fit_transform([docs[i] for i in tr]); Xte = dv.transform([docs[i] for i in te])
        elif kind == 'uni':
            Xtr, Xte = uni_featurize([docs[i] for i in tr], [docs[i] for i in te])
        else:
            Xtr, Xte = featurize([docs[i] for i in tr], [docs[i] for i in te])
        pred[te] = LinearSVC(C=1.0, random_state=0).fit(Xtr, yy[tr]).predict(Xte)
    return pred == yy


def summ(ok, ix):
    ix = np.asarray(ix); yy = y[ix]
    return {'n': int(len(ix)), 'tasks': int(len(set(g[ix]))), 'models': int(len(set(yy))), 'acc': float(ok.mean()),
            'macro_recall': float(np.mean([ok[yy == c].mean() for c in np.unique(yy)]))}


def paired(ok_a, ok_b, ix):
    ix = np.asarray(ix); gg = g[ix]; tasks = np.unique(gg); idx = {t: np.where(gg == t)[0] for t in tasks}; bs = []
    for _ in range(1000):
        ii = np.concatenate([idx[t] for t in rng.choice(tasks, len(tasks))]); bs.append(ok_a[ii].mean() - ok_b[ii].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5]); return {'diff': float(ok_a.mean() - ok_b.mean()), 'ci95': [float(lo), float(hi)]}


ALL = np.arange(len(honest)); OK = {}
log.append('\n## A. Full honest set (identical folds)')
for name in ('H3', 'H3m', 'T0', 'C1', 'C1P', 'T1', 'H3b_uni', 'LEN', 'OUT', 'TMPL', 'LOT'):
    kind = 'cat' if name in CAT else ('uni' if name == 'H3b_uni' else 'text')
    make = (lambda i, f=CAT[name]: f(honest[i])) if name in CAT else TXT[name]
    OK[name] = cv(ALL, make, kind); res['views'][name] = summ(OK[name], ALL)
    log.append(f"{name:<8} acc {res['views'][name]['acc']:.3f}  macro recall {res['views'][name]['macro_recall']:.3f}  t={time.time()-t0:.0f}s"); print(log[-1], flush=True)
for a, b in (('H3', 'T0'), ('C1', 'T1'), ('H3', 'LOT'), ('C1', 'LOT'), ('H3', 'H3m'), ('C1', 'C1P'), ('H3', 'C1')):
    res['paired'][f'{a}-{b}'] = paired(OK[a], OK[b], ALL)
    log.append(f"paired {a} - {b}: {res['paired'][f'{a}-{b}']['diff']:+.3f} [{res['paired'][f'{a}-{b}']['ci95'][0]:+.3f}, {res['paired'][f'{a}-{b}']['ci95'][1]:+.3f}]")

log.append('\n## B. Fixed prefixes: first k common-tool calls (runs with >= k such calls)')
for k in (5, 10, 20):
    ix = [i for i in ALL if len(C1[i]) >= k]
    a = cv(ix, lambda i, k=k: ' '.join(C1[i][:k])); b = cv(ix, lambda i, k=k: ' '.join(t.split(':')[0] for t in C1[i][:k]))
    cnt = collections.Counter(y[ix]); res['prefix'][k] = {'C1': summ(a, ix), 'T1': summ(b, ix), 'paired_C1-T1': paired(a, b, ix),
                                                          'runs_per_model': {MODELS[c]: int(cnt.get(c, 0)) for c in range(len(MODELS))}}
    log.append(f"k={k:<3} n={len(ix)} tasks={len(set(g[ix]))}  C1 acc {a.mean():.3f}  T1 acc {b.mean():.3f}  diff {res['prefix'][k]['paired_C1-T1']['diff']:+.3f} "
               f"{[round(x, 3) for x in res['prefix'][k]['paired_C1-T1']['ci95']]}  runs/model {res['prefix'][k]['runs_per_model']}"); print(log[-1], flush=True)

log.append('\n## C. Success-stratified, first 10 common-tool calls (main-task outcome of the honest run)')
for lab, val in (('success', True), ('failure', False)):
    ix = [i for i in ALL if len(C1[i]) >= 10 and bool(honest[i]['main_task_success']) == val]
    cnt = collections.Counter(y[ix]); keep = [c for c in cnt if cnt[c] >= 20]; ix = [i for i in ix if y[i] in keep]
    try:
        a = cv(ix, lambda i: ' '.join(C1[i][:10])); b = cv(ix, lambda i: ' '.join(t.split(':')[0] for t in C1[i][:10]))
    except InsufficientData as e:
        res['success_strata'][lab] = {'skipped': str(e)}; log.append(f'{lab:<8} skipped: insufficient data ({e})'); continue
    res['success_strata'][lab] = {'C1': summ(a, ix), 'T1': summ(b, ix), 'models_with_>=20_runs': len(keep), 'chance': 1 / len(keep),
                                  'runs_per_model': {MODELS[c]: int(cnt[c]) for c in sorted(cnt)}}
    log.append(f"{lab:<8} n={len(ix)} models={len(keep)} (chance {1/len(keep):.3f})  C1 acc {a.mean():.3f} (macro recall {res['success_strata'][lab]['C1']['macro_recall']:.3f})  "
               f"T1 acc {b.mean():.3f}  runs/model {res['success_strata'][lab]['runs_per_model']}"); print(log[-1], flush=True)

log.append('\n## D. J0 (calls + outputs) vs J1 (calls only) on identical action prefixes')
log.append('Prefixes built per action: J0 blocks are cut at the sequentially numbered headers "Action 0:", "Action 1:", ...; each block must start with')
log.append('the same "Tool call:" line as the corresponding J1 line (checked). k = min(20, complete actions), >= 5 required.')
pj0, pj1, keep, skipped = {}, {}, [], collections.Counter()
for i, d in enumerate(honest):
    t = text[d['trajectory_id']]; raw, calls = t['raw'], t['calls'].split('\n')
    blocks, pos, n = [], 0, 0
    while raw.startswith('Action 0:\n'):
        nxt = raw.find(f'\n\nAction {n + 1}:\n', pos)
        if nxt < 0:
            if len(raw) < 100000:
                blocks.append(raw[pos:])                                     # last block complete only if view not capped
            break
        blocks.append(raw[pos:nxt]); pos = nxt + 2; n += 1
    nc = len(calls) - (len(t['calls']) >= 100000)
    k = min(20, len(blocks), nc)
    if k < 5:
        skipped['short'] += 1; continue
    if not all(blocks[j].split('\n')[1] == calls[j] for j in range(k)):
        skipped['call_line_mismatch'] += 1; continue
    pj0[i] = '\n\n'.join(blocks[:k]); pj1[i] = '\n'.join(calls[:k]); keep.append(i)
cap = {'J0_prefixes_over_30k_chars': sum(len(pj0[i]) > 30000 for i in keep), 'J1_prefixes_over_30k_chars': sum(len(pj1[i]) > 30000 for i in keep)}


def word_featurize(tr, te):
    hw = HashingVectorizer(analyzer='word', ngram_range=(1, 2), n_features=2 ** 18, alternate_sign=False, token_pattern=r'\S+', lowercase=False)
    a, b = hw.transform(tr), hw.transform(te); t = TfidfTransformer(sublinear_tf=True).fit(a)
    return t.transform(a), t.transform(b)


def calls_char_featurize(tr, te):
    """word 1-2-grams on the whole prefix + char 3-5-grams on the CALL LINES ONLY (equal call coverage for J0 and J1)."""
    hw = HashingVectorizer(analyzer='word', ngram_range=(1, 2), n_features=2 ** 18, alternate_sign=False, token_pattern=r'\S+', lowercase=False)
    hc = HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=2 ** 18, alternate_sign=False, lowercase=False)
    cl = lambda x: '\n'.join(l for l in x.split('\n') if l.startswith('Tool call: '))
    W = [hw.transform(tr), hw.transform(te)]; C = [hc.transform([cl(x) for x in tr]), hc.transform([cl(x) for x in te])]
    tw, tc = TfidfTransformer(sublinear_tf=True).fit(W[0]), TfidfTransformer(sublinear_tf=True).fit(C[0])
    from scipy.sparse import hstack
    return hstack([tw.transform(W[0]), tc.transform(C[0])]).tocsr(), hstack([tw.transform(W[1]), tc.transform(C[1])]).tocsr()


res['j0_j1_prefix'] = {'n': len(keep), 'skipped': dict(skipped), 'char_cap': cap}
log.append(f"n={len(keep)} skipped {dict(skipped)}; prefixes longer than the 30k-char cap of the char analyser: {cap}")
for lab, ff in (('standard featuriser (word + char on first 30k)', None), ('word n-grams only (no char cap involved)', word_featurize),
                ('word + char on call lines only (equal call coverage)', calls_char_featurize)):
    if ff is None:
        a = cv(keep, lambda i: pj0[i]); b = cv(keep, lambda i: pj1[i])
    else:
        _f = featurize; featurize = ff
        try:
            a = cv(keep, lambda i: pj0[i]); b = cv(keep, lambda i: pj1[i])
        finally:
            featurize = _f
    pr = paired(b, a, keep); res['j0_j1_prefix'][lab] = {'J0': float(a.mean()), 'J1': float(b.mean()), 'paired_J1-J0': pr}
    log.append(f"{lab:<52} J0 {a.mean():.3f}  J1 {b.mean():.3f}  J1-J0 {pr['diff']:+.3f} [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}]"); print(log[-1], flush=True)

log.append('\n## E. Order test: C1 tokens, folds and featuriser fixed; tokens shuffled within each trajectory (5 seeded shuffles)')
sh_acc = []; res['order_shuffle'] = {'observed': res['views']['C1']['acc'], 'shuffles': [], 'paired_observed-shuffle': []}
for s_ in range(5):
    r = np.random.default_rng(100 + s_); SH = [list(r.permutation(c)) for c in C1]
    ok = cv(ALL, lambda i: ' '.join(SH[i])); pr = paired(OK['C1'], ok, ALL)
    res['order_shuffle']['shuffles'].append(float(ok.mean())); res['order_shuffle']['paired_observed-shuffle'].append(pr)
    log.append(f"shuffle {s_}: acc {ok.mean():.3f}; observed - shuffled {pr['diff']:+.3f} [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}]"); print(log[-1], flush=True)
json.dump(res, open(OUT + '.json', 'w'), indent=1); open(OUT + '.txt', 'w').write('\n'.join(log) + '\n'); print(log[-1]); print(f'done in {time.time()-t0:.0f}s')
