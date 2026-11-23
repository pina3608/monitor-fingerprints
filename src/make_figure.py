"""Headline figure: honest-run attribution by view, and per-pipeline attack-run recall (honest-trained classifier).
Reads committed results only. Usage: python -I src/make_figure.py results report/figures/headline.png
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES, OUT = sys.argv[1], sys.argv[2]
ident = json.load(open(os.path.join(RES, 'identification.json')))['per_transform']
harn = json.load(open(os.path.join(RES, 'harness_audit.json')))
base = json.load(open(os.path.join(RES, 'baselines.json')))
rec = json.load(open(os.path.join(RES, 'canonical_calls.json')))['views']['J1']['transfer_recall_per_model']

views = [('Tool calls (no outputs)', ident['J1_calls']['acc_11way'], True),
         ('Tool-call metadata only', harn['acc']['H2_wrapper_only'], True),
         ('Full monitor view (calls + outputs)', ident['J0_raw']['acc_11way'], True),
         ('Common-tool command verbs', base['views']['C1']['acc'], True),
         ('Same calls: bash vs editor only', base['views']['T1']['acc'], False),
         ('Outcome + length + template', base['views']['LOT']['acc'], False)]
SURF, INK, INK2, BLUE, GRAY, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#2a78d6', '#b9b8b2', '#e6e5e1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK})
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1.15, 1]}, facecolor=SURF)


def style(ax):
    ax.set_facecolor(SURF); [ax.spines[s].set_visible(False) for s in ('top', 'right', 'left')]
    ax.spines['bottom'].set_color(GRID); ax.tick_params(axis='y', length=0); ax.tick_params(axis='x', color=GRID)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlim(0, 1.1)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])


y = list(range(len(views)))[::-1]
a.barh(y, [v[1] for v in views], height=0.62, color=[BLUE if v[2] else GRAY for v in views], edgecolor=SURF, linewidth=2)
a.set_yticks(y); a.set_yticklabels([v[0] for v in views])
for yi, v in zip(y, views):
    a.text(v[1] + 0.012, yi, f'{v[1]:.1%}', va='center', fontsize=10)
a.axvline(1 / 11, color=INK2, ls=(0, (3, 3)), lw=1.2); a.text(0.097, len(views) - 0.45, 'chance 9.1%', fontsize=9, color=INK2, va='bottom')
style(a); a.set_title('Honest runs: which pipeline wrote it? (11-way)', loc='left', fontsize=11.5, fontweight='bold', pad=10)
order = sorted(rec, key=lambda m: rec[m]); yb = list(range(len(order)))
b.barh(yb, [rec[m] for m in order], height=0.62, color=BLUE, edgecolor=SURF, linewidth=2)
b.set_yticks(yb); b.set_yticklabels(order)
for yi, m in zip(yb, order):
    b.text(rec[m] + 0.012, yi, f'{rec[m]:.0%}', va='center', fontsize=10, bbox=dict(facecolor=SURF, edgecolor='none', pad=0.5))
pooled = ident['J1_calls']['acc_transfer_honest_to_attack']
b.axvline(pooled, color=INK2, ls=(0, (3, 3)), lw=1.2); b.text(pooled - 0.012, -0.95, f'pooled {pooled:.1%}', fontsize=9, color=INK2, va='bottom', ha='right')
style(b); b.set_title('Attack runs: recall per pipeline', loc='left', fontsize=11.5, fontweight='bold', pad=10); b.set_ylim(-1.1, len(order) - 0.4)
fig.text(0.01, 0.015, 'LinuxArena, 11 pipelines, held-out tasks. Linear SVM on n-grams of the monitor-visible text. Gray bars: baselines. Attack runs: classifier trained only on\n'
         '2,056 honest runs of tasks outside the attack set, tested on 1,012 attack runs.', fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT, dpi=200, facecolor=SURF)
