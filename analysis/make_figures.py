"""Regenerate the three data figures of the paper (3.4 x 2.8 in, 600 dpi PNG).

    python analysis/make_figures.py --cp cp_per_verify.csv --mcc-progress mcc_progress.csv --fig3 fig3_data.csv --out OUT [--mode both|mean|median]

  OUT/fig3_viper_sympol_bars.png      Fig. 3: mean / min return over 1,000 episodes and verified safe steps
  OUT/cp_progress.png                 CartPole progress (verified safe steps per fine-tuning iteration)
  OUT/mcc_progress.png                MountainCarContinuous progress (ratio of safe start state regions)
  OUT/cp_progress_plotted_<mode>.csv  the plotted CartPole values
Inputs: analysis/collect_cartpole.py, analysis/make_tables.py (mcc_progress.csv), analysis/fig3_data.py.
mode 'both' (paper) = mean solid with markers, median dashed.
Font: the paper figures use the macOS system font Avenir Next; elsewhere the first available font of
FONT_CHAIN is used (layout and colours unchanged, glyph metrics differ slightly).
"""
import argparse, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ap = argparse.ArgumentParser(description='Regenerate the paper figures.')
ap.add_argument('--cp', required=True); ap.add_argument('--mcc-progress', required=True); ap.add_argument('--fig3', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--mode', default='both', choices=['both', 'median', 'mean'])
_a = ap.parse_args()
out = _a.out
import os; os.makedirs(out, exist_ok=True)
MODE = _a.mode
STAT = 'mean' if MODE == 'both' else MODE
FONT_CHAIN = ['Avenir Next', 'Nunito Sans', 'Helvetica Neue', 'Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans']
_avail = {f.name for f in font_manager.fontManager.ttflist}
_font = next((f for f in FONT_CHAIN if f in _avail), 'DejaVu Sans')
if _font != 'Avenir Next':
    print(f'[make_figures] Avenir Next not installed, using {_font}', file=sys.stderr)
BLUE, RED, LBLUE, GREY, FONT, DPI = '#1a56c4', '#9b1c1c', '#d9e4fb', '#555', _font, 600
LW = [0, 1.0, 2.6, 4.6]      # regime-bar width for >=25 / 50 / 75 % of runs ended
ADJ = dict(left=0.175, right=0.97, top=0.895, bottom=0.175)

def rc():
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 9, 'font.family': FONT, 'axes.linewidth': 0.8, 'axes.edgecolor': '#333',
                         'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.color': '#333', 'ytick.color': '#333'})

# ---- CartPole: best-so-far verified depth per run; a run ends when a lower-bound (VRAM-censored) candidate is accepted
d = pd.read_csv(_a.cp).sort_values(['run', 'iter'])
recs = []
for (method, run), g in d.groupby(['method', 'run']):
    best, lb, stopk, hist = None, False, None, {}
    for _, x in g.iterrows():
        if x.accepted: best = x.depth; lb = not x.exact
        hist[int(x.ft)] = best
        if lb: stopk = int(x.ft)
    last = max(hist)
    for k in range(12):
        recs.append(dict(method=method, run=run, k=k, depth=hist[min(k, last)], ended=(stopk is not None and k >= stopk)))
R = pd.DataFrame(recs)
cp = R.groupby(['method', 'k']).agg(y=('depth', STAT), med=('depth', 'median'), ended=('ended', 'mean')).reset_index()
cp['ended'] *= 100
# ---- MountainCar: per-iteration statistics; a run ends when it is fully proven safe
W = pd.read_csv(_a.mcc_progress)

def line_fig(x, yv, yu, ev, eu, title, ylabel, ylim, yticks, xlim, xticks, path, mv=None, mu=None):
    rc(); fig = plt.figure(figsize=(3.4, 2.8)); gs = fig.add_gridspec(2, 1, height_ratios=(6.2, 1), hspace=0.10, **ADJ)
    ax = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1], sharex=ax)
    ax.plot(x, yu, '-s', color=RED, lw=1.1, ms=3.0); ax.plot(x, yv, '-o', color=BLUE, lw=1.1, ms=4.2)
    if mv is not None:
        ax.plot(x, mu, '--', color=RED, lw=1.1, dashes=(4, 2.5)); ax.plot(x, mv, '--', color=BLUE, lw=1.1, dashes=(4, 2.5))
    ax.grid(color='#dcdcdc', lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelbottom=False)
    ax.set_title(title, fontsize=10, fontweight='medium'); ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.set_xlim(*xlim); ax.set_xticks(xticks)
    if yticks is not None: ax.set_yticks(yticks)
    h = [Line2D([], [], color=BLUE, lw=0, marker='o', ms=4.2, label='VESPA (Ours)'), Line2D([], [], color=RED, lw=0, marker='s', ms=3.0, label='Uninformed'), Line2D([], [], lw=0, label='')]
    if mv is not None: h = h[:2] + [Line2D([], [], color='k', lw=1.1, label='mean'), Line2D([], [], color='k', lw=1.1, ls='--', dashes=(4, 2.5), label='median')]
    h += [Line2D([], [], color=GREY, lw=LW[i], solid_capstyle='butt', label=f'\u2265{p}% ended') for i, p in ((1, 25), (2, 50), (3, 75))]
    ax.legend(handles=h, loc='lower right', ncol=2, frameon=False, fontsize=6.8, handlelength=1.8, columnspacing=0.8, labelspacing=0.35, borderaxespad=0.3, handletextpad=0.5)
    for j, (e, c) in enumerate(((ev, BLUE), (eu, RED))):
        for k, val in zip(x, e):
            lvl = int(val >= 25) + int(val >= 50) + int(val >= 75)
            if lvl: b.plot([k - 0.5, k + 0.5], [-j, -j], color=c, lw=LW[lvl], solid_capstyle='butt')
    b.set_ylim(-1.55, 0.55); b.set_yticks([]); b.set_ylabel('runs\nended', fontsize=6.8, rotation=0, ha='right', va='center', labelpad=4); b.set_xlabel('fine-tuning iteration')
    fig.savefig(path, dpi=DPI); plt.close(fig)

v, u = cp[cp.method == 'VESPA'], cp[cp.method == 'Uninformed']
lo, hi = min(v.y.min(), u.y.min()), max(v.y.max(), u.y.max())
line_fig(v.k.values, v.y.values, u.y.values, v.ended.values, u.ended.values, 'CartPole', 'verified safe steps (\u2265)',
         (58 if MODE == 'mean' else 49, 96 if MODE == 'mean' else 98), None, (-0.6, 11.6), range(0, 12, 2), f'{out}/cp_progress.png', *((v.med.values, u.med.values) if MODE == 'both' else ()))
line_fig(W.k.values, W[f'VESPA_{STAT}'].values, W[f'Uninformed_{STAT}'].values, 100 * W.VESPA_n_safe / W.VESPA_n, 100 * W.Uninformed_n_safe / W.Uninformed_n,
         'MountainCarContinuous', 'safe start state regions (%)', (50, 102), range(50, 101, 10), (-0.55, 5.55), range(6), f'{out}/mcc_progress.png', *((W.VESPA_median.values, W.Uninformed_median.values) if MODE == 'both' else ()))

# ---- Bar plot: single input policies (return over 1000 episodes, verified safe steps)
B = pd.read_csv(_a.fig3).set_index('policy_id').loc[['viper_L4_s6', 'viper_L32_s5', 'viper_L32_s4', 's8u', 's27u', 's40u']]
rc(); fig, ax = plt.subplots(figsize=(3.4, 2.8)); fig.subplots_adjust(left=0.04, right=0.98, top=0.895, bottom=0.175)
xs, w = [0, 1, 2, 3.6, 4.6, 5.6], 0.27
for xc, (pid, r) in zip(xs, B.iterrows()):
    vals = [(r['mean'], GREY, '#e4e4e4', f"{r['mean']:.0f}"), (r['min'], GREY, 'white', f"{r['min']:.0f}"),
            (r.verified_value, BLUE, LBLUE, ('\u2265' if r.verified_kind == '>=' else '') + f"{r.verified_value:.0f}")]
    for i, (val, ec, fc, lab) in enumerate(vals):
        xx = xc + (i - 1) * w; ax.bar(xx, val, w * 0.86, fc=fc, ec=ec, lw=1.0)
        ax.text(xx, val + 12, lab, rotation=90, ha='center', va='bottom', fontsize=7, color=ec if ec != GREY else '#444', fontweight='bold' if ec == BLUE else 'normal')
ax.set_xticks(xs); ax.set_xticklabels(['d3', 'd9', 'd10', 's1', 's2', 's3']); ax.tick_params(axis='x', length=0)
for s in ('left', 'top', 'right'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_linewidth(1.2); ax.set_yticks([]); ax.set_ylim(0, 700); ax.set_xlim(-0.6, 6.2); ax.axvline(2.8, ymax=0.8, color='#bbb', lw=0.8)
ax.text(1, -0.115, 'VIPER', transform=ax.get_xaxis_transform(), ha='center', va='top'); ax.text(4.6, -0.115, 'SYMPOL', transform=ax.get_xaxis_transform(), ha='center', va='top')
ax.legend(handles=[Patch(fc='#e4e4e4', ec=GREY, label='mean return'), Patch(fc='white', ec=GREY, label='min return'), Patch(fc=LBLUE, ec=BLUE, label='safe steps')],
          loc='upper center', ncol=3, frameon=False, fontsize=7.5, columnspacing=1.0, handlelength=1.2)
ax.set_title('CartPole', fontsize=10, fontweight='medium')
fig.savefig(f'{out}/fig3_viper_sympol_bars.png', dpi=DPI); plt.close(fig)
cp.pivot(index='k', columns='method', values=['y', 'med', 'ended']).round(2).to_csv(f'{out}/cp_progress_plotted_{MODE}.csv')
