"""
Fixed Income Visualization Engine
====================================

Publication-quality INDIVIDUAL plots for yield curve analytics.
Each plot is saved as a separate high-resolution PNG file.

Author: Andreas Kapsalis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List
import os

from data_fetcher import MATURITY_LABELS

COLORS = {
    'spot': '#00e5ff', 'ns': '#ffd740', 'nss': '#ff6e40',
    'forward': '#e040fb', 'zero': '#00e676',
    'pc1': '#2979ff', 'pc2': '#ff6d00', 'pc3': '#00e676',
    'vasicek': '#7c4dff', 'cir': '#ff1744',
    'bg': '#0a0a0f', 'text': '#e0e0e0', 'muted': '#666666',
    'grid': '#1a1a2e', 'positive': '#00e676', 'negative': '#ff1744',
    'accent': '#e040fb',
}

def _setup():
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'], 'axes.facecolor': COLORS['bg'],
        'text.color': COLORS['text'], 'axes.labelcolor': COLORS['text'],
        'xtick.color': '#aaaaaa', 'ytick.color': '#aaaaaa',
        'axes.edgecolor': '#444444', 'grid.color': COLORS['grid'],
        'grid.alpha': 0.4, 'font.family': 'monospace', 'font.size': 12,
        'axes.titlesize': 15, 'axes.labelsize': 13,
        'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'savefig.facecolor': COLORS['bg'], 'savefig.pad_inches': 0.3,
    })

def _labels(mats):
    return [MATURITY_LABELS.get(m, f"{m:.1f}Y") for m in mats]

def _save(fig, path):
    fig.savefig(path, dpi=200, facecolor=COLORS['bg'], bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ {path}")


# ══════════════════════════════════════════════════════════════
def plot_yield_curve_fit(results, save_path="01_yield_curve_fit.png"):
    _setup()
    curve = results.get('latest_curve', {})
    ns, nss = results.get('ns_fit'), results.get('nss_fit')
    zeros = results.get('zero_rates')
    if not curve: return

    mat, ylds = curve['maturities'], curve['yields']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1], sharex=True)
    fig.suptitle('US TREASURY YIELD CURVE — PARAMETRIC FIT',
                 fontsize=18, fontweight='bold', color=COLORS['text'], y=0.96)

    ax1.scatter(mat, ylds, color=COLORS['spot'], s=120, zorder=5,
                edgecolors='white', linewidths=0.8, label='Observed Par Yields')

    tau = np.linspace(0.05, max(mat) + 1, 500)
    if ns:
        ax1.plot(tau, ns.yield_at(tau), color=COLORS['ns'], linewidth=2.5,
                 label=f"Nelson-Siegel (RMSE={ns.rmse:.1f}bp, R²={ns.r_squared:.4f})")
    if nss:
        ax1.plot(tau, nss.yield_at(tau), color=COLORS['nss'], linewidth=2, linestyle='--',
                 label=f"NS-Svensson (RMSE={nss.rmse:.1f}bp, R²={nss.r_squared:.4f})")
    if zeros is not None:
        ax1.plot(mat, zeros, color=COLORS['zero'], linewidth=1.5, linestyle=':',
                 marker='D', markersize=6, label='Bootstrapped Zeros', alpha=0.9)

    ax1.set_ylabel('Yield (%)', fontsize=14)
    ax1.legend(fontsize=11, framealpha=0.3, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(mat); ax1.set_xticklabels(_labels(mat), fontsize=11)

    for m, y in zip(mat, ylds):
        if m in [1/12, 1.0, 2.0, 5.0, 10.0, 30.0]:
            ax1.annotate(f'{y:.2f}%', (m, y), textcoords="offset points",
                        xytext=(0, 14), ha='center', fontsize=9,
                        color=COLORS['spot'], fontweight='bold')

    if ns:
        p = ns.params
        ax1.text(0.02, 0.03,
                f"β₁={p['beta1']:.3f} (long rate)\nβ₂={p['beta2']:.3f} (slope)\n"
                f"β₃={p['beta3']:.3f} (curvature)\nλ ={p['lambda']:.3f} (decay)",
                transform=ax1.transAxes, fontsize=10, color=COLORS['ns'],
                fontfamily='monospace', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                         edgecolor=COLORS['ns'], alpha=0.8))

    if ns:
        ax2.bar(mat - 0.15, (ylds - ns.yield_at(mat)) * 100, width=0.3,
                color=COLORS['ns'], alpha=0.8, label='NS')
    if nss:
        ax2.bar(mat + 0.15, (ylds - nss.yield_at(mat)) * 100, width=0.3,
                color=COLORS['nss'], alpha=0.8, label='NSS')

    ax2.axhline(y=0, color=COLORS['muted'], linewidth=0.8)
    ax2.set_xlabel('Maturity (Years)', fontsize=14)
    ax2.set_ylabel('Residual (bp)', fontsize=12)
    ax2.legend(fontsize=10, framealpha=0.3)
    ax2.grid(True, alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, save_path)


def plot_pca_loadings(results, save_path="02_pca_loadings.png"):
    _setup()
    pca = results.get('pca_result')
    if not pca: return

    fig, ax = plt.subplots(figsize=(16, 9))
    mat = pca.maturities
    pc_colors = [COLORS['pc1'], COLORS['pc2'], COLORS['pc3']]
    markers = ['o', 's', '^']

    for i in range(min(3, len(pca.loadings))):
        ax.plot(mat, pca.loadings[i], color=pc_colors[i], linewidth=3,
                marker=markers[i], markersize=10, markeredgecolor='white', markeredgewidth=1,
                label=f"{pca.labels[i]} — {pca.variance_explained[i]:.1%}")

    ax.axhline(y=0, color=COLORS['muted'], linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Maturity (Years)', fontsize=14)
    ax.set_ylabel('Factor Loading', fontsize=14)
    ax.set_title(f'PCA: LITTERMAN-SCHEINKMAN DECOMPOSITION\n'
                 f'3 factors explain {pca.cumulative_variance[min(2,len(pca.cumulative_variance)-1)]:.1%} of yield curve variance',
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=13, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(mat); ax.set_xticklabels(_labels(mat), fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)


def plot_pca_variance(results, save_path="03_pca_variance.png"):
    _setup()
    pca = results.get('pca_result')
    if not pca: return

    fig, ax = plt.subplots(figsize=(14, 8))
    n = min(len(pca.variance_explained), 5)
    x = np.arange(n)
    bar_colors = [COLORS['pc1'], COLORS['pc2'], COLORS['pc3']] + [COLORS['muted']] * 5

    bars = ax.bar(x, pca.variance_explained[:n] * 100, color=bar_colors[:n],
                  alpha=0.85, edgecolor='white', linewidth=0.8, width=0.6)

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
               f'{pca.variance_explained[i]:.1%}', ha='center', fontsize=13,
               fontweight='bold', color=COLORS['text'])

    ax2 = ax.twinx()
    ax2.plot(x, pca.cumulative_variance[:n] * 100, color=COLORS['accent'],
             linewidth=3, marker='D', markersize=10, markeredgecolor='white', markeredgewidth=1)
    for i in range(n):
        ax2.annotate(f'{pca.cumulative_variance[i]:.1%}', (x[i], pca.cumulative_variance[i]*100),
                    textcoords="offset points", xytext=(12, -5), fontsize=11,
                    color=COLORS['accent'], fontweight='bold')
    ax2.set_ylabel('Cumulative (%)', color=COLORS['accent'], fontsize=13)
    ax2.set_ylim(0, 105)

    ax.set_xticks(x)
    ax.set_xticklabels(['PC1\n(Level)', 'PC2\n(Slope)', 'PC3\n(Curvature)'] +
                       [f'PC{i+1}' for i in range(3, n)], fontsize=12)
    ax.set_ylabel('Individual Variance (%)', fontsize=13)
    ax.set_title('PCA SCREE PLOT', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    _save(fig, save_path)


def plot_pca_factor_scores(results, save_path="04_pca_factor_scores.png"):
    _setup()
    pca = results.get('pca_result')
    yields_df = results.get('yields_df')
    if not pca or yields_df is None: return

    dates = yields_df.diff().dropna().index[:len(pca.factor_scores)]
    pc_colors = [COLORS['pc1'], COLORS['pc2'], COLORS['pc3']]

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.suptitle('PCA FACTOR SCORES — HISTORICAL', fontsize=16, fontweight='bold', y=0.97)

    for i, ax in enumerate(axes):
        if i >= len(pca.labels): break
        s = pca.factor_scores[:, i]
        ax.plot(dates, s, color=pc_colors[i], linewidth=1.2, alpha=0.9)
        ax.fill_between(dates, 0, s, where=s > 0, color=pc_colors[i], alpha=0.15)
        ax.fill_between(dates, 0, s, where=s < 0, color=COLORS['negative'], alpha=0.15)
        ax.axhline(y=0, color=COLORS['muted'], linewidth=0.8)
        sigma = np.std(s)
        ax.axhline(y=2*sigma, color=pc_colors[i], linewidth=0.8, linestyle=':', alpha=0.5)
        ax.axhline(y=-2*sigma, color=pc_colors[i], linewidth=0.8, linestyle=':', alpha=0.5)
        ax.set_ylabel(pca.labels[i], fontsize=12, color=pc_colors[i])
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Date', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, save_path)


def plot_yield_surface(results, save_path="05_yield_surface.png"):
    _setup()
    yields_df = results.get('yields_df')
    if yields_df is None or yields_df.empty: return

    fig, ax = plt.subplots(figsize=(18, 10))
    n = min(500, len(yields_df))
    step = max(1, len(yields_df) // n)
    sampled = yields_df.iloc[::step]
    mat = np.array(sampled.columns, dtype=float)

    im = ax.imshow(sampled.values.T, aspect='auto', cmap='RdYlGn_r', origin='lower',
                   extent=[0, len(sampled), mat[0], mat[-1]], interpolation='bilinear')

    ticks = np.linspace(0, len(sampled)-1, 8, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([sampled.index[i].strftime('%b %Y') for i in ticks], fontsize=11, rotation=30)
    ax.set_ylabel('Maturity (Years)', fontsize=14)
    ax.set_title('HISTORICAL YIELD SURFACE — US TREASURY EVOLUTION', fontsize=16, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=30)
    cbar.set_label('Yield (%)', fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)


def plot_short_rate_paths(results, save_path="06_short_rate_paths.png"):
    _setup()
    vasicek, cir = results.get('vasicek_result'), results.get('cir_result')
    if vasicek is None and cir is None: return

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle('SHORT-RATE MODELS — 5Y MONTE CARLO', fontsize=16, fontweight='bold', y=0.97)

    for ax, model, color, name in [(axes[0], vasicek, COLORS['vasicek'], 'Vasicek'),
                                     (axes[1], cir, COLORS['cir'], 'CIR')]:
        if model is None:
            ax.text(0.5, 0.5, f'{name}: Not fitted', transform=ax.transAxes, ha='center', fontsize=14)
            continue

        paths = model.simulate_paths(T=5.0, n_steps=1260, n_paths=1000, seed=42)
        t = np.linspace(0, 5, paths.shape[1])

        for p_lo, p_hi, a in [(5, 95, 0.1), (25, 75, 0.2)]:
            ax.fill_between(t, np.percentile(paths*100, p_lo, axis=0),
                           np.percentile(paths*100, p_hi, axis=0), color=color, alpha=a)
        ax.plot(t, np.median(paths*100, axis=0), color=color, linewidth=2.5, label='Median')
        ax.axhline(y=model.theta*100, color='white', linewidth=1.2, linestyle='--', alpha=0.6,
                   label=f'θ = {model.theta*100:.2f}%')

        for j in np.random.RandomState(42).choice(paths.shape[0], 8, replace=False):
            ax.plot(t, paths[j]*100, color=color, alpha=0.08, linewidth=0.5)

        feller = f"\nFeller: {'✓' if model.feller_satisfied else '✗'}" if name == 'CIR' else ""
        ax.text(0.97, 0.97, f"κ={model.kappa:.4f}\nθ={model.theta*100:.3f}%\n"
                f"σ={model.sigma*100:.3f}%\nt½={model.half_life:.1f}y{feller}",
                transform=ax.transAxes, fontsize=11, color=color, fontfamily='monospace',
                va='top', ha='right', bbox=dict(boxstyle='round,pad=0.4',
                facecolor='#1a1a2e', edgecolor=color, alpha=0.85))

        ax.set_xlabel('Time (Years)', fontsize=13)
        ax.set_ylabel('Short Rate (%)', fontsize=13)
        sde = 'σ√r dW' if name == 'CIR' else 'σ dW'
        ax.set_title(f'{name}: dr = κ(θ−r)dt + {sde}', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, framealpha=0.3, loc='lower right')
        ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, save_path)


def plot_short_rate_fit(results, save_path="07_short_rate_curve_fit.png"):
    _setup()
    curve = results.get('latest_curve', {})
    vasicek, cir = results.get('vasicek_result'), results.get('cir_result')
    if not curve or (vasicek is None and cir is None): return

    fig, ax = plt.subplots(figsize=(16, 9))
    mat, ylds = curve['maturities'], curve['yields']

    ax.scatter(mat, ylds, color=COLORS['spot'], s=120, zorder=5,
               edgecolors='white', linewidths=0.8, label='Observed Treasury Yields')

    mm = np.linspace(0.08, 30, 300)
    if vasicek:
        ax.plot(mm, vasicek.yield_curve(mm), color=COLORS['vasicek'], linewidth=2.5,
                label=f'Vasicek (κ={vasicek.kappa:.3f}, θ={vasicek.theta*100:.2f}%)')
    if cir:
        ax.plot(mm, cir.yield_curve(mm), color=COLORS['cir'], linewidth=2, linestyle='--',
                label=f'CIR (κ={cir.kappa:.3f}, θ={cir.theta*100:.2f}%)')

    ax.set_xlabel('Maturity (Years)', fontsize=14)
    ax.set_ylabel('Yield (%)', fontsize=14)
    ax.set_title('SHORT-RATE MODEL IMPLIED YIELD CURVES vs OBSERVED', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=12, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


def plot_duration_convexity(results, save_path="08_duration_convexity.png"):
    _setup()
    ba = results.get('bond_analytics', [])
    if not ba: return

    fig, ax = plt.subplots(figsize=(14, 10))
    dur = [a.modified_duration for a in ba]
    cvx = [a.convexity for a in ba]
    ytm = [a.ytm * 100 for a in ba]

    sc = ax.scatter(dur, cvx, c=ytm, cmap='coolwarm', s=[max(a.dv01*200000, 80) for a in ba],
                   edgecolors='white', linewidths=1, zorder=5, alpha=0.9)

    for a in ba:
        ax.annotate(f'{a.name}\nDur={a.modified_duration:.1f} DV01={a.dv01:.4f}\nYTM={a.ytm*100:.2f}%',
                   (a.modified_duration, a.convexity), textcoords="offset points", xytext=(15, 10),
                   fontsize=9, color=COLORS['text'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.8, edgecolor='#444444'),
                   arrowprops=dict(arrowstyle='->', color='#666666', alpha=0.5))

    plt.colorbar(sc, ax=ax, shrink=0.8).set_label('YTM (%)', fontsize=12)
    ax.set_xlabel('Modified Duration (Years)', fontsize=14)
    ax.set_ylabel('Convexity', fontsize=14)
    ax.set_title('BOND PORTFOLIO: DURATION-CONVEXITY MAP (bubble ∝ DV01)', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


def plot_butterfly_scenarios(results, save_path="09_butterfly_scenarios.png"):
    _setup()
    fs = results.get('butterfly_scenarios')
    if fs is None or fs.empty: return

    fig, ax = plt.subplots(figsize=(16, 10))
    scenarios = fs.index.tolist()
    pnl = fs['Total_PnL'].values
    colors = [COLORS['positive'] if p > 0 else COLORS['negative'] for p in pnl]

    bars = ax.barh(range(len(scenarios)), pnl, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=12)
    ax.axvline(x=0, color='white', linewidth=1.2, alpha=0.5)

    for i, val in enumerate(pnl):
        offset = np.max(np.abs(pnl)) * 0.03
        x = val + offset if val >= 0 else val - offset
        ax.text(x, i, f'${val:+.2f}', va='center', ha='left' if val >= 0 else 'right',
               fontsize=11, fontweight='bold', color=COLORS['text'])

    ax.set_xlabel('P&L ($)', fontsize=13)
    ax.set_title('2s5s10s BUTTERFLY — SCENARIO P&L\n(Duration-Neutral: Long Wings, Short Body)',
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, axis='x', alpha=0.2)
    fig.tight_layout()
    _save(fig, save_path)


def plot_forward_vs_spot(results, save_path="10_forward_vs_spot.png"):
    _setup()
    ns = results.get('ns_fit')
    if not ns: return

    fig, ax = plt.subplots(figsize=(16, 9))
    tau = np.linspace(0.1, 30, 500)
    spot, fwd = ns.yield_at(tau), ns.forward_at(tau)

    ax.plot(tau, spot, color=COLORS['spot'], linewidth=3, label='Spot (Zero) Curve')
    ax.plot(tau, fwd, color=COLORS['forward'], linewidth=2.5, linestyle='--', label='Instantaneous Forward')
    ax.fill_between(tau, spot, fwd, where=fwd > spot, color=COLORS['forward'], alpha=0.15, label='Term Premium (+)')
    ax.fill_between(tau, spot, fwd, where=fwd < spot, color=COLORS['negative'], alpha=0.15, label='Term Premium (−)')

    for t1, t2 in [(1, 2), (2, 5), (5, 10)]:
        y = ns.yield_at(np.array([float(t1), float(t2)]))
        fr = (y[1]*t2 - y[0]*t1) / (t2 - t1)
        ax.annotate(f'{t1}y{t2-t1}y = {fr:.2f}%', xy=((t1+t2)/2, fr),
                   xytext=((t1+t2)/2 + 2, fr + 0.12), fontsize=10, color=COLORS['forward'],
                   arrowprops=dict(arrowstyle='->', color=COLORS['forward'], alpha=0.6))

    ax.set_xlabel('Maturity (Years)', fontsize=14)
    ax.set_ylabel('Rate (%)', fontsize=14)
    ax.set_title('FORWARD vs SPOT — TERM PREMIUM DECOMPOSITION', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=12, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


def plot_trading_signals(results, save_path="11_trading_signals.png"):
    _setup()
    signals = results.get('signals')
    if signals is None or signals.empty: return

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    fig.suptitle('YIELD CURVE TRADING SIGNALS', fontsize=16, fontweight='bold', y=0.97)

    for ax, col, color, title in [
        (axes[0], 'slope_2s10s', COLORS['pc2'], '2s10s Slope Signal (+ steep → expect flattening)'),
        (axes[1], 'butterfly_2s5s10s', COLORS['pc3'], '2s5s10s Butterfly (+ belly cheap → sell fly)'),
    ]:
        if col in signals.columns:
            s = signals[col]
            ax.plot(s.index, s.values, color=color, linewidth=1.5)
            ax.fill_between(s.index, 0, s.values, where=s > 0, color=color, alpha=0.2)
            ax.fill_between(s.index, 0, s.values, where=s < 0, color=COLORS['negative'], alpha=0.2)
            ax.axhline(y=0, color=COLORS['muted'], linewidth=0.8)
            for thresh in [2, -2]:
                ax.axhline(y=thresh, color=COLORS['muted'], linewidth=0.8, linestyle=':')
            ax.set_ylabel('Z-Score', fontsize=12)
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.2)

    if 'regime' in signals.columns:
        r = signals['regime']
        axes[2].fill_between(r.index, 0, r.values, where=r > 0, color=COLORS['pc2'], alpha=0.4, label='Steepening')
        axes[2].fill_between(r.index, 0, r.values, where=r < 0, color=COLORS['negative'], alpha=0.4, label='Flattening')
        axes[2].axhline(y=0, color=COLORS['muted'], linewidth=0.8)
        axes[2].set_yticks([-1, 0, 1]); axes[2].set_yticklabels(['Flattening', 'Neutral', 'Steepening'])
        axes[2].set_title('Curve Regime Indicator', fontsize=12)
        axes[2].legend(fontsize=10, framealpha=0.3)
        axes[2].grid(True, alpha=0.2)

    axes[-1].set_xlabel('Date', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, save_path)


def create_all_plots(results, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    print("  Generating individual plots...")
    for func, fname in [
        (plot_yield_curve_fit, "01_yield_curve_fit.png"),
        (plot_pca_loadings, "02_pca_loadings.png"),
        (plot_pca_variance, "03_pca_variance.png"),
        (plot_pca_factor_scores, "04_pca_factor_scores.png"),
        (plot_yield_surface, "05_yield_surface.png"),
        (plot_short_rate_paths, "06_short_rate_paths.png"),
        (plot_short_rate_fit, "07_short_rate_curve_fit.png"),
        (plot_duration_convexity, "08_duration_convexity.png"),
        (plot_butterfly_scenarios, "09_butterfly_scenarios.png"),
        (plot_forward_vs_spot, "10_forward_vs_spot.png"),
        (plot_trading_signals, "11_trading_signals.png"),
    ]:
        path = os.path.join(output_dir, fname)
        try:
            func(results, save_path=path)
            saved.append(path)
        except Exception as e:
            print(f"    ✗ {fname}: {e}")
    return saved


def create_master_dashboard(results, save_path="yield_curve_dashboard.png"):
    output_dir = os.path.dirname(save_path) or "."
    return create_all_plots(results, output_dir=output_dir)
