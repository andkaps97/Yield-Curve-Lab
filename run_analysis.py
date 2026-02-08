#!/usr/bin/env python3
"""
Yield Curve Lab — Main Entry Point
=====================================

Runs the complete fixed income analytics pipeline:
1. Fetch/generate Treasury yield data
2. Bootstrap zero-coupon rates
3. Fit Nelson-Siegel and NS-Svensson curves
4. PCA decomposition (Litterman-Scheinkman)
5. Calibrate Vasicek & CIR short-rate models
6. Bond portfolio duration/convexity analytics
7. Butterfly/barbell trade analysis
8. Forward rate extraction & curve trading signals
9. Generate comprehensive dashboard

Usage:
    python run_analysis.py

Author: Andreas Kapsalis
"""

import sys
import os
import time
import numpy as np
import pandas as pd

from data_fetcher import TreasuryDataFetcher, MATURITY_LABELS
from curve_fitting import (
    YieldCurveBootstrapper, NelsonSiegelFitter, ForwardRateCalculator
)
from pca_analysis import YieldCurvePCA
from short_rate_models import ShortRateCalibrator
from bond_analytics import Bond, BondCalculator, PortfolioAnalytics
from trading_strategies import ButterflyAnalyzer, BarbellBulletAnalyzer, CurveTradingSignals
from visualization import create_all_plots


def print_section(title: str):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def main():
    start_time = time.time()

    print("\n" + "▓" * 70)
    print("  YIELD CURVE LAB — Fixed Income Analytics & Research")
    print("  Complete Pipeline Execution")
    print("▓" * 70)

    results = {}

    # ══════════════════════════════════════════════════════════
    # 1. DATA ACQUISITION
    # ══════════════════════════════════════════════════════════
    print_section("1. TREASURY DATA ACQUISITION")

    fetcher = TreasuryDataFetcher()
    yields_df = fetcher.fetch(start="2018-01-01", use_synthetic=False)
    results['yields_df'] = yields_df

    print(f"  Period:     {yields_df.index[0].strftime('%Y-%m-%d')} to {yields_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Obs:        {len(yields_df):,} trading days")
    print(f"  Maturities: {len(yields_df.columns)} points")

    # Summary stats
    stats = fetcher.summary_stats()
    print(f"\n  Yield Level Summary:")
    print(f"  {'Tenor':<8s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Δ Mean':>8s} {'Δ Std':>8s}")
    print(f"  {'─'*56}")
    for idx, row in stats.iterrows():
        print(f"  {idx:<8s} {row['level_mean']:>7.2f}% {row['level_std']:>7.2f}% "
              f"{row['level_min']:>7.2f}% {row['level_max']:>7.2f}% "
              f"{row['chg_mean']:>7.3f}  {row['chg_std']:>7.3f}")

    # Latest curve
    maturities, latest_yields = fetcher.get_latest_curve()
    results['latest_curve'] = {'maturities': maturities, 'yields': latest_yields}

    print(f"\n  Latest Yield Curve ({yields_df.index[-1].strftime('%Y-%m-%d')}):")
    for m, y in zip(maturities, latest_yields):
        label = MATURITY_LABELS.get(m, f"{m:.1f}Y")
        print(f"    {label:>5s}: {y:.3f}%")

    # ══════════════════════════════════════════════════════════
    # 2. BOOTSTRAPPING
    # ══════════════════════════════════════════════════════════
    print_section("2. YIELD CURVE BOOTSTRAPPING")

    bootstrapper = YieldCurveBootstrapper()
    zero_rates, disc_factors, fwd_rates = bootstrapper.bootstrap(maturities, latest_yields)
    results['zero_rates'] = zero_rates
    results['discount_factors'] = disc_factors
    results['forward_rates'] = fwd_rates

    print(f"\n  {'Tenor':<8s} {'Par Yield':>10s} {'Zero Rate':>10s} {'Disc Factor':>12s} {'Fwd Rate':>10s}")
    print(f"  {'─'*54}")
    for m, py, zr, df, fr in zip(maturities, latest_yields, zero_rates, disc_factors, fwd_rates):
        label = MATURITY_LABELS.get(m, f"{m:.1f}Y")
        print(f"  {label:<8s} {py:>9.3f}% {zr:>9.3f}% {df:>11.6f} {fr:>9.3f}%")

    # ══════════════════════════════════════════════════════════
    # 3. NELSON-SIEGEL & NSS FITTING
    # ══════════════════════════════════════════════════════════
    print_section("3. PARAMETRIC CURVE FITTING")

    fitter = NelsonSiegelFitter()

    # Nelson-Siegel
    ns_result = fitter.fit_ns(maturities, latest_yields)
    results['ns_fit'] = ns_result

    print(f"\n  Nelson-Siegel Fit:")
    print(f"    β₁ (level):     {ns_result.params['beta1']:>8.4f}  (long-run rate)")
    print(f"    β₂ (slope):     {ns_result.params['beta2']:>8.4f}  (short rate = β₁+β₂ = {ns_result.short_rate:.4f})")
    print(f"    β₃ (curvature): {ns_result.params['beta3']:>8.4f}  (hump/trough)")
    print(f"    λ  (decay):     {ns_result.params['lambda']:>8.4f}")
    print(f"    RMSE:           {ns_result.rmse:>8.2f} bp")
    print(f"    Max Error:      {ns_result.max_error:>8.2f} bp")
    print(f"    R²:             {ns_result.r_squared:>8.6f}")

    # Nelson-Siegel-Svensson
    nss_result = fitter.fit_nss(maturities, latest_yields)
    results['nss_fit'] = nss_result

    print(f"\n  Nelson-Siegel-Svensson Fit:")
    print(f"    β₁ (level):     {nss_result.params['beta1']:>8.4f}")
    print(f"    β₂ (slope):     {nss_result.params['beta2']:>8.4f}")
    print(f"    β₃ (curv 1):    {nss_result.params['beta3']:>8.4f}")
    print(f"    β₄ (curv 2):    {nss_result.params['beta4']:>8.4f}")
    print(f"    λ₁:             {nss_result.params['lambda1']:>8.4f}")
    print(f"    λ₂:             {nss_result.params['lambda2']:>8.4f}")
    print(f"    RMSE:           {nss_result.rmse:>8.2f} bp")
    print(f"    R²:             {nss_result.r_squared:>8.6f}")

    # Forward rates from NS fit
    print(f"\n  Forward Rates (from NS fit):")
    frc = ForwardRateCalculator()
    forward_tenors = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
    for start in [1.0, 2.0, 5.0]:
        for tenor in [1.0, 2.0, 5.0]:
            fwd = frc.discrete_forward(ns_result, start, start + tenor)
            print(f"    {int(start)}y{int(tenor)}y forward: {fwd:.3f}%")
        print()

    # ══════════════════════════════════════════════════════════
    # 4. PCA DECOMPOSITION
    # ══════════════════════════════════════════════════════════
    print_section("4. PCA — LITTERMAN-SCHEINKMAN DECOMPOSITION")

    pca_engine = YieldCurvePCA()
    pca_result = pca_engine.decompose(yields_df, n_components=3)
    results['pca_result'] = pca_result

    print(f"\n  Variance Decomposition:")
    print(f"  {'Factor':<25s} {'Eigenvalue':>12s} {'Var Explained':>14s} {'Cumulative':>12s}")
    print(f"  {'─'*63}")
    for i in range(3):
        print(f"  {pca_result.labels[i]:<25s} "
              f"{pca_result.eigenvalues[i]:>12.6f} "
              f"{pca_result.variance_explained[i]:>13.1%} "
              f"{pca_result.cumulative_variance[i]:>11.1%}")

    print(f"\n  Factor Loadings:")
    mat_labels = [MATURITY_LABELS.get(m, f"{m:.1f}Y") for m in pca_result.maturities]
    header = f"  {'':>12s}" + "".join(f"{l:>8s}" for l in mat_labels)
    print(header)
    print(f"  {'─' * (12 + 8 * len(mat_labels))}")
    for i in range(3):
        row = f"  {pca_result.labels[i]:>12s}"
        for j in range(len(pca_result.maturities)):
            row += f"{pca_result.loadings[i, j]:>8.4f}"
        print(row)

    # Scenario analysis
    scenarios = {
        'Level +1σ':    np.array([1.0, 0.0, 0.0]),
        'Level -1σ':    np.array([-1.0, 0.0, 0.0]),
        'Steepening':   np.array([0.0, 1.0, 0.0]),
        'Flattening':   np.array([0.0, -1.0, 0.0]),
        'Curvature +':  np.array([0.0, 0.0, 1.0]),
        'Curvature -':  np.array([0.0, 0.0, -1.0]),
        'Bear Flatten': np.array([1.0, -1.0, 0.0]),
        'Bull Steep':   np.array([-1.0, 1.0, 0.0]),
    }
    scenario_curves = pca_engine.scenario_analysis(pca_result, scenarios)

    print(f"\n  Scenario Analysis (yield changes in % at key maturities):")
    key_mats = [2.0, 5.0, 10.0, 30.0]
    key_idx = [np.argmin(np.abs(pca_result.maturities - m)) for m in key_mats]
    print(f"  {'Scenario':<18s}" + "".join(f"{MATURITY_LABELS.get(m, ''):>10s}" for m in key_mats))
    print(f"  {'─'*58}")
    for name in scenarios:
        curve = scenario_curves[name].values
        base = pca_result.mean_curve
        changes = curve - base
        row = f"  {name:<18s}"
        for idx in key_idx:
            row += f"{changes[idx]:>+9.3f}%"
        print(row)

    # ══════════════════════════════════════════════════════════
    # 5. SHORT-RATE MODELS
    # ══════════════════════════════════════════════════════════
    print_section("5. SHORT-RATE MODEL CALIBRATION")

    calibrator = ShortRateCalibrator()

    # Use short end as proxy for short rate
    short_mat = maturities[0]
    short_rates = yields_df[short_mat].values / 100.0  # Convert to decimal

    # Vasicek
    vasicek = calibrator.fit_vasicek(short_rates, dt=1/252)
    results['vasicek_result'] = vasicek

    print(f"\n  Vasicek Model (dr = κ(θ-r)dt + σdW):")
    print(f"    κ (mean-rev speed):  {vasicek.kappa:.4f} (half-life: {vasicek.half_life:.1f} years)")
    print(f"    θ (long-run mean):   {vasicek.theta*100:.3f}%")
    print(f"    σ (volatility):      {vasicek.sigma*100:.3f}%")
    print(f"    r₀ (current rate):   {vasicek.r0*100:.3f}%")
    print(f"    Log-Likelihood:      {vasicek.log_likelihood:.2f}")
    print(f"    AIC: {vasicek.aic:.2f}  |  BIC: {vasicek.bic:.2f}")

    # Model-implied curve
    model_mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    vasicek_curve = vasicek.yield_curve(model_mats)
    print(f"\n    Model-Implied Yield Curve:")
    for m, y in zip(model_mats, vasicek_curve):
        print(f"      {m:>5.1f}Y: {y:.3f}%")

    # CIR
    cir = calibrator.fit_cir(short_rates, dt=1/252)
    results['cir_result'] = cir

    print(f"\n  CIR Model (dr = κ(θ-r)dt + σ√r dW):")
    print(f"    κ:                   {cir.kappa:.4f} (half-life: {cir.half_life:.1f} years)")
    print(f"    θ:                   {cir.theta*100:.3f}%")
    print(f"    σ:                   {cir.sigma*100:.3f}%")
    print(f"    Feller (2κθ > σ²):   {'✓ Satisfied' if cir.feller_satisfied else '✗ VIOLATED'}")
    print(f"    AIC: {cir.aic:.2f}  |  BIC: {cir.bic:.2f}")
    print(f"\n    Model Comparison: {'Vasicek' if vasicek.aic < cir.aic else 'CIR'} preferred (lower AIC)")

    # ══════════════════════════════════════════════════════════
    # 6. BOND PORTFOLIO ANALYTICS
    # ══════════════════════════════════════════════════════════
    print_section("6. BOND PORTFOLIO ANALYTICS")

    calc = BondCalculator()

    # Create representative Treasury bonds
    bonds = [
        Bond("UST 2Y",  coupon_rate=0.04,  maturity=2.0),
        Bond("UST 3Y",  coupon_rate=0.04,  maturity=3.0),
        Bond("UST 5Y",  coupon_rate=0.04,  maturity=5.0),
        Bond("UST 7Y",  coupon_rate=0.04,  maturity=7.0),
        Bond("UST 10Y", coupon_rate=0.04,  maturity=10.0),
        Bond("UST 20Y", coupon_rate=0.045, maturity=20.0),
        Bond("UST 30Y", coupon_rate=0.045, maturity=30.0),
    ]

    # Get yields at matching maturities
    bond_yields = []
    for bond in bonds:
        nearest_mat = maturities[np.argmin(np.abs(maturities - bond.maturity))]
        bond_yields.append(latest_yields[np.where(maturities == nearest_mat)[0][0]] / 100)

    analytics_list = []
    print(f"\n  {'Bond':<12s} {'Price':>8s} {'YTM':>7s} {'MacDur':>8s} {'ModDur':>8s} {'Convex':>8s} {'DV01':>7s} {'Carry':>7s}")
    print(f"  {'─'*75}")
    for bond, ytm in zip(bonds, bond_yields):
        a = calc.full_analytics(bond, ytm)
        analytics_list.append(a)
        print(f"  {a.name:<12s} {a.clean_price:>7.3f} {a.ytm*100:>6.3f}% "
              f"{a.macaulay_duration:>7.3f} {a.modified_duration:>7.3f} "
              f"{a.convexity:>7.2f} {a.dv01:>6.4f} {a.carry_3m:>+6.1f}")

    results['bond_analytics'] = analytics_list

    # Duration/Convexity Taylor expansion test
    print(f"\n  Price Sensitivity Analysis (UST 10Y, ±100bp shift):")
    bond_10y = bonds[4]
    ytm_10y = bond_yields[4]
    for dy_bps in [-100, -50, -25, 25, 50, 100]:
        dy = dy_bps / 10000
        pc = calc.price_change(bond_10y, ytm_10y, dy)
        print(f"    Δy = {dy_bps:>+4d}bp: Actual={pc['actual_pct']:>+7.3f}%  "
              f"Dur={pc['duration_approx_pct']:>+7.3f}%  "
              f"Dur+Conv={pc['dur_conv_approx_pct']:>+7.3f}%  "
              f"Error={pc['dur_conv_error_bps']:>+5.1f}bp")

    # ══════════════════════════════════════════════════════════
    # 7. BUTTERFLY & BARBELL TRADES
    # ══════════════════════════════════════════════════════════
    print_section("7. BUTTERFLY & BARBELL TRADE ANALYSIS")

    fly_analyzer = ButterflyAnalyzer()

    # 2s5s10s butterfly
    fly_mats = (2.0, 5.0, 10.0)
    fly_yields_idx = [np.argmin(np.abs(maturities - m)) for m in fly_mats]
    fly_yields = tuple(latest_yields[i] / 100 for i in fly_yields_idx)

    fly = fly_analyzer.construct_butterfly(fly_mats, fly_yields, method='duration_neutral')

    print(f"\n  2s5s10s Duration-Neutral Butterfly:")
    print(f"    Short Wing (2Y):   {fly.short_weight:>+8.2f} notional")
    print(f"    Body (5Y):         {fly.body_weight:>+8.2f} notional")
    print(f"    Long Wing (10Y):   {fly.long_weight:>+8.2f} notional")
    print(f"    Net DV01:          {fly.net_dv01:>+8.6f} (target: 0)")
    print(f"    Net Carry:         {fly.carry_bps:>+8.2f} bps")
    print(f"    Convexity Pickup:  {fly.convexity_pickup:>+8.4f}")

    # Scenario analysis
    fly_scenarios = {
        'Parallel +50bp':       (0.005, 0.005, 0.005),
        'Parallel -50bp':       (-0.005, -0.005, -0.005),
        'Bull Steepener':       (-0.005, -0.003, 0.0),
        'Bear Flattener':       (0.005, 0.003, 0.0),
        'Belly Cheapens +25bp': (0.0, 0.0025, 0.0),
        'Belly Richens -25bp':  (0.0, -0.0025, 0.0),
        'Wings Widen':          (-0.002, 0.0, -0.002),
        'Twist (2s up, 10s dn)':(0.005, 0.0, -0.005),
        'Vol Spike (convex)':   (0.01, 0.005, 0.01),
    }
    scenario_df = fly_analyzer.scenario_analysis(fly, fly_yields, fly_scenarios)
    results['butterfly_scenarios'] = scenario_df

    print(f"\n  Scenario Analysis:")
    print(f"  {'Scenario':<25s} {'Δ2Y':>6s} {'Δ5Y':>6s} {'Δ10Y':>6s} {'P&L':>10s}")
    print(f"  {'─'*53}")
    for idx, row in scenario_df.iterrows():
        print(f"  {idx:<25s} {row['Δy_short']:>+5.1f} {row['Δy_body']:>+5.1f} "
              f"{row['Δy_long']:>+5.1f} ${row['Total_PnL']:>+8.3f}")

    # Barbell vs Bullet
    bb_analyzer = BarbellBulletAnalyzer()
    bb_result = bb_analyzer.compare(
        bonds[0], bonds[2], bonds[4],  # 2Y, 5Y, 10Y
        bond_yields[0], bond_yields[2], bond_yields[4]
    )

    print(f"\n  Barbell (2Y+10Y) vs Bullet (5Y):")
    print(f"    {'Metric':<25s} {'Barbell':>10s} {'Bullet':>10s} {'Diff':>10s}")
    print(f"    {'─'*55}")
    bb = bb_result['barbell']
    bu = bb_result['bullet']
    co = bb_result['comparison']
    print(f"    {'Duration':.<25s} {bb['duration']:>9.3f} {bu['duration']:>9.3f} {bb['duration']-bu['duration']:>+9.3f}")
    print(f"    {'Convexity':.<25s} {bb['convexity']:>9.2f} {bu['convexity']:>9.2f} {co['convexity_advantage']:>+9.2f}")
    print(f"    {'YTM (%)':.<25s} {bb['ytm']:>9.3f} {bu['ytm']:>9.3f} {bb['ytm']-bu['ytm']:>+9.3f}")
    print(f"    {'Carry 3M (bps)':.<25s} {bb['carry_3m_bps']:>9.1f} {bu['carry_3m_bps']:>9.1f} {co['carry_difference']:>+9.1f}")

    # ══════════════════════════════════════════════════════════
    # 8. TRADING SIGNALS
    # ══════════════════════════════════════════════════════════
    print_section("8. CURVE TRADING SIGNALS")

    sig_engine = CurveTradingSignals()
    signals = sig_engine.generate_all(yields_df)
    results['signals'] = signals

    print(f"\n  Latest Signals ({signals.index[-1].strftime('%Y-%m-%d')}):")
    for col in signals.columns:
        val = signals[col].iloc[-1]
        if 'regime' in col:
            regime_map = {1: 'STEEPENING', -1: 'FLATTENING', 0: 'NEUTRAL'}
            print(f"    {col:.<35s} {regime_map.get(int(val), 'NEUTRAL')}")
        else:
            direction = '↑' if val > 0.5 else '↓' if val < -0.5 else '→'
            print(f"    {col:.<35s} {val:>+7.3f} {direction}")

    # ══════════════════════════════════════════════════════════
    # 9. VISUALIZATION
    # ══════════════════════════════════════════════════════════
    print_section("9. GENERATING DASHBOARD")

    plot_dir = "plots"
    saved_plots = create_all_plots(results, output_dir=plot_dir)
    print(f"  ✓ {len(saved_plots)} plots saved to {plot_dir}/")

    # Save data exports
    yields_df.to_csv("yield_data.csv")
    print(f"  ✓ Yield data CSV: yield_data.csv")

    if not signals.empty:
        signals.to_csv("trading_signals.csv")
        print(f"  ✓ Trading signals: trading_signals.csv")

    # ══════════════════════════════════════════════════════════
    # REPORT SUMMARY
    # ══════════════════════════════════════════════════════════
    elapsed = time.time() - start_time

    report = []
    report.append("=" * 70)
    report.append("  YIELD CURVE LAB — RESEARCH SUMMARY")
    report.append("=" * 70)
    report.append(f"\n  Data:  {len(yields_df):,} days, {len(yields_df.columns)} maturities")
    report.append(f"  Period: {yields_df.index[0].strftime('%Y-%m-%d')} → {yields_df.index[-1].strftime('%Y-%m-%d')}")
    report.append(f"\n  Curve Shape:")
    report.append(f"    2s10s Spread:   {latest_yields[np.argmin(np.abs(maturities-10))] - latest_yields[np.argmin(np.abs(maturities-2))]:.0f} bp")
    report.append(f"    2s5s10s Fly:    {(latest_yields[np.argmin(np.abs(maturities-2))] + latest_yields[np.argmin(np.abs(maturities-10))])/2 - latest_yields[np.argmin(np.abs(maturities-5))]:.0f} bp")
    report.append(f"\n  Nelson-Siegel:    R²={ns_result.r_squared:.6f}, RMSE={ns_result.rmse:.1f}bp")
    report.append(f"  NS-Svensson:      R²={nss_result.r_squared:.6f}, RMSE={nss_result.rmse:.1f}bp")
    report.append(f"\n  PCA (3 factors):  {pca_result.cumulative_variance[2]:.1%} variance explained")
    report.append(f"    PC1 (Level):    {pca_result.variance_explained[0]:.1%}")
    report.append(f"    PC2 (Slope):    {pca_result.variance_explained[1]:.1%}")
    report.append(f"    PC3 (Curvature):{pca_result.variance_explained[2]:.1%}")
    report.append(f"\n  Short-Rate:       {'Vasicek' if vasicek.aic < cir.aic else 'CIR'} preferred")
    report.append(f"    Vasicek θ:      {vasicek.theta*100:.3f}% (half-life {vasicek.half_life:.1f}y)")
    report.append(f"    CIR θ:          {cir.theta*100:.3f}% (Feller: {'✓' if cir.feller_satisfied else '✗'})")
    report.append(f"\n  Runtime: {elapsed:.1f}s")
    report.append("=" * 70)

    report_text = "\n".join(report)
    print("\n" + report_text)

    with open("analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  ✓ Report saved: analysis_report.txt")

    print("\n" + "▓" * 70)
    print("  PIPELINE COMPLETE")
    print("▓" * 70 + "\n")


if __name__ == "__main__":
    main()
