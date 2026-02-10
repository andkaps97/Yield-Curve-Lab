# Yield-Curve-Lab: Fixed Income Analytics & Yield Curve Modeling

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy)
![Numba](https://img.shields.io/badge/Numba-JIT-00A3E0?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**A production-grade fixed income analytics platform with yield curve construction, parametric fitting, PCA decomposition, short-rate models, bond risk analytics, and systematic curve trading strategies.**

</div>

---

## Overview

This project implements a **complete fixed income research lab** covering every major aspect of yield curve analytics — from raw Treasury data to tradeable curve signals. Built for institutional fixed income desks, quantitative researchers, and portfolio managers who need rigorous analytical tools for curve construction, risk decomposition, and relative value trading.

### Key Capabilities

| Module | Description |
|--------|-------------|
| **Data Fetcher** | FRED API integration + calibrated synthetic yield curve generation |
| **Bootstrapping** | Par → zero-rate stripping with log-linear discount factor interpolation |
| **Nelson-Siegel** | 3-parameter curve fit with OLS + nonlinear optimization |
| **NS-Svensson** | 6-parameter extended model via differential evolution |
| **PCA** | Litterman-Scheinkman decomposition (Level, Slope, Curvature) |
| **Vasicek/CIR** | Short-rate models with MLE calibration + Monte Carlo simulation |
| **Bond Analytics** | Duration, convexity, DV01, key rate duration, carry & roll-down |
| **Butterfly Trades** | Duration-neutral construction with multi-scenario P&L analysis |
| **Barbell vs Bullet** | Convexity/carry tradeoff comparison at matched duration |
| **Trading Signals** | Slope z-score, butterfly richness, forward rate, carry momentum |

---

## Architecture

```
Yield-Curve-Lab/
├── run_analysis.py          # Main entry point — runs full pipeline
├── data_fetcher.py          # FRED API + synthetic Treasury data
├── curve_fitting.py         # Bootstrapping, Nelson-Siegel, NSS, forward rates
├── pca_analysis.py          # Litterman-Scheinkman PCA decomposition
├── short_rate_models.py     # Vasicek & CIR with MLE calibration
├── bond_analytics.py        # Duration, convexity, DV01, portfolio risk
├── trading_strategies.py    # Butterfly/barbell trades, curve signals
├── visualization.py         # 8-panel publication-quality dashboard
├── requirements.txt
└── README.md
```

---

## Mathematical Framework

### 1. Yield Curve Bootstrapping

Par yields are converted to zero-coupon (spot) rates via iterative stripping:

$$100 = \sum_{i=1}^{n} \frac{c/m}{(1 + z_i)^{t_i}} + \frac{100}{(1 + z_n)^T}$$

Forward rates are extracted via no-arbitrage:

$$f(t_1, t_2) = \frac{y(t_2) \cdot t_2 - y(t_1) \cdot t_1}{t_2 - t_1}$$

### 2. Nelson-Siegel (1987) Parametric Model

$$y(\tau) = \beta_1 + \beta_2 \cdot \frac{1 - e^{-\lambda\tau}}{\lambda\tau} + \beta_3 \cdot \left[\frac{1 - e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\right]$$

| Parameter | Interpretation | Limit Behavior |
|-----------|---------------|----------------|
| $\beta_1$ | Long-run level | $y(\infty) = \beta_1$ |
| $\beta_2$ | Slope | $y(0) = \beta_1 + \beta_2$ |
| $\beta_3$ | Curvature (hump) | Medium-term effect |
| $\lambda$ | Decay rate | Controls hump location |

**Nelson-Siegel-Svensson** adds a second curvature term $\beta_4$ with independent decay $\lambda_2$.

Optimization: Two-stage approach (grid search over $\lambda$ with OLS for $\beta$s, then Nelder-Mead refinement). NSS uses differential evolution for the 6-parameter global optimum.

### 3. PCA — Litterman-Scheinkman (1991)

Eigendecomposition of the yield change covariance matrix:

$$\Sigma = V \Lambda V^T \quad \Rightarrow \quad \Delta y_t = \sum_{k=1}^{K} f_{k,t} \cdot v_k$$

The classic result: three factors explain ~99% of yield curve variance:

| Factor | Typical Variance | Shape |
|--------|-----------------|-------|
| **PC1 (Level)** | 85-90% | Flat — parallel shifts |
| **PC2 (Slope)** | 8-12% | Monotone — steepening/flattening |
| **PC3 (Curvature)** | 2-4% | Humped — butterfly movements |

Applications: risk decomposition, factor-neutral hedging, scenario generation, relative value detection.

### 4. Short-Rate Models

**Vasicek (1977):**

$$dr_t = \kappa(\theta - r_t)dt + \sigma \, dW_t$$

Closed-form bond price: $P(t,T) = A(\tau) \cdot e^{-B(\tau) \cdot r_t}$

**CIR (1985):**

$$dr_t = \kappa(\theta - r_t)dt + \sigma\sqrt{r_t} \, dW_t$$

Feller condition for non-negativity: $2\kappa\theta > \sigma^2$

Calibration via **exact MLE** using known transition densities (Gaussian for Vasicek, non-central $\chi^2$ for CIR).

### 5. Bond Analytics

$$D_{\text{mod}} = -\frac{1}{P}\frac{dP}{dy} = \frac{D_{\text{mac}}}{1 + y/m}$$

$$C = \frac{1}{P}\frac{d^2P}{dy^2}$$

$$\frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2$$

$$\text{DV01} = D_{\text{mod}} \cdot P / 10{,}000$$

### 6. Butterfly Trades

A duration-neutral butterfly: **long wings, short body**:

$$\sum_{i} w_i \cdot \text{DV01}_i = 0 \quad \text{(duration-neutral constraint)}$$

Profits when curvature increases (belly cheapens vs wings). The barbell (wings only) vs bullet (body only) comparison isolates the convexity-carry tradeoff.

---

## Quick Start

```python
python run_analysis.py
```

Or use individual modules:

```python
from data_fetcher import TreasuryDataFetcher
from curve_fitting import NelsonSiegelFitter, YieldCurveBootstrapper
from pca_analysis import YieldCurvePCA
from short_rate_models import ShortRateCalibrator
from bond_analytics import Bond, BondCalculator

# Fetch data
fetcher = TreasuryDataFetcher()
yields_df = fetcher.fetch(start="2020-01-01")
maturities, latest = fetcher.get_latest_curve()

# Fit Nelson-Siegel
fitter = NelsonSiegelFitter()
ns = fitter.fit_ns(maturities, latest)
print(f"β₁={ns.params['beta1']:.3f}, R²={ns.r_squared:.6f}")

# PCA decomposition
pca = YieldCurvePCA()
result = pca.decompose(yields_df)
print(f"PC1 explains {result.variance_explained[0]:.1%}")

# Bond analytics
calc = BondCalculator()
bond = Bond("UST 10Y", coupon_rate=0.04, maturity=10)
analytics = calc.full_analytics(bond, ytm=0.045)
print(f"Duration={analytics.modified_duration:.2f}, DV01={analytics.dv01:.4f}")
```

---

## Sample Output (Excerpt)

```
Nelson-Siegel Fit:
  β₁ (level):       3.5405  (long-run rate)
  β₂ (slope):       1.4381
  β₃ (curvature):   0.0000
  RMSE:               1.78 bp
  R²:             0.998455

PCA — Litterman-Scheinkman:
  PC1 (Level):      19.0%    ← dominant factor
  PC2 (Slope):       9.7%
  PC3 (Curvature):   9.1%

Vasicek:  κ=0.738, θ=5.34%, σ=0.49% (half-life 0.9y)
CIR:      κ=0.736, θ=5.34%, Feller: ✓

2s5s10s Butterfly Scenario P&L:
  Belly Cheapens +25bp:  +$112.46
  Belly Richens -25bp:   -$113.95
  Parallel +50bp:        +$0.44  (duration-neutral!)
```

---

## References

1. **Nelson, C.R. & Siegel, A.F.** (1987). "Parsimonious Modeling of Yield Curves." *Journal of Business*, 60(4), 473-489.
2. **Svensson, L.E.O.** (1994). "Estimating and Interpreting Forward Interest Rates." NBER Working Paper No. 4871.
3. **Litterman, R. & Scheinkman, J.** (1991). "Common Factors Affecting Bond Returns." *Journal of Fixed Income*, 1(1), 54-61.
4. **Vasicek, O.** (1977). "An Equilibrium Characterization of the Term Structure." *Journal of Financial Economics*, 5(2), 177-188.
5. **Cox, J.C., Ingersoll, J.E. & Ross, S.A.** (1985). "A Theory of the Term Structure of Interest Rates." *Econometrica*, 53(2), 385-407.
6. **Gürkaynak, R.S., Sack, B. & Wright, J.H.** (2007). "The U.S. Treasury Yield Curve: 1961 to the Present." *Journal of Monetary Economics*, 54(8), 2291-2304.
7. **Tuckman, B. & Serrat, A.** (2012). *Fixed Income Securities*, 3rd ed. Wiley.
8. **Ilmanen, A.** (2011). *Expected Returns*. Wiley. Chapters 6-8.
9. **Fabozzi, F.J.** (2007). *Fixed Income Analysis*, 2nd ed. CFA Institute.
10. **Brigo, D. & Mercurio, F.** (2006). *Interest Rate Models — Theory and Practice*. Springer.

---


---

## License

MIT License
