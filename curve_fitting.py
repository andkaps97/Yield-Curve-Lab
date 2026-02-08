"""
Yield Curve Bootstrapping & Parametric Fitting
================================================

Implements:
1. Piecewise-linear bootstrapping of zero-coupon (spot) rates from par yields
2. Nelson-Siegel (1987) parametric curve fitting
3. Nelson-Siegel-Svensson (1994) extended model with second hump
4. Forward rate extraction from fitted curves
5. Discount factor and zero-rate computation

Theory — Nelson-Siegel Model:
    y(τ) = β₁ + β₂ · [(1 - e^{-λτ}) / (λτ)]
              + β₃ · [(1 - e^{-λτ}) / (λτ) - e^{-λτ}]

    where:
        β₁ = long-run level (asymptotic yield as τ → ∞)
        β₂ = slope (short-end loading, decays with maturity)
        β₃ = curvature (hump/trough shape)
        λ  = decay rate (controls where hump peaks)

    Economic interpretation:
        β₁ ≈ long-term rate expectation
        β₂ ≈ -(term premium slope)
        β₃ ≈ curvature / butterfly risk
        β₁ + β₂ = instantaneous short rate as τ → 0

Nelson-Siegel-Svensson Extension:
    y(τ) = NS(τ) + β₄ · [(1 - e^{-λ₂τ}) / (λ₂τ) - e^{-λ₂τ}]

    Adds a second curvature term with independent decay λ₂,
    allowing better fit to the belly of the curve.

References:
    - Nelson & Siegel (1987): "Parsimonious Modeling of Yield Curves", J. Business
    - Svensson (1994): "Estimating and Interpreting Forward Interest Rates"
    - Gürkaynak, Sack, Wright (2007): "The U.S. Treasury Yield Curve"
    - BIS (2005): "Zero-coupon yield curves: technical documentation"

Author: Andreas Kapsalis
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import CubicSpline
from numba import njit
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# Numba-Accelerated Nelson-Siegel Kernels
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def _ns_yield(tau: np.ndarray, beta1: float, beta2: float,
              beta3: float, lam: float) -> np.ndarray:
    """
    Nelson-Siegel yield curve function (vectorized).

    y(τ) = β₁ + β₂·[(1-e^{-λτ})/(λτ)] + β₃·[(1-e^{-λτ})/(λτ) - e^{-λτ}]
    """
    n = len(tau)
    result = np.empty(n)
    for i in range(n):
        t = tau[i]
        if t < 1e-10:
            result[i] = beta1 + beta2
        else:
            exp_lt = np.exp(-lam * t)
            factor1 = (1.0 - exp_lt) / (lam * t)
            factor2 = factor1 - exp_lt
            result[i] = beta1 + beta2 * factor1 + beta3 * factor2
    return result


@njit(cache=True)
def _nss_yield(tau: np.ndarray, beta1: float, beta2: float,
               beta3: float, beta4: float,
               lam1: float, lam2: float) -> np.ndarray:
    """
    Nelson-Siegel-Svensson yield curve function (vectorized).

    y(τ) = β₁ + β₂·L₂(λ₁,τ) + β₃·L₃(λ₁,τ) + β₄·L₃(λ₂,τ)
    """
    n = len(tau)
    result = np.empty(n)
    for i in range(n):
        t = tau[i]
        if t < 1e-10:
            result[i] = beta1 + beta2
        else:
            exp_l1t = np.exp(-lam1 * t)
            exp_l2t = np.exp(-lam2 * t)
            f1 = (1.0 - exp_l1t) / (lam1 * t)
            f2 = f1 - exp_l1t
            f3 = (1.0 - exp_l2t) / (lam2 * t) - exp_l2t
            result[i] = beta1 + beta2 * f1 + beta3 * f2 + beta4 * f3
    return result


@njit(cache=True)
def _ns_forward(tau: np.ndarray, beta1: float, beta2: float,
                beta3: float, lam: float) -> np.ndarray:
    """
    Nelson-Siegel instantaneous forward rate.

    f(τ) = β₁ + β₂·e^{-λτ} + β₃·λτ·e^{-λτ}
    """
    n = len(tau)
    result = np.empty(n)
    for i in range(n):
        t = tau[i]
        exp_lt = np.exp(-lam * t)
        result[i] = beta1 + beta2 * exp_lt + beta3 * lam * t * exp_lt
    return result


# ─────────────────────────────────────────────────────────────
# Curve Fitting Results
# ─────────────────────────────────────────────────────────────

@dataclass
class CurveFitResult:
    """Container for parametric curve fit results."""
    model: str                      # 'NS' or 'NSS'
    params: Dict[str, float]        # Fitted parameters
    rmse: float                     # Root mean squared error (bps)
    max_error: float                # Maximum absolute error (bps)
    r_squared: float                # Goodness of fit
    maturities_fit: np.ndarray      # Input maturities
    yields_fit: np.ndarray          # Input yields
    yields_model: np.ndarray        # Model-fitted yields
    residuals: np.ndarray           # Fit residuals

    def yield_at(self, tau: np.ndarray) -> np.ndarray:
        """Evaluate fitted curve at arbitrary maturities."""
        p = self.params
        if self.model == 'NS':
            return _ns_yield(tau, p['beta1'], p['beta2'], p['beta3'], p['lambda'])
        else:
            return _nss_yield(tau, p['beta1'], p['beta2'], p['beta3'],
                            p['beta4'], p['lambda1'], p['lambda2'])

    def forward_at(self, tau: np.ndarray) -> np.ndarray:
        """Evaluate instantaneous forward rate at given maturities."""
        p = self.params
        if self.model == 'NS':
            return _ns_forward(tau, p['beta1'], p['beta2'], p['beta3'], p['lambda'])
        else:
            # Numerical forward for NSS
            dt = 0.001
            y1 = self.yield_at(tau)
            y2 = self.yield_at(tau + dt)
            return y1 + tau * (y2 - y1) / dt

    @property
    def short_rate(self) -> float:
        """Instantaneous short rate (τ → 0)."""
        return self.params['beta1'] + self.params['beta2']

    @property
    def long_rate(self) -> float:
        """Asymptotic long rate (τ → ∞)."""
        return self.params['beta1']


# ─────────────────────────────────────────────────────────────
# Yield Curve Bootstrapper
# ─────────────────────────────────────────────────────────────

class YieldCurveBootstrapper:
    """
    Bootstrap zero-coupon (spot) rates from par Treasury yields.

    Par yields assume coupon = yield, so the par price is always 100.
    We iteratively solve for discount factors:

        100 = Σ (c/2) · d(tᵢ) + 100 · d(T)

    where c is the par coupon rate and d(t) is the discount factor.

    For bills (τ ≤ 1Y): simple zero-rate conversion
    For notes/bonds (τ > 1Y): iterative bootstrap stripping

    Usage:
        >>> bootstrapper = YieldCurveBootstrapper()
        >>> zeros, discounts, forwards = bootstrapper.bootstrap(maturities, par_yields)
    """

    def bootstrap(
        self, maturities: np.ndarray, par_yields: np.ndarray,
        freq: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bootstrap zero rates from par yields.

        Args:
            maturities: Array of maturities in years
            par_yields: Array of par yields in percent (e.g., 4.5 = 4.5%)
            freq:       Coupon frequency (2 = semi-annual)

        Returns:
            (zero_rates, discount_factors, forward_rates)
            All rates in percent terms.
        """
        n = len(maturities)
        zero_rates = np.zeros(n)
        discount_factors = np.zeros(n)

        # Sort by maturity
        sort_idx = np.argsort(maturities)
        mat_sorted = maturities[sort_idx]
        par_sorted = par_yields[sort_idx] / 100.0  # Convert to decimal

        for i in range(n):
            T = mat_sorted[i]
            c = par_sorted[i]  # Par coupon rate (decimal)

            if T <= 1.0:
                # Bills: simple zero-rate
                zero_rates[i] = c * 100.0
                discount_factors[i] = 1.0 / (1.0 + c * T)
            else:
                # Notes/Bonds: strip coupons
                coupon = c / freq  # Periodic coupon rate
                n_periods = int(T * freq)

                # Sum of PV of coupons using already-bootstrapped zeros
                pv_coupons = 0.0
                for j in range(1, n_periods):
                    t_j = j / freq
                    # Interpolate discount factor at coupon date
                    df_j = self._interpolate_df(
                        t_j, mat_sorted[:i], discount_factors[:i]
                    )
                    pv_coupons += coupon * 100.0 * df_j

                # Solve for discount factor at maturity T
                # 100 = PV(coupons) + (coupon + 1) * 100 * d(T)
                discount_factors[i] = (100.0 - pv_coupons) / ((1.0 + coupon) * 100.0)

                # Convert to continuously compounded zero rate
                if discount_factors[i] > 0:
                    zero_rates[i] = -np.log(discount_factors[i]) / T * 100.0
                else:
                    zero_rates[i] = par_sorted[i] * 100.0

        # Compute forward rates: f(t₁,t₂) = [z(t₂)·t₂ - z(t₁)·t₁] / (t₂ - t₁)
        forward_rates = np.zeros(n)
        forward_rates[0] = zero_rates[0]
        for i in range(1, n):
            dt = mat_sorted[i] - mat_sorted[i-1]
            if dt > 0:
                forward_rates[i] = (
                    (zero_rates[i] * mat_sorted[i] - zero_rates[i-1] * mat_sorted[i-1]) / dt
                )
            else:
                forward_rates[i] = zero_rates[i]

        # Unsort back to original order
        unsort_idx = np.argsort(sort_idx)
        return zero_rates[unsort_idx], discount_factors[unsort_idx], forward_rates[unsort_idx]

    def _interpolate_df(
        self, t: float, maturities: np.ndarray, dfs: np.ndarray
    ) -> float:
        """Interpolate discount factor at time t using log-linear interpolation."""
        if len(maturities) == 0:
            return 1.0

        if t <= maturities[0]:
            # Extrapolate flat from first point
            if maturities[0] > 0:
                rate = -np.log(dfs[0]) / maturities[0]
                return np.exp(-rate * t)
            return 1.0

        if t >= maturities[-1]:
            rate = -np.log(dfs[-1]) / maturities[-1]
            return np.exp(-rate * t)

        # Log-linear interpolation
        for i in range(len(maturities) - 1):
            if maturities[i] <= t <= maturities[i+1]:
                w = (t - maturities[i]) / (maturities[i+1] - maturities[i])
                log_df = (1 - w) * np.log(max(dfs[i], 1e-10)) + w * np.log(max(dfs[i+1], 1e-10))
                return np.exp(log_df)

        return 1.0


# ─────────────────────────────────────────────────────────────
# Nelson-Siegel Fitter
# ─────────────────────────────────────────────────────────────

class NelsonSiegelFitter:
    """
    Fit Nelson-Siegel and Nelson-Siegel-Svensson models to yield data.

    Optimization strategy:
    1. Grid search over λ to find good starting point (λ is the only
       nonlinear parameter — given λ, betas are linear OLS)
    2. Full nonlinear optimization from best grid point
    3. Optional differential evolution for global optimum

    This two-stage approach is standard in central bank implementations
    (ECB, BoE, Bundesbank all use similar procedures).

    Usage:
        >>> fitter = NelsonSiegelFitter()
        >>> result = fitter.fit_ns(maturities, yields)
        >>> print(f"β₁={result.params['beta1']:.3f}, RMSE={result.rmse:.2f}bp")
    """

    def fit_ns(
        self, maturities: np.ndarray, yields: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> CurveFitResult:
        """
        Fit Nelson-Siegel model to observed yields.

        Args:
            maturities: Array of maturities in years
            yields:     Array of yields in percent
            weights:    Optional fitting weights (e.g., inverse duration)

        Returns:
            CurveFitResult with fitted parameters and diagnostics
        """
        tau = maturities.astype(np.float64)
        y_obs = yields.astype(np.float64)

        if weights is None:
            weights = np.ones_like(y_obs)

        # Stage 1: Grid search over λ, OLS for betas
        best_rmse = np.inf
        best_params = None

        for lam in np.linspace(0.1, 5.0, 100):
            betas = self._ols_betas_ns(tau, y_obs, lam, weights)
            y_hat = _ns_yield(tau, betas[0], betas[1], betas[2], lam)
            rmse = np.sqrt(np.mean(weights * (y_obs - y_hat)**2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (*betas, lam)

        # Stage 2: Nonlinear refinement
        def objective(params):
            b1, b2, b3, l = params
            if l <= 0.01:
                return 1e10
            y_hat = _ns_yield(tau, b1, b2, b3, l)
            return np.sum(weights * (y_obs - y_hat)**2)

        res = minimize(
            objective, best_params,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10}
        )

        b1, b2, b3, lam = res.x
        y_model = _ns_yield(tau, b1, b2, b3, lam)
        residuals = y_obs - y_model

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return CurveFitResult(
            model='NS',
            params={'beta1': b1, 'beta2': b2, 'beta3': b3, 'lambda': lam},
            rmse=np.sqrt(np.mean(residuals**2)) * 100,  # In basis points
            max_error=np.max(np.abs(residuals)) * 100,
            r_squared=r_sq,
            maturities_fit=tau,
            yields_fit=y_obs,
            yields_model=y_model,
            residuals=residuals,
        )

    def fit_nss(
        self, maturities: np.ndarray, yields: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> CurveFitResult:
        """
        Fit Nelson-Siegel-Svensson model.

        Uses differential evolution for robust global optimization
        of the 6-parameter model.
        """
        tau = maturities.astype(np.float64)
        y_obs = yields.astype(np.float64)

        if weights is None:
            weights = np.ones_like(y_obs)

        # First fit NS to get good starting point for first 4 params
        ns_result = self.fit_ns(maturities, yields, weights)
        ns_p = ns_result.params

        def objective(params):
            b1, b2, b3, b4, l1, l2 = params
            if l1 <= 0.01 or l2 <= 0.01:
                return 1e10
            y_hat = _nss_yield(tau, b1, b2, b3, b4, l1, l2)
            return np.sum(weights * (y_obs - y_hat)**2)

        # Bounds for differential evolution
        bounds = [
            (0.0, 15.0),       # beta1: long rate
            (-10.0, 10.0),     # beta2: slope
            (-10.0, 10.0),     # beta3: curvature 1
            (-10.0, 10.0),     # beta4: curvature 2
            (0.05, 5.0),       # lambda1
            (0.05, 5.0),       # lambda2
        ]

        # Differential evolution with seeded initial
        res = differential_evolution(
            objective, bounds,
            seed=42, maxiter=500, tol=1e-10, polish=True,
            init='sobol',
        )

        b1, b2, b3, b4, l1, l2 = res.x
        y_model = _nss_yield(tau, b1, b2, b3, b4, l1, l2)
        residuals = y_obs - y_model

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return CurveFitResult(
            model='NSS',
            params={
                'beta1': b1, 'beta2': b2, 'beta3': b3, 'beta4': b4,
                'lambda1': l1, 'lambda2': l2,
            },
            rmse=np.sqrt(np.mean(residuals**2)) * 100,
            max_error=np.max(np.abs(residuals)) * 100,
            r_squared=r_sq,
            maturities_fit=tau,
            yields_fit=y_obs,
            yields_model=y_model,
            residuals=residuals,
        )

    def _ols_betas_ns(
        self, tau: np.ndarray, y: np.ndarray, lam: float,
        weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """Solve for NS betas via weighted OLS given fixed lambda."""
        n = len(tau)
        X = np.zeros((n, 3))
        for i in range(n):
            t = tau[i]
            if t < 1e-10:
                X[i, 0] = 1.0
                X[i, 1] = 1.0
                X[i, 2] = 0.0
            else:
                exp_lt = np.exp(-lam * t)
                X[i, 0] = 1.0
                X[i, 1] = (1.0 - exp_lt) / (lam * t)
                X[i, 2] = X[i, 1] - exp_lt

        W = np.diag(weights)
        try:
            betas = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        except np.linalg.LinAlgError:
            betas = np.linalg.lstsq(X, y, rcond=None)[0]

        return (betas[0], betas[1], betas[2])

    def fit_time_series(
        self, yields_df, model: str = 'NS'
    ) -> Dict:
        """
        Fit NS/NSS to each date in a yield DataFrame.

        Returns:
            Dict with 'params_df' (time series of parameters),
            'rmse_series', and 'fit_results' list
        """
        maturities = np.array(yields_df.columns, dtype=float)
        params_list = []
        rmse_list = []
        dates = []

        for date, row in yields_df.iterrows():
            y = row.values.astype(float)
            if np.any(np.isnan(y)):
                continue

            if model == 'NS':
                result = self.fit_ns(maturities, y)
            else:
                result = self.fit_nss(maturities, y)

            params_list.append(result.params)
            rmse_list.append(result.rmse)
            dates.append(date)

        params_df = pd.DataFrame(params_list, index=dates)
        params_df['rmse_bps'] = rmse_list

        return {
            'params_df': params_df,
            'rmse_mean': np.mean(rmse_list),
            'rmse_max': np.max(rmse_list),
        }


# Need pandas for fit_time_series
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Forward Rate Calculator
# ─────────────────────────────────────────────────────────────

class ForwardRateCalculator:
    """
    Extract forward rates from fitted yield curves.

    Computes:
    - Instantaneous forward rates: f(t) = y(t) + t·y'(t)
    - Discrete forward rates: f(t₁,t₂)
    - Par forward swap rates
    - Forward curve term structure
    """

    @staticmethod
    def instantaneous_forward(
        fit_result: CurveFitResult, tau: np.ndarray
    ) -> np.ndarray:
        """Compute instantaneous forward rate f(τ) from fitted curve."""
        return fit_result.forward_at(tau)

    @staticmethod
    def discrete_forward(
        fit_result: CurveFitResult, t1: float, t2: float
    ) -> float:
        """
        Compute discrete forward rate f(t₁, t₂).

        f(t₁,t₂) = [y(t₂)·t₂ - y(t₁)·t₁] / (t₂ - t₁)
        """
        tau = np.array([t1, t2])
        y = fit_result.yield_at(tau)
        return (y[1] * t2 - y[0] * t1) / (t2 - t1)

    @staticmethod
    def forward_curve(
        fit_result: CurveFitResult,
        tenors: np.ndarray,
        forward_start: float = 1.0,
    ) -> np.ndarray:
        """
        Compute forward rates starting at forward_start for each tenor.

        E.g., 1y1y, 1y2y, 1y5y, 1y10y forward rates.
        """
        forwards = np.zeros(len(tenors))
        for i, tenor in enumerate(tenors):
            t1 = forward_start
            t2 = forward_start + tenor
            tau = np.array([t1, t2])
            y = fit_result.yield_at(tau)
            forwards[i] = (y[1] * t2 - y[0] * t1) / tenor
        return forwards

    @staticmethod
    def term_structure_of_forwards(
        fit_result: CurveFitResult,
        max_maturity: float = 30.0,
        step: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full term structure of instantaneous forward rates.

        Returns (maturities, forward_rates) for plotting.
        """
        tau = np.arange(step, max_maturity + step, step)
        fwd = fit_result.forward_at(tau)
        return tau, fwd
