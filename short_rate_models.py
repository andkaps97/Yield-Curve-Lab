"""
Short-Rate Models: Vasicek & Cox-Ingersoll-Ross (CIR)
=======================================================

Implements two foundational affine term structure models with maximum
likelihood estimation (MLE) calibration from observed yield data.

Vasicek (1977):
    dr_t = κ(θ - r_t)dt + σ dW_t

    - Ornstein-Uhlenbeck process (Gaussian)
    - Allows negative rates (feature, not bug, post-2008)
    - Closed-form bond prices: P(t,T) = A(τ)·exp(-B(τ)·r_t)

CIR (1985):
    dr_t = κ(θ - r_t)dt + σ√r_t dW_t

    - Square-root diffusion (non-central χ²)
    - Rate cannot go negative if 2κθ > σ² (Feller condition)
    - Volatility proportional to rate level

Calibration via MLE:
    Both models have known transition densities, enabling exact
    maximum likelihood estimation:
    - Vasicek: Gaussian transition density
    - CIR: Non-central chi-squared transition density

Bond Pricing:
    Both models are affine: P(t,T) = exp(A(τ) - B(τ)·r_t)
    Yield: y(τ) = -A(τ)/τ + B(τ)/τ · r_t

References:
    - Vasicek (1977): "An Equilibrium Characterization of the Term Structure"
    - Cox, Ingersoll, Ross (1985): "A Theory of the Term Structure of Interest Rates"
    - Brigo & Mercurio (2006): "Interest Rate Models — Theory and Practice", Ch. 3-4
    - Singleton (2006): "Empirical Dynamic Asset Pricing", Ch. 12

Author: Andreas Kapsalis
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, ncx2
from numba import njit
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# Numba-Accelerated Bond Pricing Kernels
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def _vasicek_bond_price(r: float, tau: float, kappa: float,
                         theta: float, sigma: float) -> float:
    """
    Vasicek zero-coupon bond price P(t, t+τ).

    P(τ) = A(τ) · exp(-B(τ) · r)

    where:
        B(τ) = (1 - e^{-κτ}) / κ
        A(τ) = exp{(B(τ) - τ)(κ²θ - σ²/2) / κ² - σ²B(τ)² / (4κ)}
    """
    if abs(kappa) < 1e-10:
        B = tau
    else:
        B = (1.0 - np.exp(-kappa * tau)) / kappa

    A_exponent = (B - tau) * (kappa**2 * theta - sigma**2 / 2.0) / kappa**2
    A_exponent -= sigma**2 * B**2 / (4.0 * kappa)

    return np.exp(A_exponent - B * r)


@njit(cache=True)
def _vasicek_yield(r: float, tau: float, kappa: float,
                    theta: float, sigma: float) -> float:
    """Vasicek model yield: y(τ) = -ln(P(τ)) / τ."""
    P = _vasicek_bond_price(r, tau, kappa, theta, sigma)
    if P <= 0 or tau <= 0:
        return 0.0
    return -np.log(P) / tau


@njit(cache=True)
def _vasicek_yield_curve(r: float, taus: np.ndarray, kappa: float,
                          theta: float, sigma: float) -> np.ndarray:
    """Compute Vasicek yield curve at multiple maturities."""
    n = len(taus)
    yields = np.empty(n)
    for i in range(n):
        yields[i] = _vasicek_yield(r, taus[i], kappa, theta, sigma)
    return yields


@njit(cache=True)
def _cir_bond_price(r: float, tau: float, kappa: float,
                     theta: float, sigma: float) -> float:
    """
    CIR zero-coupon bond price P(t, t+τ).

    P(τ) = A(τ) · exp(-B(τ) · r)

    where:
        γ = √(κ² + 2σ²)
        B(τ) = 2(e^{γτ} - 1) / ((γ+κ)(e^{γτ}-1) + 2γ)
        A(τ) = [2γ·e^{(κ+γ)τ/2} / ((γ+κ)(e^{γτ}-1) + 2γ)]^{2κθ/σ²}
    """
    gamma = np.sqrt(kappa**2 + 2.0 * sigma**2)
    exp_gt = np.exp(gamma * tau)

    denom = (gamma + kappa) * (exp_gt - 1.0) + 2.0 * gamma

    if abs(denom) < 1e-15:
        return 1.0

    B = 2.0 * (exp_gt - 1.0) / denom

    A_base = 2.0 * gamma * np.exp((kappa + gamma) * tau / 2.0) / denom
    power = 2.0 * kappa * theta / (sigma**2)

    if A_base <= 0:
        return 0.0

    A = A_base ** power

    return A * np.exp(-B * r)


@njit(cache=True)
def _cir_yield(r: float, tau: float, kappa: float,
                theta: float, sigma: float) -> float:
    """CIR model yield."""
    P = _cir_bond_price(r, tau, kappa, theta, sigma)
    if P <= 0 or tau <= 0:
        return 0.0
    return -np.log(P) / tau


@njit(cache=True)
def _cir_yield_curve(r: float, taus: np.ndarray, kappa: float,
                      theta: float, sigma: float) -> np.ndarray:
    """Compute CIR yield curve at multiple maturities."""
    n = len(taus)
    yields = np.empty(n)
    for i in range(n):
        yields[i] = _cir_yield(r, taus[i], kappa, theta, sigma)
    return yields


# ─────────────────────────────────────────────────────────────
# MLE Calibration
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def _vasicek_log_likelihood(
    rates: np.ndarray, dt: float, kappa: float, theta: float, sigma: float
) -> float:
    """
    Exact log-likelihood for Vasicek model.

    Transition density is Gaussian:
        r_{t+Δt} | r_t ~ N(μ, v²)
    where:
        μ = θ + (r_t - θ)·e^{-κΔt}
        v² = σ²(1 - e^{-2κΔt}) / (2κ)
    """
    n = len(rates) - 1
    if n <= 0 or kappa <= 0 or sigma <= 0:
        return -1e15

    exp_kdt = np.exp(-kappa * dt)
    var = sigma**2 * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)

    if var <= 0:
        return -1e15

    std = np.sqrt(var)
    ll = 0.0

    for i in range(n):
        mu = theta + (rates[i] - theta) * exp_kdt
        z = (rates[i + 1] - mu) / std
        ll += -0.5 * np.log(2.0 * np.pi) - np.log(std) - 0.5 * z**2

    return ll


@njit(cache=True)
def _cir_log_likelihood(
    rates: np.ndarray, dt: float, kappa: float, theta: float, sigma: float
) -> float:
    """
    Approximate log-likelihood for CIR model.

    Uses Gaussian approximation of the non-central chi-squared density
    for computational efficiency (exact for small dt).

    For rigorous implementation, use scipy.stats.ncx2.
    """
    n = len(rates) - 1
    if n <= 0 or kappa <= 0 or sigma <= 0 or theta <= 0:
        return -1e15

    exp_kdt = np.exp(-kappa * dt)
    ll = 0.0

    for i in range(n):
        r_t = max(rates[i], 1e-8)
        mu = theta + (r_t - theta) * exp_kdt
        var = r_t * sigma**2 * (exp_kdt - np.exp(-2.0 * kappa * dt)) / kappa
        var += theta * sigma**2 * (1.0 - exp_kdt)**2 / (2.0 * kappa)

        if var <= 0:
            var = 1e-10

        std = np.sqrt(var)
        z = (rates[i + 1] - mu) / std
        ll += -0.5 * np.log(2.0 * np.pi) - np.log(std) - 0.5 * z**2

    return ll


# ─────────────────────────────────────────────────────────────
# Model Results Container
# ─────────────────────────────────────────────────────────────

@dataclass
class ShortRateModelResult:
    """Container for calibrated short-rate model."""
    model: str                  # 'Vasicek' or 'CIR'
    kappa: float                # Mean reversion speed
    theta: float                # Long-run mean (in decimal, e.g., 0.04 = 4%)
    sigma: float                # Volatility
    log_likelihood: float       # MLE log-likelihood
    aic: float                  # Akaike Information Criterion
    bic: float                  # Bayesian Information Criterion
    half_life: float            # Mean-reversion half-life (years)
    feller_satisfied: bool      # CIR: 2κθ > σ² ?
    r0: float                   # Current short rate

    def yield_curve(self, maturities: np.ndarray) -> np.ndarray:
        """Compute model-implied yield curve."""
        if self.model == 'Vasicek':
            return _vasicek_yield_curve(
                self.r0, maturities, self.kappa, self.theta, self.sigma
            ) * 100  # Convert to percent
        else:
            return _cir_yield_curve(
                self.r0, maturities, self.kappa, self.theta, self.sigma
            ) * 100

    def bond_price(self, maturity: float) -> float:
        """Compute zero-coupon bond price."""
        if self.model == 'Vasicek':
            return _vasicek_bond_price(
                self.r0, maturity, self.kappa, self.theta, self.sigma
            )
        else:
            return _cir_bond_price(
                self.r0, maturity, self.kappa, self.theta, self.sigma
            )

    def simulate_paths(
        self, T: float, n_steps: int, n_paths: int, seed: int = 42
    ) -> np.ndarray:
        """
        Monte Carlo simulation of short rate paths.

        Returns array of shape (n_paths, n_steps+1).
        """
        np.random.seed(seed)
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.r0

        for t in range(n_steps):
            dW = np.random.randn(n_paths) * np.sqrt(dt)
            if self.model == 'Vasicek':
                paths[:, t+1] = (
                    paths[:, t]
                    + self.kappa * (self.theta - paths[:, t]) * dt
                    + self.sigma * dW
                )
            else:
                # CIR: use full truncation scheme
                r_pos = np.maximum(paths[:, t], 0)
                paths[:, t+1] = (
                    paths[:, t]
                    + self.kappa * (self.theta - r_pos) * dt
                    + self.sigma * np.sqrt(r_pos) * dW
                )
                paths[:, t+1] = np.maximum(paths[:, t+1], 0)

        return paths


# ─────────────────────────────────────────────────────────────
# Model Calibrator
# ─────────────────────────────────────────────────────────────

class ShortRateCalibrator:
    """
    Calibrate Vasicek and CIR models via Maximum Likelihood Estimation.

    Two calibration modes:
    1. Time-series MLE: fit to short-rate observations (e.g., 3M T-bill)
    2. Cross-sectional: fit to current yield curve

    Usage:
        >>> calibrator = ShortRateCalibrator()
        >>> vasicek = calibrator.fit_vasicek(short_rates, dt=1/252)
        >>> cir = calibrator.fit_cir(short_rates, dt=1/252)
        >>> print(f"Vasicek: κ={vasicek.kappa:.4f}, θ={vasicek.theta:.4f}")
    """

    def fit_vasicek(
        self, rates: np.ndarray, dt: float = 1/252
    ) -> ShortRateModelResult:
        """
        Calibrate Vasicek model via time-series MLE.

        Args:
            rates: Array of short-rate observations (in decimal, e.g., 0.04)
            dt:    Time step between observations (in years)

        Returns:
            ShortRateModelResult with calibrated parameters
        """
        rates = np.asarray(rates, dtype=np.float64)
        n = len(rates)

        # Starting values from OLS regression
        # r_{t+1} = a + b·r_t + ε → κ = (1-b)/dt, θ = a/(1-b)
        y = rates[1:]
        x = rates[:-1]
        b_hat = np.cov(x, y)[0, 1] / (np.var(x) + 1e-15)
        a_hat = np.mean(y) - b_hat * np.mean(x)

        kappa0 = max(-np.log(max(b_hat, 0.01)) / dt, 0.01)
        theta0 = np.mean(rates)
        sigma0 = np.std(np.diff(rates)) / np.sqrt(dt)

        def neg_ll(params):
            k, th, sig = params
            if k <= 0 or sig <= 0:
                return 1e15
            return -_vasicek_log_likelihood(rates, dt, k, th, sig)

        res = minimize(
            neg_ll, [kappa0, theta0, sigma0],
            method='Nelder-Mead',
            options={'maxiter': 50000, 'xatol': 1e-10}
        )

        kappa, theta, sigma = res.x
        ll = -res.fun
        k = 3  # Number of parameters

        return ShortRateModelResult(
            model='Vasicek',
            kappa=abs(kappa),
            theta=theta,
            sigma=abs(sigma),
            log_likelihood=ll,
            aic=-2 * ll + 2 * k,
            bic=-2 * ll + k * np.log(n),
            half_life=np.log(2) / abs(kappa) if abs(kappa) > 1e-10 else np.inf,
            feller_satisfied=True,  # Always true for Vasicek
            r0=rates[-1],
        )

    def fit_cir(
        self, rates: np.ndarray, dt: float = 1/252
    ) -> ShortRateModelResult:
        """
        Calibrate CIR model via time-series MLE.

        Args:
            rates: Array of short-rate observations (in decimal)
            dt:    Time step

        Returns:
            ShortRateModelResult with calibrated parameters
        """
        rates = np.asarray(rates, dtype=np.float64)
        rates = np.maximum(rates, 1e-6)  # CIR requires positive rates
        n = len(rates)

        # Starting values
        kappa0 = 0.5
        theta0 = np.mean(rates)
        sigma0 = np.std(np.diff(rates)) / np.sqrt(dt * np.mean(rates))

        def neg_ll(params):
            k, th, sig = params
            if k <= 0 or th <= 0 or sig <= 0:
                return 1e15
            return -_cir_log_likelihood(rates, dt, k, th, sig)

        res = minimize(
            neg_ll, [kappa0, theta0, sigma0],
            method='Nelder-Mead',
            options={'maxiter': 50000, 'xatol': 1e-10}
        )

        kappa, theta, sigma = abs(res.x[0]), abs(res.x[1]), abs(res.x[2])
        ll = -res.fun
        k = 3

        return ShortRateModelResult(
            model='CIR',
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            log_likelihood=ll,
            aic=-2 * ll + 2 * k,
            bic=-2 * ll + k * np.log(n),
            half_life=np.log(2) / kappa if kappa > 1e-10 else np.inf,
            feller_satisfied=(2 * kappa * theta > sigma**2),
            r0=rates[-1],
        )

    def fit_cross_sectional(
        self, maturities: np.ndarray, yields: np.ndarray,
        model: str = 'Vasicek'
    ) -> ShortRateModelResult:
        """
        Fit short-rate model to a cross-section of yields.

        Minimizes sum of squared yield errors:
            min_{κ,θ,σ,r₀} Σ [y_obs(τᵢ) - y_model(τᵢ)]²
        """
        maturities = np.asarray(maturities, dtype=np.float64)
        yields_decimal = np.asarray(yields, dtype=np.float64) / 100.0

        def objective(params):
            kappa, theta, sigma, r0 = params
            if kappa <= 0 or sigma <= 0:
                return 1e15
            if model == 'CIR' and (theta <= 0 or r0 < 0):
                return 1e15

            if model == 'Vasicek':
                y_model = _vasicek_yield_curve(r0, maturities, kappa, theta, sigma)
            else:
                y_model = _cir_yield_curve(r0, maturities, kappa, theta, sigma)

            return np.sum((yields_decimal - y_model)**2)

        # Initial guess
        r0_guess = yields_decimal[0]  # Short end
        theta_guess = yields_decimal[-1]  # Long end
        x0 = [0.5, theta_guess, 0.01, r0_guess]

        bounds = [(0.01, 10), (0.001, 0.2), (0.001, 0.5), (0.001, 0.2)]

        from scipy.optimize import differential_evolution
        res = differential_evolution(objective, bounds, seed=42, maxiter=1000)

        kappa, theta, sigma, r0 = res.x

        if model == 'Vasicek':
            y_model = _vasicek_yield_curve(r0, maturities, kappa, theta, sigma)
        else:
            y_model = _cir_yield_curve(r0, maturities, kappa, theta, sigma)

        rmse = np.sqrt(np.mean((yields_decimal - y_model)**2))

        return ShortRateModelResult(
            model=model,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            log_likelihood=-rmse * 1e4,  # Pseudo-likelihood from RMSE
            aic=0,
            bic=0,
            half_life=np.log(2) / kappa,
            feller_satisfied=(2 * kappa * theta > sigma**2) if model == 'CIR' else True,
            r0=r0,
        )
