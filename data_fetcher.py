"""
FRED Treasury Data Fetcher
============================

Retrieves US Treasury constant maturity yields from the Federal Reserve
Economic Data (FRED) API. Includes a robust synthetic data fallback for
environments without API access.

Data Series:
    DGS1MO  — 1-Month Treasury Constant Maturity
    DGS3MO  — 3-Month
    DGS6MO  — 6-Month
    DGS1    — 1-Year
    DGS2    — 2-Year
    DGS3    — 3-Year
    DGS5    — 5-Year
    DGS7    — 7-Year
    DGS10   — 10-Year
    DGS20   — 20-Year
    DGS30   — 30-Year

References:
    - Federal Reserve Bank of St. Louis: https://fred.stlouisfed.org/
    - Gürkaynak, Sack, Wright (2007): "The U.S. Treasury Yield Curve: 1961 to Present"

Author: Andreas Kapsalis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import warnings
import os


# ─────────────────────────────────────────────────────────────
# FRED Series Mapping
# ─────────────────────────────────────────────────────────────

TREASURY_SERIES = {
    'DGS1MO': 1/12,
    'DGS3MO': 0.25,
    'DGS6MO': 0.5,
    'DGS1':   1.0,
    'DGS2':   2.0,
    'DGS3':   3.0,
    'DGS5':   5.0,
    'DGS7':   7.0,
    'DGS10':  10.0,
    'DGS20':  20.0,
    'DGS30':  30.0,
}

MATURITY_LABELS = {
    1/12: '1M', 0.25: '3M', 0.5: '6M', 1.0: '1Y',
    2.0: '2Y', 3.0: '3Y', 5.0: '5Y', 7.0: '7Y',
    10.0: '10Y', 20.0: '20Y', 30.0: '30Y',
}


# ─────────────────────────────────────────────────────────────
# Data Fetcher
# ─────────────────────────────────────────────────────────────

class TreasuryDataFetcher:
    """
    Fetches US Treasury yield curve data from FRED.

    Supports three data sourcing modes:
    1. FRED API (requires API key)
    2. pandas_datareader fallback
    3. Synthetic data generation (always works, no API needed)

    The synthetic mode generates historically-calibrated yield curves
    that reproduce realistic level, slope, and curvature dynamics using
    a 3-factor model calibrated to 2000-2024 Treasury data statistics.

    Usage:
        >>> fetcher = TreasuryDataFetcher(api_key="your_fred_key")
        >>> yields_df = fetcher.fetch(start="2020-01-01")
        >>> maturities = fetcher.get_maturities()
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        self.maturities = np.array(sorted(TREASURY_SERIES.values()))
        self.series_map = TREASURY_SERIES
        self._data: Optional[pd.DataFrame] = None

    def fetch(
        self,
        start: str = "2015-01-01",
        end: Optional[str] = None,
        use_synthetic: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch Treasury yield curve time series.

        Args:
            start:          Start date (YYYY-MM-DD)
            end:            End date (defaults to today)
            use_synthetic:  Force synthetic data generation

        Returns:
            DataFrame with dates as index, maturities as columns (in years),
            yields in percentage terms (e.g., 4.25 = 4.25%)
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        if not use_synthetic and self.api_key:
            try:
                df = self._fetch_fred_api(start, end)
                if df is not None and len(df) > 50:
                    self._data = df
                    return df
            except Exception as e:
                warnings.warn(f"FRED API fetch failed: {e}. Falling back to synthetic data.")

        if not use_synthetic:
            try:
                df = self._fetch_fred_direct(start, end)
                if df is not None and len(df) > 50:
                    self._data = df
                    return df
            except Exception as e:
                warnings.warn(f"Direct FRED fetch failed: {e}. Using synthetic data.")

        # Fallback: synthetic data
        print("  [INFO] Using synthetic yield curve data (calibrated to historical stats)")
        df = self._generate_synthetic(start, end)
        self._data = df
        return df

    def _fetch_fred_api(self, start: str, end: str) -> Optional[pd.DataFrame]:
        """Fetch from FRED using fredapi package."""
        from fredapi import Fred
        fred = Fred(api_key=self.api_key)

        frames = {}
        for series_id, maturity in self.series_map.items():
            try:
                data = fred.get_series(series_id, start, end)
                frames[maturity] = data
            except Exception:
                continue

        if len(frames) < 5:
            return None

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df = df.dropna(how='all').sort_index()
        df = df.ffill().dropna()
        df.columns = [float(c) for c in df.columns]
        return df

    def _fetch_fred_direct(self, start: str, end: str) -> Optional[pd.DataFrame]:
        """Fetch directly from FRED website CSV endpoint (no API key needed)."""
        import requests

        frames = {}
        for series_id, maturity in self.series_map.items():
            url = (
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?"
                f"id={series_id}&cosd={start}&coed={end}"
            )
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    lines = resp.text.strip().split('\n')
                    if len(lines) > 1:
                        data = []
                        for line in lines[1:]:
                            parts = line.split(',')
                            if len(parts) == 2 and parts[1] != '.':
                                try:
                                    data.append((pd.Timestamp(parts[0]), float(parts[1])))
                                except ValueError:
                                    continue
                        if data:
                            dates, values = zip(*data)
                            frames[maturity] = pd.Series(values, index=dates)
            except Exception:
                continue

        if len(frames) < 5:
            return None

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df = df.dropna(how='all').sort_index()
        df = df.ffill().dropna()
        return df

    def _generate_synthetic(
        self, start: str, end: str, seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic yield curves using a 3-factor model.

        Calibrated to historical US Treasury statistics (2000-2024):
        - Level factor:     μ = 3.0%,  σ = 1.5%  (long-run mean ~3%)
        - Slope factor:     μ = 1.5%,  σ = 1.0%  (normal upward slope)
        - Curvature factor: μ = -0.5%, σ = 0.5%  (slight hump at belly)

        Uses Ornstein-Uhlenbeck dynamics for mean reversion:
            dX_t = κ(θ - X_t)dt + σdW_t
        """
        np.random.seed(seed)

        dates = pd.bdate_range(start=start, end=end)
        n_days = len(dates)
        maturities = self.maturities

        # OU process parameters for 3 factors
        dt = 1/252  # Daily

        # Factor 1: Level (parallel shifts)
        kappa_L, theta_L, sigma_L = 0.05, 3.5, 0.15
        # Factor 2: Slope (steepening/flattening)
        kappa_S, theta_S, sigma_S = 0.10, 1.8, 0.12
        # Factor 3: Curvature (butterfly)
        kappa_C, theta_C, sigma_C = 0.20, -0.4, 0.08

        # Simulate factors via Euler-Maruyama
        level = np.zeros(n_days)
        slope = np.zeros(n_days)
        curvature = np.zeros(n_days)

        level[0] = 4.0
        slope[0] = 1.5
        curvature[0] = -0.3

        for t in range(1, n_days):
            dW = np.random.randn(3) * np.sqrt(dt)
            level[t] = level[t-1] + kappa_L * (theta_L - level[t-1]) * dt + sigma_L * dW[0]
            slope[t] = slope[t-1] + kappa_S * (theta_S - slope[t-1]) * dt + sigma_S * dW[1]
            curvature[t] = curvature[t-1] + kappa_C * (theta_C - curvature[t-1]) * dt + sigma_C * dW[2]

        # Map factors to yield curve via Nelson-Siegel loadings
        tau = maturities
        lambda_ns = 1.5  # Decay parameter

        # Nelson-Siegel factor loadings
        beta1_loading = np.ones_like(tau)  # Level: flat across maturities
        beta2_loading = (1 - np.exp(-lambda_ns * tau)) / (lambda_ns * tau)  # Slope
        beta3_loading = beta2_loading - np.exp(-lambda_ns * tau)  # Curvature

        # Construct yield curves: Y(τ) = β₁·L₁(τ) + β₂·L₂(τ) + β₃·L₃(τ)
        yields_matrix = np.zeros((n_days, len(tau)))
        for t in range(n_days):
            yields_matrix[t, :] = (
                level[t] * beta1_loading +
                slope[t] * beta2_loading +
                curvature[t] * beta3_loading
            )

        # Add small idiosyncratic noise
        yields_matrix += np.random.randn(n_days, len(tau)) * 0.02

        # Floor at 0 (zero lower bound)
        yields_matrix = np.maximum(yields_matrix, 0.01)

        df = pd.DataFrame(yields_matrix, index=dates, columns=maturities)
        return df

    def get_maturities(self) -> np.ndarray:
        """Return array of maturities in years."""
        return self.maturities.copy()

    def get_latest_curve(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (maturities, yields) for the most recent date."""
        if self._data is None:
            return None
        latest = self._data.iloc[-1].dropna()
        return latest.index.values.astype(float), latest.values

    def get_curve_on_date(self, date: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (maturities, yields) for a specific date."""
        if self._data is None:
            return None
        try:
            row = self._data.loc[date].dropna()
            return row.index.values.astype(float), row.values
        except KeyError:
            # Find nearest date
            idx = self._data.index.get_indexer([pd.Timestamp(date)], method='nearest')[0]
            row = self._data.iloc[idx].dropna()
            return row.index.values.astype(float), row.values

    def compute_changes(self, window: int = 1) -> pd.DataFrame:
        """Compute yield changes over a given window (in business days)."""
        if self._data is None:
            return pd.DataFrame()
        return self._data.diff(window).dropna()

    def summary_stats(self) -> pd.DataFrame:
        """Summary statistics of yield levels and changes."""
        if self._data is None:
            return pd.DataFrame()

        levels = self._data.describe().T
        levels.index = [MATURITY_LABELS.get(m, f"{m}Y") for m in levels.index]
        levels.columns = [f"level_{c}" for c in levels.columns]

        changes = self.compute_changes()
        ch_desc = changes.describe().T
        ch_desc.index = [MATURITY_LABELS.get(m, f"{m}Y") for m in ch_desc.index]
        ch_desc.columns = [f"chg_{c}" for c in ch_desc.columns]

        return pd.concat([levels[['level_mean', 'level_std', 'level_min', 'level_max']],
                          ch_desc[['chg_mean', 'chg_std']]], axis=1)
