"""
PCA Yield Curve Decomposition
===============================

Implements the Litterman-Scheinkman (1991) principal component analysis
of yield curve movements, decomposing yield changes into orthogonal
factors: Level, Slope, and Curvature.

Key Result:
    Three factors explain ~99% of yield curve variance:
    - PC1 (Level):     ~85-90% — parallel shifts
    - PC2 (Slope):     ~8-12%  — steepening/flattening
    - PC3 (Curvature): ~2-4%   — butterfly movements

    This is one of the most robust empirical results in fixed income,
    replicated across markets, currencies, and time periods.

Applications:
    - Risk decomposition: express portfolio exposure to L/S/C factors
    - Hedging: construct factor-neutral portfolios
    - Scenario generation: stress test using factor shocks
    - Relative value: identify dislocations from factor model fair value
    - Volatility estimation: factor-based yield curve VaR

References:
    - Litterman & Scheinkman (1991): "Common Factors Affecting Bond Returns"
    - Barber & Copper (1996): "Immunization Using Principal Component Analysis"
    - Lord & Pelsser (2007): "Level-Slope-Curvature — Fact or Artefact?"

Author: Andreas Kapsalis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PCAResult:
    """Container for yield curve PCA results."""
    # Eigenvectors (loadings) — shape (n_components, n_maturities)
    loadings: np.ndarray
    # Eigenvalues (variance explained)
    eigenvalues: np.ndarray
    # Variance explained ratios
    variance_explained: np.ndarray
    # Cumulative variance explained
    cumulative_variance: np.ndarray
    # Factor scores (time series) — shape (n_dates, n_components)
    factor_scores: np.ndarray
    # Maturities used
    maturities: np.ndarray
    # Factor labels
    labels: List[str]
    # Original data stats
    mean_curve: np.ndarray
    std_curve: np.ndarray
    # Covariance matrix
    covariance_matrix: np.ndarray
    # Correlation matrix
    correlation_matrix: np.ndarray


class YieldCurvePCA:
    """
    Principal Component Analysis of yield curve movements.

    Implements the standard Litterman-Scheinkman methodology:
    1. Compute yield changes (first differences)
    2. Estimate covariance matrix of yield changes
    3. Eigendecomposition → factor loadings and scores
    4. Interpret factors as Level, Slope, Curvature

    Sign Convention:
        PC1 (Level):     positive = parallel rise in all yields
        PC2 (Slope):     positive = steepening (short down, long up)
        PC3 (Curvature): positive = belly cheapening (butterfly widens)

    Usage:
        >>> pca = YieldCurvePCA()
        >>> result = pca.decompose(yields_df, n_components=3)
        >>> print(f"Level explains {result.variance_explained[0]:.1%}")
    """

    def decompose(
        self,
        yields_df: pd.DataFrame,
        n_components: int = 3,
        use_changes: bool = True,
        standardize: bool = False,
    ) -> PCAResult:
        """
        Perform PCA on yield curve data.

        Args:
            yields_df:     DataFrame (dates × maturities) of yields
            n_components:  Number of principal components to extract
            use_changes:   If True, analyze yield changes (standard approach)
            standardize:   If True, use correlation matrix (not covariance)

        Returns:
            PCAResult with loadings, scores, and diagnostics
        """
        maturities = np.array(yields_df.columns, dtype=float)

        if use_changes:
            data = yields_df.diff().dropna().values
        else:
            data = yields_df.values

        n_obs, n_mat = data.shape
        mean_vec = np.mean(data, axis=0)
        std_vec = np.std(data, axis=0)

        # Center the data
        centered = data - mean_vec

        if standardize:
            # Use correlation matrix (normalize by std)
            centered = centered / (std_vec + 1e-10)
            cov_matrix = np.corrcoef(data.T)
        else:
            # Use covariance matrix (standard for yield curve PCA)
            cov_matrix = np.cov(data.T)

        corr_matrix = np.corrcoef(data.T)

        # Eigendecomposition (sorted descending by eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sort_idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Keep top n_components
        eigenvalues = eigenvalues[:n_components]
        loadings = eigenvectors[:, :n_components].T  # Shape: (n_components, n_mat)

        # Enforce sign convention:
        # PC1: all positive (parallel shift up)
        if np.mean(loadings[0]) < 0:
            loadings[0] *= -1

        # PC2: positive for long end (steepening)
        if n_components >= 2 and loadings[1, -1] < loadings[1, 0]:
            loadings[1] *= -1

        # PC3: negative at belly (curvature — wings up, belly down)
        if n_components >= 3:
            mid_idx = len(maturities) // 2
            if loadings[2, mid_idx] > 0:
                loadings[2] *= -1

        # Variance explained
        total_var = np.sum(np.linalg.eigvalsh(cov_matrix))
        var_explained = eigenvalues / total_var
        cum_var = np.cumsum(var_explained)

        # Factor scores: project centered data onto loadings
        factor_scores = centered @ loadings.T

        # Labels
        labels = ['Level (PC1)', 'Slope (PC2)', 'Curvature (PC3)']
        if n_components > 3:
            labels += [f'PC{i+1}' for i in range(3, n_components)]

        # Mean curve (for levels analysis)
        mean_curve = np.mean(yields_df.values, axis=0)
        std_curve = np.std(yields_df.values, axis=0)

        return PCAResult(
            loadings=loadings,
            eigenvalues=eigenvalues,
            variance_explained=var_explained,
            cumulative_variance=cum_var,
            factor_scores=factor_scores,
            maturities=maturities,
            labels=labels[:n_components],
            mean_curve=mean_curve,
            std_curve=std_curve,
            covariance_matrix=cov_matrix,
            correlation_matrix=corr_matrix,
        )

    def factor_risk_decomposition(
        self, result: PCAResult, portfolio_dv01: np.ndarray
    ) -> Dict:
        """
        Decompose portfolio risk into factor exposures.

        Given a portfolio's DV01 vector (sensitivity to each maturity point),
        compute factor exposures:
            Factor DV01_k = Σᵢ DV01ᵢ · Loading_k(τᵢ)

        Args:
            result:        PCA result
            portfolio_dv01: DV01 at each maturity point (same length as maturities)

        Returns:
            Dict with factor exposures and risk decomposition
        """
        n_factors = len(result.eigenvalues)
        factor_dv01 = np.zeros(n_factors)

        for k in range(n_factors):
            factor_dv01[k] = np.sum(portfolio_dv01 * result.loadings[k])

        # Factor VaR (1-sigma)
        factor_var = np.abs(factor_dv01) * np.sqrt(result.eigenvalues)

        # Total VaR (assuming independence of factors)
        total_var = np.sqrt(np.sum(factor_var**2))

        return {
            'factor_dv01': factor_dv01,
            'factor_var_1sigma': factor_var,
            'total_var_1sigma': total_var,
            'pct_risk_by_factor': (factor_var**2) / (total_var**2 + 1e-15) * 100,
        }

    def reconstruct_curve(
        self, result: PCAResult, factor_shocks: np.ndarray,
        base_curve: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Reconstruct a yield curve from factor shocks.

        Δy = Σ shock_k · Loading_k

        Args:
            result:        PCA result
            factor_shocks: Array of shocks to each factor (in std dev units)
            base_curve:    Starting curve (uses mean if None)

        Returns:
            Reconstructed yield curve
        """
        if base_curve is None:
            base_curve = result.mean_curve

        delta_y = np.zeros(len(result.maturities))
        for k in range(len(factor_shocks)):
            # Shock in standard deviation units × eigenvalue
            delta_y += factor_shocks[k] * np.sqrt(result.eigenvalues[k]) * result.loadings[k]

        return base_curve + delta_y

    def scenario_analysis(self, result: PCAResult, scenarios: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Generate yield curves under named scenarios.

        Args:
            result:    PCA result
            scenarios: Dict of scenario_name → factor_shocks array

        Returns:
            DataFrame with scenario curves
        """
        curves = {}
        for name, shocks in scenarios.items():
            curves[name] = self.reconstruct_curve(result, shocks)

        df = pd.DataFrame(curves, index=result.maturities)
        df.index.name = 'Maturity'
        return df

    def historical_factor_stats(self, result: PCAResult, yields_df: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptive statistics of factor score time series."""
        dates = yields_df.diff().dropna().index
        factor_df = pd.DataFrame(
            result.factor_scores,
            index=dates[:len(result.factor_scores)],
            columns=result.labels,
        )

        stats = factor_df.describe().T
        stats['skewness'] = factor_df.skew()
        stats['kurtosis'] = factor_df.kurtosis()
        stats['variance_pct'] = result.variance_explained * 100

        return stats
