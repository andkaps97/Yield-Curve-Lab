"""
Fixed Income Trading Strategies
=================================

Implements institutional curve trading strategies:
1. Butterfly trades (sell belly, buy wings)
2. Barbell vs Bullet portfolio construction
3. Curve steepener/flattener trades
4. Duration-neutral relative value
5. Forward rate trading signals
6. Regime-based curve signal generation

Butterfly Trade:
    A butterfly is a 3-legged trade:
    - Long wings (short + long maturity)
    - Short body (intermediate maturity)
    - Duration-neutral: Σ DV01ᵢ · weightᵢ = 0

    Profit when curvature increases (belly cheapens relative to wings).

    Weighting schemes:
    - Cash-neutral: Σ notional = 0
    - Duration-neutral: Σ DV01 = 0
    - Regression-weighted: hedge ratios from PCA or regression

Barbell vs Bullet:
    Barbell: concentrated at short + long end (higher convexity)
    Bullet:  concentrated at intermediate maturity (higher carry)
    The barbell outperforms in volatile environments (convexity advantage).

References:
    - Ilmanen (2011): "Expected Returns", Ch. 6-8
    - Tuckman & Serrat (2012): "Fixed Income Securities", Ch. 5-7
    - Martellini, Priaulet, Priaulet (2003): "Fixed-Income Securities"
    - Ang & Piazzesi (2003): "A no-arbitrage vector autoregression"

Author: Andreas Kapsalis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from bond_analytics import Bond, BondCalculator, BondAnalytics


# ─────────────────────────────────────────────────────────────
# Butterfly Trade Analyzer
# ─────────────────────────────────────────────────────────────

@dataclass
class ButterflyTrade:
    """Represents a butterfly curve trade."""
    short_wing: Bond        # Short maturity (long position)
    body: Bond              # Intermediate (short position)
    long_wing: Bond         # Long maturity (long position)
    short_weight: float     # Weight of short wing
    body_weight: float      # Weight of body (negative = short)
    long_weight: float      # Weight of long wing
    net_dv01: float         # Residual DV01 (target: 0)
    carry_bps: float        # Total carry
    convexity_pickup: float # Convexity advantage


class ButterflyAnalyzer:
    """
    Construct and analyze butterfly curve trades.

    A butterfly is the canonical curve trade: it isolates curvature
    exposure while hedging level and slope risk.

    Standard butterflies:
        2s5s10s: Short 2Y, Long 5Y, Short 10Y (or vice versa)
        2s10s30s: Short 2Y, Long 10Y, Short 30Y

    Usage:
        >>> analyzer = ButterflyAnalyzer()
        >>> fly = analyzer.construct_butterfly(
        ...     maturities=(2, 5, 10),
        ...     yields=(4.5, 4.3, 4.4),
        ...     method='duration_neutral'
        ... )
    """

    def __init__(self):
        self.calc = BondCalculator()

    def construct_butterfly(
        self,
        maturities: Tuple[float, float, float],
        yields: Tuple[float, float, float],
        coupon_rates: Optional[Tuple[float, float, float]] = None,
        method: str = 'duration_neutral',
        body_notional: float = 100.0,
    ) -> ButterflyTrade:
        """
        Construct a duration-neutral butterfly trade.

        Convention: SHORT the body (intermediate), LONG the wings.
        Profits when curvature increases (belly yield rises vs wings).

        Args:
            maturities:    (short, body, long) maturities in years
            yields:        (short, body, long) yields in decimal
            coupon_rates:  Optional coupon rates (default = par)
            method:        'duration_neutral' or 'cash_neutral'
            body_notional: Notional on the body leg
        """
        m_s, m_b, m_l = maturities
        y_s, y_b, y_l = yields

        if coupon_rates is None:
            coupon_rates = yields  # Par bonds

        # Create bonds
        short_wing = Bond(f"{m_s:.0f}Y Wing", coupon_rates[0], m_s)
        body = Bond(f"{m_b:.0f}Y Body", coupon_rates[1], m_b)
        long_wing = Bond(f"{m_l:.0f}Y Wing", coupon_rates[2], m_l)

        # Compute DV01s
        dv01_s = self.calc.dv01(short_wing, y_s)
        dv01_b = self.calc.dv01(body, y_b)
        dv01_l = self.calc.dv01(long_wing, y_l)

        if method == 'duration_neutral':
            # Solve: w_s · DV01_s - 1 · DV01_b + w_l · DV01_l = 0
            # With constraint: w_s + w_l = body_notional (balanced wings)
            # Two equations, two unknowns:
            #   w_s · DV01_s + w_l · DV01_l = DV01_b · body_notional
            #   w_s = w_l · DV01_l / DV01_s  (proportional)

            # Standard approach: 50/50 risk weight on wings
            w_l = (body_notional * dv01_b) / (dv01_l + dv01_s * dv01_l / dv01_s)
            w_s = (body_notional * dv01_b - w_l * dv01_l) / dv01_s
        else:
            # Cash neutral: w_s + w_l = body_notional
            w_s = body_notional * 0.5
            w_l = body_notional * 0.5

        # Net DV01
        net_dv01 = w_s * dv01_s - body_notional * dv01_b + w_l * dv01_l

        # Carry analysis
        carry_s = self.calc.carry_and_rolldown(short_wing, y_s)[0]
        carry_b = self.calc.carry_and_rolldown(body, y_b)[0]
        carry_l = self.calc.carry_and_rolldown(long_wing, y_l)[0]
        net_carry = w_s * carry_s - body_notional * carry_b + w_l * carry_l

        # Convexity
        conv_s = self.calc.convexity(short_wing, y_s)
        conv_b = self.calc.convexity(body, y_b)
        conv_l = self.calc.convexity(long_wing, y_l)
        net_conv = w_s * conv_s - body_notional * conv_b + w_l * conv_l

        return ButterflyTrade(
            short_wing=short_wing,
            body=body,
            long_wing=long_wing,
            short_weight=w_s,
            body_weight=-body_notional,
            long_weight=w_l,
            net_dv01=net_dv01,
            carry_bps=net_carry,
            convexity_pickup=net_conv,
        )

    def scenario_analysis(
        self,
        fly: ButterflyTrade,
        yields: Tuple[float, float, float],
        scenarios: Dict[str, Tuple[float, float, float]],
    ) -> pd.DataFrame:
        """
        P&L analysis under yield curve scenarios.

        Args:
            fly:       ButterflyTrade object
            yields:    Current (short, body, long) yields
            scenarios: Dict of scenario_name → (Δy_short, Δy_body, Δy_long)

        Returns:
            DataFrame with P&L by scenario
        """
        y_s, y_b, y_l = yields
        results = []

        # Base prices
        p0_s = self.calc.price(fly.short_wing, y_s)
        p0_b = self.calc.price(fly.body, y_b)
        p0_l = self.calc.price(fly.long_wing, y_l)

        for name, (dy_s, dy_b, dy_l) in scenarios.items():
            p1_s = self.calc.price(fly.short_wing, y_s + dy_s)
            p1_b = self.calc.price(fly.body, y_b + dy_b)
            p1_l = self.calc.price(fly.long_wing, y_l + dy_l)

            pnl_s = fly.short_weight * (p1_s - p0_s)
            pnl_b = fly.body_weight * (p1_b - p0_b)
            pnl_l = fly.long_weight * (p1_l - p0_l)
            total_pnl = pnl_s + pnl_b + pnl_l

            results.append({
                'Scenario': name,
                'Δy_short': dy_s * 100,
                'Δy_body': dy_b * 100,
                'Δy_long': dy_l * 100,
                'PnL_short_wing': pnl_s,
                'PnL_body': pnl_b,
                'PnL_long_wing': pnl_l,
                'Total_PnL': total_pnl,
            })

        return pd.DataFrame(results).set_index('Scenario')


# ─────────────────────────────────────────────────────────────
# Barbell vs Bullet Analyzer
# ─────────────────────────────────────────────────────────────

class BarbellBulletAnalyzer:
    """
    Compare barbell vs bullet portfolio construction.

    Barbell: concentrated at short + long end
    Bullet:  concentrated at intermediate maturity

    Key tradeoff:
        Barbell → higher convexity (outperforms in vol)
        Bullet  → higher carry (outperforms in stable markets)
    """

    def __init__(self):
        self.calc = BondCalculator()

    def compare(
        self,
        short_bond: Bond, body_bond: Bond, long_bond: Bond,
        y_short: float, y_body: float, y_long: float,
        target_duration: Optional[float] = None,
    ) -> Dict:
        """
        Compare barbell (short+long) vs bullet (body) at matched duration.

        Args:
            short_bond, body_bond, long_bond: Bond objects
            y_short, y_body, y_long: Current yields
            target_duration: Target modified duration (defaults to bullet duration)
        """
        # Analytics
        a_s = self.calc.full_analytics(short_bond, y_short)
        a_b = self.calc.full_analytics(body_bond, y_body)
        a_l = self.calc.full_analytics(long_bond, y_long)

        # Bullet duration is the target
        if target_duration is None:
            target_duration = a_b.modified_duration

        # Barbell weights to match duration: w_s·D_s + w_l·D_l = D_target
        # w_s + w_l = 1
        if abs(a_l.modified_duration - a_s.modified_duration) < 1e-10:
            w_l = 0.5
        else:
            w_l = (target_duration - a_s.modified_duration) / (a_l.modified_duration - a_s.modified_duration)
        w_s = 1.0 - w_l

        barbell_dur = w_s * a_s.modified_duration + w_l * a_l.modified_duration
        barbell_conv = w_s * a_s.convexity + w_l * a_l.convexity
        barbell_ytm = w_s * y_short + w_l * y_long
        barbell_carry = w_s * a_s.carry_3m + w_l * a_l.carry_3m

        return {
            'bullet': {
                'bond': body_bond.name,
                'duration': a_b.modified_duration,
                'convexity': a_b.convexity,
                'ytm': y_body * 100,
                'carry_3m_bps': a_b.carry_3m,
            },
            'barbell': {
                'weights': {short_bond.name: w_s, long_bond.name: w_l},
                'duration': barbell_dur,
                'convexity': barbell_conv,
                'ytm': barbell_ytm * 100,
                'carry_3m_bps': barbell_carry,
            },
            'comparison': {
                'duration_match': abs(barbell_dur - a_b.modified_duration) < 0.01,
                'convexity_advantage': barbell_conv - a_b.convexity,
                'yield_give_up': (y_body - barbell_ytm) * 100,
                'carry_difference': barbell_carry - a_b.carry_3m,
            }
        }


# ─────────────────────────────────────────────────────────────
# Curve Trading Signals
# ─────────────────────────────────────────────────────────────

class CurveTradingSignals:
    """
    Generate systematic curve trading signals from yield data.

    Signal Suite:
    1. Slope mean-reversion: 2s10s spread z-score
    2. Butterfly richness: 2s5s10s fly vs historical
    3. Forward rate signal: implied forwards vs spot
    4. Carry momentum: total return ranking
    5. Regime indicator: steepening vs flattening phases

    All signals are standardized to z-scores for comparability.

    Usage:
        >>> signals = CurveTradingSignals()
        >>> signal_df = signals.generate_all(yields_df)
    """

    def slope_signal(
        self, yields_df: pd.DataFrame, short_mat: float = 2.0,
        long_mat: float = 10.0, lookback: int = 252,
    ) -> pd.Series:
        """
        Slope mean-reversion signal (z-score of 2s10s spread).

        Signal > 0 → curve steep vs history → expect flattening → enter flattener
        Signal < 0 → curve flat vs history → expect steepening → enter steepener
        """
        if short_mat not in yields_df.columns or long_mat not in yields_df.columns:
            # Find nearest
            cols = np.array(yields_df.columns, dtype=float)
            short_mat = cols[np.argmin(np.abs(cols - short_mat))]
            long_mat = cols[np.argmin(np.abs(cols - long_mat))]

        spread = yields_df[long_mat] - yields_df[short_mat]
        rolling_mean = spread.rolling(lookback, min_periods=60).mean()
        rolling_std = spread.rolling(lookback, min_periods=60).std()

        z_score = (spread - rolling_mean) / (rolling_std + 1e-10)
        z_score.name = f'slope_{int(short_mat)}s{int(long_mat)}s_zscore'
        return z_score

    def butterfly_signal(
        self, yields_df: pd.DataFrame,
        wings: Tuple[float, float] = (2.0, 10.0),
        body: float = 5.0, lookback: int = 252,
    ) -> pd.Series:
        """
        Butterfly richness/cheapness signal.

        Butterfly spread = (y_short + y_long)/2 - y_body

        Positive → belly cheap → sell butterfly (profit from curvature compression)
        Negative → belly rich → buy butterfly (profit from curvature expansion)
        """
        cols = np.array(yields_df.columns, dtype=float)
        w1 = cols[np.argmin(np.abs(cols - wings[0]))]
        w2 = cols[np.argmin(np.abs(cols - wings[1]))]
        b = cols[np.argmin(np.abs(cols - body))]

        fly = (yields_df[w1] + yields_df[w2]) / 2.0 - yields_df[b]
        rolling_mean = fly.rolling(lookback, min_periods=60).mean()
        rolling_std = fly.rolling(lookback, min_periods=60).std()

        z_score = (fly - rolling_mean) / (rolling_std + 1e-10)
        z_score.name = f'butterfly_{int(w1)}s{int(b)}s{int(w2)}s_zscore'
        return z_score

    def forward_rate_signal(
        self, yields_df: pd.DataFrame, forward_start: float = 1.0,
        forward_tenor: float = 1.0,
    ) -> pd.Series:
        """
        Forward rate vs spot signal.

        If implied forward > current spot → term premium → receive fixed
        If implied forward < current spot → expectations of lower rates

        Signal = (forward - spot) z-score
        """
        cols = np.array(yields_df.columns, dtype=float)
        t1 = cols[np.argmin(np.abs(cols - forward_start))]
        t2 = cols[np.argmin(np.abs(cols - (forward_start + forward_tenor)))]

        # Discrete forward rate
        forward = (yields_df[t2] * t2 - yields_df[t1] * t1) / (t2 - t1)
        spot = yields_df[t1]

        premium = forward - spot
        z = (premium - premium.rolling(252, min_periods=60).mean()) / (
            premium.rolling(252, min_periods=60).std() + 1e-10
        )
        z.name = f'forward_{int(forward_start)}y{int(forward_tenor)}y_signal'
        return z

    def carry_signal(
        self, yields_df: pd.DataFrame, maturities_to_rank: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Carry momentum signal: rank maturities by total expected return.

        Carry ≈ yield + roll-down - financing
        Momentum ≈ trailing 3M total return

        Cross-sectional ranking: z-score across maturities.
        """
        if maturities_to_rank is None:
            maturities_to_rank = [2.0, 5.0, 10.0, 30.0]

        cols = np.array(yields_df.columns, dtype=float)
        available = []
        for m in maturities_to_rank:
            nearest = cols[np.argmin(np.abs(cols - m))]
            if nearest not in available:
                available.append(nearest)

        # Simple carry proxy: yield level (higher yield = higher carry)
        carry_df = yields_df[available].copy()

        # Cross-sectional z-score (rank maturities at each date)
        carry_z = carry_df.sub(carry_df.mean(axis=1), axis=0).div(
            carry_df.std(axis=1) + 1e-10, axis=0
        )

        return carry_z

    def regime_indicator(
        self, yields_df: pd.DataFrame, lookback: int = 63
    ) -> pd.Series:
        """
        Curve regime indicator based on slope trend.

        +1 = Steepening regime (slope increasing)
        -1 = Flattening regime (slope decreasing)
         0 = Neutral / range-bound
        """
        cols = np.array(yields_df.columns, dtype=float)
        short = cols[np.argmin(np.abs(cols - 2.0))]
        long = cols[np.argmin(np.abs(cols - 10.0))]

        slope = yields_df[long] - yields_df[short]
        slope_ma = slope.rolling(lookback, min_periods=20).mean()
        slope_trend = slope_ma.diff(lookback)

        regime = pd.Series(0, index=yields_df.index, name='curve_regime')
        regime[slope_trend > slope_trend.rolling(252, min_periods=60).std()] = 1
        regime[slope_trend < -slope_trend.rolling(252, min_periods=60).std()] = -1

        return regime

    def generate_all(self, yields_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all trading signals and combine into a single DataFrame."""
        signals = pd.DataFrame(index=yields_df.index)

        signals['slope_2s10s'] = self.slope_signal(yields_df)
        signals['butterfly_2s5s10s'] = self.butterfly_signal(yields_df)
        signals['forward_1y1y'] = self.forward_rate_signal(yields_df, 1.0, 1.0)
        signals['regime'] = self.regime_indicator(yields_df)

        carry = self.carry_signal(yields_df)
        for col in carry.columns:
            from data_fetcher import MATURITY_LABELS
            label = MATURITY_LABELS.get(col, f"{col}Y")
            signals[f'carry_{label}'] = carry[col]

        return signals.dropna()
