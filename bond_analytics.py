"""
Bond Analytics: Duration, Convexity & Portfolio Risk
=====================================================

Institutional-grade fixed income analytics:
1. Macaulay & Modified Duration
2. Effective (OAS) Duration
3. Convexity and DV01/BPV
4. Key Rate Duration (KRD) decomposition
5. Portfolio-level risk aggregation
6. Carry and roll-down analysis

Theory:
    Modified Duration:
        D_mod = -1/P · dP/dy = D_mac / (1 + y/m)

    Convexity:
        C = 1/P · d²P/dy²

    Price change approximation (Taylor expansion):
        ΔP/P ≈ -D_mod · Δy + ½ · C · (Δy)² + O(Δy³)

    DV01 (Dollar Value of 01):
        DV01 = D_mod · P / 10000

    Key Rate Duration:
        KRD(τ) = -1/P · ∂P/∂y(τ)
        Measures sensitivity to specific maturity point shifts.

References:
    - Fabozzi (2007): "Fixed Income Analysis", CFA Institute
    - Tuckman & Serrat (2012): "Fixed Income Securities", 3rd ed.
    - Ho (1992): "Key Rate Durations"
    - Ilmanen (2011): "Expected Returns", Ch. 6-7

Author: Andreas Kapsalis
"""

import numpy as np
from numba import njit
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# Numba-Accelerated Bond Pricing
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def _bond_price(coupon_rate: float, ytm: float, maturity: float,
                face: float, freq: int) -> float:
    """
    Price a fixed-coupon bond using discounted cash flow.

    P = Σ (C/m) / (1 + y/m)^t + F / (1 + y/m)^n

    Args:
        coupon_rate: Annual coupon rate (decimal, e.g., 0.05)
        ytm:         Yield to maturity (decimal)
        maturity:    Years to maturity
        face:        Face value
        freq:        Coupon frequency (1=annual, 2=semi)
    """
    n_periods = int(maturity * freq)
    if n_periods <= 0:
        return face

    periodic_rate = ytm / freq
    coupon = coupon_rate * face / freq

    price = 0.0
    for t in range(1, n_periods + 1):
        price += coupon / (1.0 + periodic_rate) ** t
    price += face / (1.0 + periodic_rate) ** n_periods

    return price


@njit(cache=True)
def _macaulay_duration(coupon_rate: float, ytm: float, maturity: float,
                        face: float, freq: int) -> float:
    """
    Macaulay duration: weighted-average time to cash flows.

    D_mac = (1/P) · Σ t · CF_t / (1+y/m)^t
    """
    n_periods = int(maturity * freq)
    if n_periods <= 0:
        return 0.0

    periodic_rate = ytm / freq
    coupon = coupon_rate * face / freq

    price = 0.0
    weighted_time = 0.0

    for t in range(1, n_periods + 1):
        pv = coupon / (1.0 + periodic_rate) ** t
        price += pv
        weighted_time += (t / freq) * pv

    pv_face = face / (1.0 + periodic_rate) ** n_periods
    price += pv_face
    weighted_time += (n_periods / freq) * pv_face

    if price <= 0:
        return 0.0
    return weighted_time / price


@njit(cache=True)
def _convexity(coupon_rate: float, ytm: float, maturity: float,
               face: float, freq: int) -> float:
    """
    Bond convexity: curvature of price-yield relationship.

    C = (1/P) · (1/(1+y/m)²) · Σ t(t+1)/m² · CF_t / (1+y/m)^t
    """
    n_periods = int(maturity * freq)
    if n_periods <= 0:
        return 0.0

    periodic_rate = ytm / freq
    coupon = coupon_rate * face / freq

    price = 0.0
    conv_sum = 0.0

    for t in range(1, n_periods + 1):
        pv = coupon / (1.0 + periodic_rate) ** t
        price += pv
        conv_sum += t * (t + 1) * pv

    pv_face = face / (1.0 + periodic_rate) ** n_periods
    price += pv_face
    conv_sum += n_periods * (n_periods + 1) * pv_face

    if price <= 0:
        return 0.0

    return conv_sum / (price * freq**2 * (1.0 + periodic_rate)**2)


# ─────────────────────────────────────────────────────────────
# Bond Data Structure
# ─────────────────────────────────────────────────────────────

@dataclass
class Bond:
    """Represents a fixed-coupon bond."""
    name: str
    coupon_rate: float      # Annual coupon (decimal)
    maturity: float         # Years to maturity
    face_value: float = 100.0
    freq: int = 2           # Semi-annual


@dataclass
class BondAnalytics:
    """Complete analytics for a single bond."""
    name: str
    clean_price: float
    dirty_price: float
    ytm: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float             # Dollar value of 1bp
    bpv: float              # Basis point value (= DV01 × notional/100)
    carry_3m: float         # 3-month carry (yield - repo)
    rolldown_3m: float      # 3-month roll-down return


class BondCalculator:
    """
    Fixed income analytics calculator.

    Provides a complete set of risk metrics for bonds and portfolios,
    following CFA Institute and institutional conventions.

    Usage:
        >>> calc = BondCalculator()
        >>> bond = Bond("UST 10Y", coupon_rate=0.04, maturity=10)
        >>> analytics = calc.full_analytics(bond, ytm=0.045)
    """

    def price(self, bond: Bond, ytm: float) -> float:
        """Clean price of a bond given YTM."""
        return _bond_price(bond.coupon_rate, ytm, bond.maturity,
                          bond.face_value, bond.freq)

    def ytm_from_price(self, bond: Bond, price: float, tol: float = 1e-10) -> float:
        """Solve for YTM given price using Newton-Raphson."""
        y = bond.coupon_rate  # Initial guess

        for _ in range(200):
            p = self.price(bond, y)
            dp = -(self.price(bond, y + 0.0001) - p) / 0.0001  # Numerical derivative
            if abs(dp) < 1e-15:
                break
            y_new = y + (p - price) / dp
            if abs(y_new - y) < tol:
                y = y_new
                break
            y = y_new

        return y

    def macaulay_duration(self, bond: Bond, ytm: float) -> float:
        """Macaulay duration in years."""
        return _macaulay_duration(bond.coupon_rate, ytm, bond.maturity,
                                  bond.face_value, bond.freq)

    def modified_duration(self, bond: Bond, ytm: float) -> float:
        """Modified duration = Macaulay / (1 + y/m)."""
        mac = self.macaulay_duration(bond, ytm)
        return mac / (1.0 + ytm / bond.freq)

    def effective_duration(self, bond: Bond, ytm: float, dy: float = 0.0001) -> float:
        """Effective duration via finite differences."""
        p_up = self.price(bond, ytm + dy)
        p_down = self.price(bond, ytm - dy)
        p0 = self.price(bond, ytm)
        return -(p_up - p_down) / (2.0 * dy * p0)

    def convexity(self, bond: Bond, ytm: float) -> float:
        """Bond convexity."""
        return _convexity(bond.coupon_rate, ytm, bond.maturity,
                         bond.face_value, bond.freq)

    def effective_convexity(self, bond: Bond, ytm: float, dy: float = 0.0001) -> float:
        """Effective convexity via finite differences."""
        p_up = self.price(bond, ytm + dy)
        p_down = self.price(bond, ytm - dy)
        p0 = self.price(bond, ytm)
        return (p_up + p_down - 2 * p0) / (dy**2 * p0)

    def dv01(self, bond: Bond, ytm: float) -> float:
        """Dollar value of a basis point (DV01)."""
        dur = self.modified_duration(bond, ytm)
        price = self.price(bond, ytm)
        return dur * price / 10000.0

    def key_rate_durations(
        self, bond: Bond, ytm: float, yield_curve: np.ndarray,
        maturities: np.ndarray, dy: float = 0.0001,
    ) -> np.ndarray:
        """
        Key Rate Duration (KRD) decomposition.

        Shifts each maturity point independently and measures
        the resulting price change. Sums of KRDs = effective duration.

        Args:
            bond:        Bond to analyze
            ytm:         Current YTM
            yield_curve: Baseline yield curve (same length as maturities)
            maturities:  Curve maturities
            dy:          Shift size (1bp default)

        Returns:
            Array of key rate durations at each maturity point
        """
        p0 = self.price(bond, ytm)
        n_points = len(maturities)
        krds = np.zeros(n_points)

        for i in range(n_points):
            # Create shifted curve: bump only maturity point i
            # Interpolate bump effect on bond's YTM
            if abs(maturities[i] - bond.maturity) < 0.5:
                # Bond maturity close to this key rate point
                weight = 1.0 - abs(maturities[i] - bond.maturity) / max(0.5, 1.0)
            elif bond.maturity < maturities[0]:
                weight = 1.0 if i == 0 else 0.0
            elif bond.maturity > maturities[-1]:
                weight = 1.0 if i == n_points - 1 else 0.0
            else:
                # Linear interpolation weight
                for j in range(n_points - 1):
                    if maturities[j] <= bond.maturity <= maturities[j+1]:
                        if i == j:
                            weight = (maturities[j+1] - bond.maturity) / (maturities[j+1] - maturities[j])
                        elif i == j + 1:
                            weight = (bond.maturity - maturities[j]) / (maturities[j+1] - maturities[j])
                        else:
                            weight = 0.0
                        break
                else:
                    weight = 0.0

            shifted_ytm = ytm + dy * weight
            p_shifted = self.price(bond, shifted_ytm)
            krds[i] = -(p_shifted - p0) / (dy * p0)

        return krds

    def carry_and_rolldown(
        self, bond: Bond, ytm: float, repo_rate: float = 0.04,
        horizon_months: int = 3,
    ) -> Tuple[float, float]:
        """
        Compute carry and roll-down for a bond.

        Carry = coupon income - financing cost over horizon
        Roll-down = price appreciation from "rolling down" the curve
                    (maturity shortens, yield typically drops for
                     normal upward-sloping curves)

        Args:
            bond:            Bond
            ytm:             Current yield
            repo_rate:       Financing rate (decimal)
            horizon_months:  Holding period

        Returns:
            (carry_bps, rolldown_bps) annualized in basis points
        """
        horizon_years = horizon_months / 12.0
        price = self.price(bond, ytm)

        # Carry: coupon income - financing cost
        coupon_income = bond.coupon_rate * bond.face_value * horizon_years
        financing = repo_rate * price * horizon_years
        carry = (coupon_income - financing) / price

        # Roll-down: assume curve doesn't change, bond matures by horizon
        new_maturity = bond.maturity - horizon_years
        if new_maturity > 0:
            # Assume yield drops by duration × slope effect
            # Simple approximation: yield change = slope × Δmaturity
            slope_approx = (ytm * 0.1) / bond.maturity  # Rough estimate
            new_ytm = ytm - slope_approx * horizon_years
            new_price = _bond_price(bond.coupon_rate, new_ytm, new_maturity,
                                    bond.face_value, bond.freq)
            rolldown = (new_price - price) / price
        else:
            rolldown = 0.0

        return (carry * 10000, rolldown * 10000)  # Convert to bps

    def full_analytics(self, bond: Bond, ytm: float, repo_rate: float = 0.04) -> BondAnalytics:
        """Compute complete analytics for a bond."""
        price = self.price(bond, ytm)
        mac_dur = self.macaulay_duration(bond, ytm)
        mod_dur = self.modified_duration(bond, ytm)
        conv = self.convexity(bond, ytm)
        dv01 = self.dv01(bond, ytm)
        carry, rolldown = self.carry_and_rolldown(bond, ytm, repo_rate)

        return BondAnalytics(
            name=bond.name,
            clean_price=price,
            dirty_price=price,  # Simplified (no accrued)
            ytm=ytm,
            macaulay_duration=mac_dur,
            modified_duration=mod_dur,
            convexity=conv,
            dv01=dv01,
            bpv=dv01,
            carry_3m=carry,
            rolldown_3m=rolldown,
        )

    def price_change(self, bond: Bond, ytm: float, dy: float) -> Dict:
        """
        Estimate price change for a yield shift using duration/convexity.

        Returns actual vs approximated change.
        """
        p0 = self.price(bond, ytm)
        p1 = self.price(bond, ytm + dy)
        actual_pct = (p1 - p0) / p0

        dur = self.modified_duration(bond, ytm)
        conv = self.convexity(bond, ytm)

        # First-order (duration only)
        approx_1st = -dur * dy
        # Second-order (duration + convexity)
        approx_2nd = -dur * dy + 0.5 * conv * dy**2

        return {
            'actual_pct': actual_pct * 100,
            'duration_approx_pct': approx_1st * 100,
            'dur_conv_approx_pct': approx_2nd * 100,
            'duration_error_bps': (actual_pct - approx_1st) * 10000,
            'dur_conv_error_bps': (actual_pct - approx_2nd) * 10000,
        }


# ─────────────────────────────────────────────────────────────
# Portfolio Analytics
# ─────────────────────────────────────────────────────────────

class PortfolioAnalytics:
    """
    Portfolio-level fixed income risk analytics.

    Aggregates individual bond risk measures to portfolio level
    using market-value weighting.

    Usage:
        >>> pa = PortfolioAnalytics()
        >>> bonds = [Bond("2Y", 0.04, 2), Bond("10Y", 0.04, 10)]
        >>> weights = [0.5, 0.5]
        >>> ytms = [0.04, 0.045]
        >>> report = pa.portfolio_report(bonds, weights, ytms)
    """

    def __init__(self):
        self.calc = BondCalculator()

    def portfolio_report(
        self, bonds: List[Bond], weights: np.ndarray,
        ytms: np.ndarray, repo_rate: float = 0.04,
    ) -> Dict:
        """Generate comprehensive portfolio risk report."""
        weights = np.asarray(weights)
        ytms = np.asarray(ytms)

        # Individual analytics
        analytics = []
        for bond, ytm in zip(bonds, ytms):
            analytics.append(self.calc.full_analytics(bond, ytm, repo_rate))

        # Portfolio-weighted metrics
        port_duration = np.sum([w * a.modified_duration for w, a in zip(weights, analytics)])
        port_convexity = np.sum([w * a.convexity for w, a in zip(weights, analytics)])
        port_dv01 = np.sum([w * a.dv01 for w, a in zip(weights, analytics)])
        port_carry = np.sum([w * a.carry_3m for w, a in zip(weights, analytics)])
        port_rolldown = np.sum([w * a.rolldown_3m for w, a in zip(weights, analytics)])
        port_ytm = np.sum([w * y for w, y in zip(weights, ytms)])

        return {
            'individual': analytics,
            'portfolio': {
                'weighted_ytm': port_ytm,
                'modified_duration': port_duration,
                'convexity': port_convexity,
                'dv01': port_dv01,
                'carry_3m_bps': port_carry,
                'rolldown_3m_bps': port_rolldown,
                'total_return_3m_bps': port_carry + port_rolldown,
            }
        }
