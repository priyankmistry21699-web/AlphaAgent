"""
AlphaAgent — 0DTE (Zero Days-to-Expiry) Options Flow Signal

Theory:
  0DTE options now represent ~50% of all US equity options volume.
  Because they expire same-day, market makers must delta-hedge aggressively.
  Net 0DTE call buying → dealers short gamma → they BUY the underlying
  (price acceleration).  Net 0DTE put buying → dealers short gamma → they
  SELL the underlying (price deceleration / crash).

  This creates a self-fulfilling, intraday momentum signal.

Method (using only free yfinance options data):
  1. Get today's options expirations — identify the nearest one (0DTE or 1DTE).
  2. For that expiry, compute:
       a. Net call open interest (calls OI - puts OI) at near-the-money strikes
       b. Net premium (call premium - put premium) across all strikes
       c. Gamma exposure at each strike → sum up net dealer gamma
  3. Score based on:
       - Net call OI ratio > threshold → bullish 0DTE flow
       - Net dealer gamma (GEX) > 0    → price stabilising (support)
       - Net dealer gamma (GEX) < 0    → price amplification (momentum)

Score (0-100):
  Strong call flow + positive GEX : 78
  Moderate call flow              : 62
  Neutral                         : 50
  Moderate put flow               : 38
  Strong put flow + negative GEX  : 22
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_GEX_SCALE    = 1e6     # normalise GEX to human-readable units
_MIN_OI       = 50      # minimum OI to include a strike


@dataclass
class ZeroDTEResult:
    score: float                  # 0-100
    prob_adjustment: float        # -0.08 to +0.08
    net_call_oi: float            # calls OI - puts OI (near ATM)
    net_gex: float                # net dealer gamma exposure (normalised)
    call_put_premium_ratio: float # total call prem / total put prem
    days_to_expiry: int           # 0 = same day, 1 = next day
    regime: str                   # CALL_DOMINATED / PUT_DOMINATED / NEUTRAL
    n_strikes: int


def compute_zero_dte(ticker: str, current_price: Optional[float] = None) -> Optional[ZeroDTEResult]:
    """
    Compute 0DTE options flow signal for a US equity ticker.
    Returns None on any data failure.
    """
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)

        # ── 1. Find nearest expiry (0DTE or 1DTE) ────────────────────────
        try:
            expirations = t.options
        except Exception:
            return None

        if not expirations:
            return None

        today = date.today()
        nearest_exp = None
        dte = 0

        for exp_str in expirations[:3]:
            try:
                exp_date = date.fromisoformat(exp_str)
                delta    = (exp_date - today).days
                if 0 <= delta <= 2:
                    nearest_exp = exp_str
                    dte = delta
                    break
            except Exception:
                continue

        if nearest_exp is None:
            return None   # no near-term expiry

        # ── 2. Fetch options chain ────────────────────────────────────────
        try:
            chain = t.option_chain(nearest_exp)
        except Exception:
            return None

        calls = chain.calls
        puts  = chain.puts

        if calls is None or puts is None or calls.empty or puts.empty:
            return None

        # Get current price
        if current_price is None or current_price <= 0:
            try:
                hist = t.history(period="1d", interval="1d")
                current_price = float(hist["Close"].iloc[-1]) if not hist.empty else None
            except Exception:
                pass
        if not current_price or current_price <= 0:
            return None

        # ── 3. Near-the-money strikes (within ±5% of current price) ──────
        atm_low  = current_price * 0.95
        atm_high = current_price * 1.05

        atm_calls = calls[
            (calls["strike"] >= atm_low) &
            (calls["strike"] <= atm_high) &
            (calls["openInterest"] >= _MIN_OI)
        ]
        atm_puts = puts[
            (puts["strike"] >= atm_low) &
            (puts["strike"] <= atm_high) &
            (puts["openInterest"] >= _MIN_OI)
        ]

        if atm_calls.empty and atm_puts.empty:
            return None

        net_call_oi = (
            float(atm_calls["openInterest"].sum()) -
            float(atm_puts["openInterest"].sum())
        )

        # ── 4. Total premium ratio ────────────────────────────────────────
        def _premium(df):
            mid = (df.get("bid", df.get("lastPrice", 0)) +
                   df.get("ask", df.get("lastPrice", 0))) / 2
            return float((mid * df["openInterest"]).fillna(0).sum())

        call_prem = _premium(calls)
        put_prem  = _premium(puts)
        cp_ratio  = call_prem / put_prem if put_prem > 0 else 1.0

        # ── 5. Net dealer gamma exposure (GEX) ───────────────────────────
        # Dealers: short calls, long puts → dealer gamma = call_gamma - put_gamma
        # If net_gex > 0 → dealers absorb vol (stabilising)
        # If net_gex < 0 → dealers amplify vol (acceleration)
        net_gex = 0.0
        T = max(dte, 0.5) / 365.0
        try:
            from scipy.stats import norm
            from math import log, sqrt

            rf = 0.045
            for _, row in calls.iterrows():
                K  = row.get("strike", 0)
                iv = row.get("impliedVolatility", 0)
                oi = row.get("openInterest", 0)
                if K > 0 and iv > 0 and oi > 0:
                    d1 = (log(current_price / K) + (rf + 0.5 * iv**2) * T) / (iv * sqrt(T))
                    gamma = norm.pdf(d1) / (current_price * iv * sqrt(T))
                    net_gex += gamma * oi * 100  # 100 shares per contract

            for _, row in puts.iterrows():
                K  = row.get("strike", 0)
                iv = row.get("impliedVolatility", 0)
                oi = row.get("openInterest", 0)
                if K > 0 and iv > 0 and oi > 0:
                    d1 = (log(current_price / K) + (rf + 0.5 * iv**2) * T) / (iv * sqrt(T))
                    gamma = norm.pdf(d1) / (current_price * iv * sqrt(T))
                    net_gex -= gamma * oi * 100

            net_gex = net_gex * current_price / _GEX_SCALE
        except Exception:
            net_gex = 0.0

        # ── 6. Score ──────────────────────────────────────────────────────
        n_atm = len(atm_calls) + len(atm_puts)

        # Normalise net_call_oi
        total_oi = float(atm_calls["openInterest"].sum() + atm_puts["openInterest"].sum())
        oi_ratio = net_call_oi / total_oi if total_oi > 0 else 0.0

        if oi_ratio > 0.35 and cp_ratio > 1.3:
            score    = 78.0
            regime   = "CALL_DOMINATED"
            prob_adj = 0.07
        elif oi_ratio > 0.15 or cp_ratio > 1.1:
            score    = 63.0
            regime   = "CALL_LEAN"
            prob_adj = 0.03
        elif oi_ratio < -0.35 and cp_ratio < 0.75:
            score    = 22.0
            regime   = "PUT_DOMINATED"
            prob_adj = -0.07
        elif oi_ratio < -0.15 or cp_ratio < 0.90:
            score    = 37.0
            regime   = "PUT_LEAN"
            prob_adj = -0.03
        else:
            score    = 50.0
            regime   = "NEUTRAL"
            prob_adj = 0.0

        # GEX adjustment
        if net_gex > 5.0:
            score    = min(88.0, score + 5.0)
        elif net_gex < -5.0:
            score    = max(12.0, score - 5.0)
            prob_adj -= 0.02

        # 0DTE has stronger signal than 1DTE
        if dte == 0:
            score    = 50 + (score - 50) * 1.2
            prob_adj = prob_adj * 1.2
            score    = float(np.clip(score, 10, 90))
            prob_adj = float(np.clip(prob_adj, -0.10, 0.10))

        logger.info(
            f"[0DTE] {ticker}: dte={dte} regime={regime} "
            f"oi_ratio={oi_ratio:+.2f} gex={net_gex:+.1f} score={score:.1f}"
        )
        return ZeroDTEResult(
            score=round(score, 1),
            prob_adjustment=round(prob_adj, 4),
            net_call_oi=round(net_call_oi, 0),
            net_gex=round(net_gex, 2),
            call_put_premium_ratio=round(cp_ratio, 3),
            days_to_expiry=dte,
            regime=regime,
            n_strikes=n_atm,
        )

    except Exception as e:
        logger.warning(f"[0DTE] {ticker}: error — {e}")
        return None
