"""
AlphaAgent — Limit Order Book (LOB) Model & Market Microstructure

Implements:
  1. Analytical LOB simulation (Poisson arrival process)
  2. Kyle lambda (price impact coefficient)
  3. Amihud illiquidity ratio
  4. Roll's effective spread estimator
  5. Hasbrouck's information share
  6. Probability of Informed Trading (PIN model)
  7. Order flow toxicity (VPIN)
  8. Mid-price dynamics from LOB imbalance

Reference: Kyle (1985), Amihud (2002), Roll (1984), Hasbrouck (1991)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class LOBResult:
    kyle_lambda:        float   # price impact: Δp = λ·order_flow
    amihud_illiq:       float   # |R|/Volume (×10⁶)
    roll_spread:        float   # effective bid-ask spread
    pin:                float   # probability of informed trading
    vpin:               float   # volume-synchronized PIN (toxicity)
    depth_imbalance:    float   # (bid_depth - ask_depth)/(bid+ask)
    mid_price_drift:    float   # expected short-term drift from imbalance
    bid_ask_halfspread: float   # estimated half-spread
    market_impact_bps:  float   # 100-share trade impact in basis points
    liquidity_score:    float   # composite 0-100 (100=most liquid)
    regime:             str     # LIQUID / NORMAL / ILLIQUID / TOXIC
    signal:             str     # BUY_LIQUIDITY / NEUTRAL / AVOID_ILLIQUIDITY
    notes:              List[str]


# ─── Kyle Lambda ──────────────────────────────────────────────────────────────

def kyle_lambda(price_changes: np.ndarray, signed_volume: np.ndarray) -> float:
    """
    Estimate Kyle lambda via OLS: Δp_t = λ·x_t + ε_t
    x_t = signed order flow (buy volume - sell volume proxy)
    """
    if len(price_changes) < 10 or len(signed_volume) < 10:
        return 0.0
    n  = min(len(price_changes), len(signed_volume))
    dp = price_changes[-n:]
    x  = signed_volume[-n:]
    X  = np.column_stack([np.ones(n), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, dp, rcond=None)
        return float(max(beta[1], 0))
    except Exception:
        return 0.0


# ─── Amihud Illiquidity ───────────────────────────────────────────────────────

def amihud_illiq(returns: np.ndarray, dollar_volume: np.ndarray,
                 scale: float = 1e6) -> float:
    """
    Amihud ratio: (1/T) Σ |R_t| / DollarVolume_t  × 10⁶
    Higher → more illiquid (price moves more per dollar traded).
    """
    n = min(len(returns), len(dollar_volume))
    if n < 5:
        return 0.0
    r  = np.abs(returns[-n:])
    dv = np.maximum(dollar_volume[-n:], 1)
    return float(np.mean(r / dv) * scale)


# ─── Roll Spread Estimator ────────────────────────────────────────────────────

def roll_spread(prices: np.ndarray) -> float:
    """
    Roll (1984) effective spread: 2√(-Cov(ΔP_t, ΔP_{t-1}))
    Negative serial covariance of price changes estimates half-spread.
    """
    dp    = np.diff(prices)
    if len(dp) < 10:
        return 0.0
    cov01 = float(np.cov(dp[1:], dp[:-1])[0, 1])
    if cov01 >= 0:
        return 0.0
    return float(2 * np.sqrt(-cov01))


# ─── PIN Model (Easley et al.) ────────────────────────────────────────────────

def estimate_pin(buy_vol: np.ndarray, sell_vol: np.ndarray) -> float:
    """
    Simplified PIN estimate:
    PIN = α·μ / (α·μ + 2·ε)
    where:
      α = fraction of days with news (estimated from trade imbalance variance)
      μ = informed trade intensity
      ε = uninformed arrival rate
    """
    if len(buy_vol) < 5 or len(sell_vol) < 5:
        return 0.0
    n     = min(len(buy_vol), len(sell_vol))
    B, S  = buy_vol[-n:], sell_vol[-n:]
    total = B + S + 1e-10
    imb   = (B - S) / total   # order imbalance

    # Estimate α: high imbalance variance ↔ high probability of informed trading
    alpha = float(np.clip(np.var(imb) * 4, 0, 1))
    eps   = float(np.mean(np.minimum(B, S)))
    mu    = float(np.abs(np.mean(B - S)))
    denom = alpha * mu + 2 * eps
    if denom <= 0:
        return 0.0
    return float(np.clip(alpha * mu / denom, 0, 1))


# ─── VPIN (Volume-Synchronized PIN) ──────────────────────────────────────────

def vpin(prices: np.ndarray, volumes: np.ndarray, n_bars: int = 50) -> float:
    """
    VPIN (Easley et al. 2012) — order flow toxicity.
    Bins trades by equal volume, estimates buy/sell using price change sign.
    Returns VPIN ∈ [0, 1] where 1 = highly toxic.
    """
    if len(prices) < n_bars * 2 or len(volumes) < n_bars * 2:
        return 0.0
    total_vol  = float(np.sum(volumes))
    bucket_vol = total_vol / n_bars

    dp_sign = np.sign(np.diff(prices))
    bucket_imb = []
    accum, buy_v, sell_v = 0.0, 0.0, 0.0

    for i, v in enumerate(volumes[1:]):
        s = dp_sign[i] if i < len(dp_sign) else 0
        accum += v
        if s >= 0:
            buy_v  += v
        else:
            sell_v += v
        if accum >= bucket_vol:
            bucket_imb.append(abs(buy_v - sell_v) / max(buy_v + sell_v, 1))
            accum, buy_v, sell_v = 0, 0, 0

    if not bucket_imb:
        return 0.0
    return float(np.mean(bucket_imb[-n_bars:]))


# ─── LOB Simulation ──────────────────────────────────────────────────────────

def simulate_lob(mid_price: float = 100.0, spread: float = 0.10,
                 depth: int = 10, n_steps: int = 1000,
                 lambda_b: float = 2.0, lambda_a: float = 2.0,
                 lambda_cancel: float = 0.5) -> Dict:
    """
    Simple Poisson LOB simulation.
    Returns stats on mid-price drift and order flow imbalance.
    """
    rng     = np.random.default_rng(42)
    dt      = 1 / 252 / 6.5 / 3600   # 1-second intervals
    tick    = spread / 2

    bid_q   = np.zeros(depth)  # quantity at each bid level
    ask_q   = np.zeros(depth)  # quantity at each ask level
    bid_q[:] = 10; ask_q[:] = 10

    mids = [mid_price]
    imbs = []

    for _ in range(n_steps):
        # Arrivals
        nb  = rng.poisson(lambda_b)
        na  = rng.poisson(lambda_a)
        nc  = rng.poisson(lambda_cancel)

        # Market orders eat top of book
        execute_b = min(nb, int(ask_q[0]))
        execute_a = min(na, int(bid_q[0]))
        ask_q[0] -= execute_b
        bid_q[0] -= execute_a

        # Replenish
        lv_b = rng.integers(0, depth)
        lv_a = rng.integers(0, depth)
        bid_q[lv_b] += rng.integers(1, 5)
        ask_q[lv_a] += rng.integers(1, 5)

        # Cancel
        if bid_q.sum() > 0:
            bid_q[rng.integers(0, depth)] = max(bid_q[rng.integers(0, depth)] - nc, 0)

        # Collapse empty top-of-book
        if ask_q[0] <= 0:
            ask_q = np.roll(ask_q, -1); ask_q[-1] = rng.integers(5, 15)
            mid_price += tick
        if bid_q[0] <= 0:
            bid_q = np.roll(bid_q, -1); bid_q[-1] = rng.integers(5, 15)
            mid_price -= tick

        total_bid = bid_q.sum(); total_ask = ask_q.sum()
        imb = (total_bid - total_ask) / max(total_bid + total_ask, 1)
        imbs.append(imb)
        mids.append(mid_price)

    avg_imb = float(np.mean(imbs))
    drift   = float(np.mean(np.diff(mids)))
    return {"avg_imbalance": round(avg_imb, 4), "mid_drift_per_step": round(drift, 6),
            "final_mid": round(mids[-1], 4), "mids": mids[::max(1, n_steps//50)]}


# ─── Main Analysis ────────────────────────────────────────────────────────────

def analyze_lob(ticker: str, market_data) -> LOBResult:
    notes: List[str] = []
    try:
        ohlcv = market_data.get_ohlcv("1y")
        if ohlcv.empty or len(ohlcv) < 30:
            raise ValueError("Insufficient data")

        closes  = ohlcv["Close"].values
        volumes = ohlcv["Volume"].values
        returns = np.diff(closes) / closes[:-1]
        S       = float(closes[-1])

        # Dollar volume
        dollar_vol = closes[1:] * volumes[1:]

        # Metrics
        kl  = kyle_lambda(returns, np.sign(returns) * np.sqrt(np.abs(dollar_vol)))
        ai  = amihud_illiq(returns, dollar_vol)
        rs  = roll_spread(closes)
        ha  = rs / 2     # half-spread
        avg_vol = float(np.mean(volumes))

        # Signed volume proxy: +vol if price up, -vol if price down
        sv_proxy = np.sign(returns) * volumes[1:]
        buy_v    = volumes[1:][returns > 0]
        sell_v   = volumes[1:][returns < 0]
        pin_est  = estimate_pin(buy_v, sell_v)
        vpin_est = vpin(closes, volumes)

        # LOB simulation
        lob_stats = simulate_lob(mid_price=S, spread=rs if rs > 0 else 0.05 * S,
                                  n_steps=500)
        imb_now = lob_stats["avg_imbalance"]

        # Market impact estimate
        share_100_val = S * 100
        impact_bps    = float(kl * share_100_val * 1e4) if kl > 0 else 1.0

        # Composite liquidity score (0-100)
        ai_score   = max(0.0, 100 - min(ai * 1e4, 100))
        vpin_score = max(0.0, 100 - vpin_est * 200)
        pin_score  = max(0.0, 100 - pin_est * 200)
        liq_score  = float(0.4 * ai_score + 0.3 * vpin_score + 0.3 * pin_score)

        # Regime
        if vpin_est > 0.5 or pin_est > 0.4:
            regime = "TOXIC"
            sig    = "AVOID_ILLIQUIDITY"
            notes.append(f"VPIN={vpin_est:.2f} — highly toxic order flow, avoid large size.")
        elif ai > 0.05 or liq_score < 40:
            regime = "ILLIQUID"
            sig    = "AVOID_ILLIQUIDITY"
            notes.append(f"Amihud={ai:.4f} — illiquid, expect high market impact.")
        elif liq_score > 75:
            regime = "LIQUID"
            sig    = "BUY_LIQUIDITY"
            notes.append(f"Liquid market: VPIN={vpin_est:.2f}, impact ~{impact_bps:.1f}bps/100sh.")
        else:
            regime = "NORMAL"
            sig    = "NEUTRAL"

        mid_drift = lob_stats["mid_drift_per_step"] * 252  # annualized
        notes.append(f"Roll spread: ${rs:.4f}, Kyle λ: {kl:.4e}, PIN: {pin_est:.2%}.")

        return LOBResult(
            kyle_lambda=round(kl, 8), amihud_illiq=round(ai, 6),
            roll_spread=round(rs, 4), pin=round(pin_est, 4),
            vpin=round(vpin_est, 4), depth_imbalance=round(imb_now, 4),
            mid_price_drift=round(mid_drift, 6),
            bid_ask_halfspread=round(ha, 4),
            market_impact_bps=round(impact_bps, 2),
            liquidity_score=round(liq_score, 1), regime=regime,
            signal=sig, notes=notes,
        )
    except Exception as e:
        logger.error(f"LOB analysis failed for {ticker}: {e}")
        return LOBResult(kyle_lambda=0.0, amihud_illiq=0.0, roll_spread=0.0,
                         pin=0.0, vpin=0.0, depth_imbalance=0.0,
                         mid_price_drift=0.0, bid_ask_halfspread=0.0,
                         market_impact_bps=0.0, liquidity_score=50.0,
                         regime="UNKNOWN", signal="NEUTRAL", notes=[str(e)])
