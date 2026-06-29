"""
AlphaAgent — Post-Earnings Announcement Drift (PEAD)

Theory: Stocks that beat (miss) earnings estimates continue drifting
up (down) for 30-60 days after the report.  This is one of the most
robust, academically-replicated anomalies in finance.

Reference: Bernard & Thomas (1989), Ball & Brown (1968)

Implementation:
  1. Fetch the last earnings date + actual vs estimated EPS from yfinance
  2. Compute standardised unexpected earnings (SUE):
       SUE = (actual_EPS - estimated_EPS) / std(historical_eps_surprises)
  3. If within the PEAD drift window (0–60 days), score accordingly:
       - Day 0–5   : strongest signal (market initial under-reaction)
       - Day 5–20  : strong signal
       - Day 20–40 : moderate drift
       - Day 40–60 : decaying signal
  4. Return a PEADResult with score (0–100) and probability adjustment
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

_DRIFT_WINDOW_DAYS = 60     # max drift window
_DECAY_START_DAYS  = 20     # decay begins after this many days
_MIN_SUE_SIGNAL    = 0.3    # minimum |SUE| to generate a signal


@dataclass
class PEADResult:
    score: float                    # 0-100 (50 = neutral)
    prob_adjustment: float          # direct probability nudge (-0.12 to +0.12)
    sue: float                      # Standardised Unexpected Earnings
    days_since_earnings: int        # days elapsed since report
    direction: str                  # BEAT / MISS / NEUTRAL
    decay_factor: float             # 1.0 = fresh, 0.0 = expired
    earnings_date: Optional[date]
    actual_eps: Optional[float]
    estimated_eps: Optional[float]


def compute_pead(ticker: str) -> Optional[PEADResult]:
    """
    Compute the PEAD signal for a given ticker.
    Returns None if no usable earnings data is available.
    """
    try:
        t = yf.Ticker(ticker)

        # ── 1. Get earnings history ───────────────────────────────────────
        try:
            earnings = t.earnings_history
        except Exception:
            earnings = None

        if earnings is None or earnings.empty:
            # Try quarterly earnings as fallback
            try:
                earnings = t.quarterly_earnings
            except Exception:
                pass

        if earnings is None or earnings.empty:
            logger.debug(f"[PEAD] {ticker}: no earnings history")
            return None

        # ── 2. Find most recent earnings release ─────────────────────────
        # yfinance earnings_history columns: Earnings Date, EPS Estimate, Reported EPS, Surprise(%)
        recent = None
        earnings_date = None
        actual_eps = None
        estimated_eps = None
        surprise_pct = None

        if hasattr(earnings, "columns"):
            cols = [c.lower().replace(" ", "_") for c in earnings.columns]
            earnings.columns = cols
            col_map = {c: c for c in cols}

            # Map possible column names
            date_col    = next((c for c in cols if "date" in c), None)
            actual_col  = next((c for c in cols if "reported" in c or "actual" in c), None)
            est_col     = next((c for c in cols if "estimate" in c), None)
            surp_col    = next((c for c in cols if "surprise" in c), None)

            if date_col:
                earnings[date_col] = pd.to_datetime(earnings[date_col], errors="coerce")
                past = earnings[earnings[date_col] <= pd.Timestamp.now(tz="UTC")]
                if not past.empty:
                    recent = past.iloc[-1]
                    raw_date = recent[date_col]
                    earnings_date = raw_date.date() if hasattr(raw_date, "date") else None

            if recent is not None:
                actual_eps    = float(recent[actual_col]) if actual_col and pd.notna(recent[actual_col]) else None
                estimated_eps = float(recent[est_col])    if est_col    and pd.notna(recent[est_col])    else None
                surprise_pct  = float(recent[surp_col])   if surp_col   and pd.notna(recent[surp_col])   else None

        if earnings_date is None:
            logger.debug(f"[PEAD] {ticker}: could not parse earnings date")
            return None

        # ── 3. Days since earnings ────────────────────────────────────────
        days_since = (date.today() - earnings_date).days
        if days_since < 0 or days_since > _DRIFT_WINDOW_DAYS:
            logger.debug(f"[PEAD] {ticker}: earnings {days_since}d ago — outside window")
            return PEADResult(
                score=50.0, prob_adjustment=0.0, sue=0.0,
                days_since_earnings=days_since, direction="NEUTRAL",
                decay_factor=0.0, earnings_date=earnings_date,
                actual_eps=actual_eps, estimated_eps=estimated_eps,
            )

        # ── 4. Compute SUE ────────────────────────────────────────────────
        sue = 0.0
        if surprise_pct is not None:
            # surprise_pct from yfinance is already (actual - estimate) / |estimate| * 100
            sue = surprise_pct / 100.0
        elif actual_eps is not None and estimated_eps is not None and estimated_eps != 0:
            sue = (actual_eps - estimated_eps) / abs(estimated_eps)

        # Normalise SUE across historical surprises for this ticker
        try:
            hist_eps = t.earnings_history
            if hist_eps is not None and not hist_eps.empty:
                surp_col_raw = next(
                    (c for c in hist_eps.columns if "surprise" in c.lower()), None
                )
                if surp_col_raw:
                    surprises = hist_eps[surp_col_raw].dropna().astype(float)
                    if len(surprises) >= 4:
                        std = surprises.std()
                        if std > 0:
                            sue = (sue * 100 - surprises.mean()) / std
        except Exception:
            pass  # keep raw sue

        # ── 5. Decay function ─────────────────────────────────────────────
        # Linear decay after day 20, zero at day 60
        if days_since <= _DECAY_START_DAYS:
            decay = 1.0
        else:
            decay = max(0.0, 1.0 - (days_since - _DECAY_START_DAYS) /
                        (_DRIFT_WINDOW_DAYS - _DECAY_START_DAYS))

        # ── 6. Score ──────────────────────────────────────────────────────
        # SUE > 0 → beat → bullish drift; SUE < 0 → miss → bearish drift
        if abs(sue) < _MIN_SUE_SIGNAL:
            score = 50.0
            prob_adj = 0.0
            direction = "NEUTRAL"
        elif sue > 0:
            # Beat: score 65–88 depending on magnitude and decay
            raw_score = min(88.0, 62.0 + sue * 12.0)
            score = 50.0 + (raw_score - 50.0) * decay
            prob_adj = min(0.12, sue * 0.05 * decay)
            direction = "BEAT"
        else:
            raw_score = max(12.0, 38.0 + sue * 12.0)
            score = 50.0 - (50.0 - raw_score) * decay
            prob_adj = max(-0.12, sue * 0.05 * decay)
            direction = "MISS"

        logger.info(
            f"[PEAD] {ticker}: {direction} | SUE={sue:.2f} | "
            f"days={days_since} | decay={decay:.2f} | score={score:.1f}"
        )
        return PEADResult(
            score=round(score, 1),
            prob_adjustment=round(prob_adj, 4),
            sue=round(sue, 3),
            days_since_earnings=days_since,
            direction=direction,
            decay_factor=round(decay, 3),
            earnings_date=earnings_date,
            actual_eps=actual_eps,
            estimated_eps=estimated_eps,
        )

    except Exception as e:
        logger.warning(f"[PEAD] {ticker}: error — {e}")
        return None


# pandas is needed for date parsing above but was not imported
import pandas as pd  # noqa: E402 — intentional late import after body
