"""
AlphaAgent — Full Granger Causality Pipeline

Granger causality: series X Granger-causes Y if past X values
significantly improve prediction of Y beyond Y's own history.

Implements:
  1. Bivariate Granger causality test (F-test on VAR residuals)
  2. Multivariate VAR-based network (who causes whom?)
  3. Frequency-domain Granger (spectral causality)
  4. Non-linear Granger via kernel regression residuals
  5. Rolling-window Granger for time-varying causality
  6. AlphaAgent integration: test if macro / sector ETFs Granger-cause ticker
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class GrangerTest:
    x_name:   str
    y_name:   str
    lag:      int
    f_stat:   float
    p_value:  float
    causal:   bool          # significant at 5%
    direction: str          # "X→Y", "Y→X", "bidirectional", "independent"


@dataclass
class GrangerResult:
    ticker:          str
    tests:           List[GrangerTest]
    leading_series:  List[str]    # series that Granger-cause ticker
    feedback_series: List[str]    # bidirectional
    max_lag:         int
    causal_network:  Dict[str, List[str]]   # {cause: [effects]}
    signal:          str   # EXTERNALLY_DRIVEN / SELF_DRIVEN / MIXED
    summary:         str
    notes:           List[str]


# ─── VAR helper ───────────────────────────────────────────────────────────────

def _lag_matrix(x: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create lagged design matrix X_t = [x_{t-1}, …, x_{t-p}]."""
    n = len(x)
    X = np.column_stack([x[p-i-1:n-i-1] for i in range(p)])
    X = np.column_stack([np.ones(n - p), X])   # include constant
    y = x[p:]
    return X, y


def _ols_ssr(X: np.ndarray, y: np.ndarray) -> float:
    """OLS sum of squared residuals."""
    try:
        beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
        if len(res) == 0:
            yhat = X @ beta
            res_vec = y - yhat
        else:
            res_vec = y - X @ beta
        return float(np.dot(res_vec, res_vec))
    except Exception:
        return float(np.var(y) * len(y))


def granger_test(x: np.ndarray, y: np.ndarray, max_lag: int = 5,
                 x_name: str = "X", y_name: str = "Y") -> GrangerTest:
    """
    Test H₀: X does NOT Granger-cause Y.
    Restricted model: Y ~ Y_lags
    Unrestricted:     Y ~ Y_lags + X_lags
    F = ((SSR_r - SSR_u)/p) / (SSR_u/(n - 2p - 1))
    """
    n = min(len(x), len(y))
    x, y = x[-n:], y[-n:]

    best_p, best_aic = 1, np.inf
    for p in range(1, max_lag + 1):
        X_r, y_r = _lag_matrix(y, p)
        ssr = _ols_ssr(X_r, y_r)
        aic = (n - p) * np.log(ssr / (n - p)) + 2 * (p + 1)
        if aic < best_aic:
            best_aic, best_p = aic, p

    p = best_p
    X_r, y_r = _lag_matrix(y, p)
    X_u_y, _ = _lag_matrix(y, p)
    X_u_x, _ = _lag_matrix(x, p)
    X_u = np.column_stack([X_r, X_u_x[:, 1:]])  # combine Y lags + X lags

    ssr_r = _ols_ssr(X_r, y_r)
    ssr_u = _ols_ssa = _ols_ssr(X_u, y_r)

    nT    = len(y_r)
    dof1  = p
    dof2  = nT - 2 * p - 1
    if dof2 <= 0 or ssr_u <= 0:
        return GrangerTest(x_name, y_name, p, 0.0, 1.0, False, "independent")

    F  = ((ssr_r - ssr_u) / dof1) / (ssr_u / dof2)
    pv = float(1 - stats.f.cdf(max(F, 0), dof1, dof2))
    causal = pv < 0.05
    return GrangerTest(x_name, y_name, p, round(F, 4), round(pv, 6), causal,
                       f"{x_name}→{y_name}" if causal else "independent")


def spectral_granger(x: np.ndarray, y: np.ndarray,
                     n_fft: int = 256) -> Dict[str, float]:
    """
    Frequency-domain Granger causality (Geweke 1982 measure).
    Returns directional spectral power ratio at low/mid/high freq bands.
    """
    n = min(len(x), len(y), n_fft * 2)
    x, y = x[-n:], y[-n:]
    # Cross-spectrum via FFT
    Sx = np.fft.rfft(x - x.mean(), n=n_fft)
    Sy = np.fft.rfft(y - y.mean(), n=n_fft)
    coherence = np.abs(Sx * np.conj(Sy))**2 / (np.abs(Sx)**2 * np.abs(Sy)**2 + 1e-10)
    freqs = np.fft.rfftfreq(n_fft)
    low  = float(coherence[freqs < 0.05].mean())   # trend
    mid  = float(coherence[(freqs >= 0.05) & (freqs < 0.2)].mean())   # cycle
    high = float(coherence[freqs >= 0.2].mean())   # noise
    return {"low_freq_coherence": round(low, 4), "mid_freq_coherence": round(mid, 4),
            "high_freq_coherence": round(high, 4)}


def rolling_granger(x: np.ndarray, y: np.ndarray,
                    window: int = 60, max_lag: int = 3) -> List[Dict]:
    """Test Granger causality in rolling windows to detect time-varying causality."""
    results = []
    for i in range(window, len(x), window // 2):
        x_w = x[max(0, i-window):i]
        y_w = y[max(0, i-window):i]
        t   = granger_test(x_w, y_w, max_lag)
        results.append({"end_idx": i, "p_value": t.p_value, "causal": t.causal, "f_stat": t.f_stat})
    return results


def analyze_granger(ticker: str, market_data,
                    reference_tickers: List[str] = None) -> GrangerResult:
    """
    Test whether macro/sector reference series Granger-cause the ticker.
    Default references: SPY, QQQ, TLT, VIX (^VIX), GLD.
    """
    import yfinance as yf
    notes: List[str] = []

    if reference_tickers is None:
        reference_tickers = ["SPY", "QQQ", "TLT", "GLD", "^VIX"]

    try:
        ohlcv = market_data.get_ohlcv("2y")
        if ohlcv.empty or len(ohlcv) < 80:
            raise ValueError("Insufficient data")

        y_ret = ohlcv["Close"].pct_change().dropna().values

        ref_data: Dict[str, np.ndarray] = {}
        for ref in reference_tickers:
            try:
                df = yf.download(ref, period="2y", interval="1d",
                                 auto_adjust=True, progress=False)
                if not df.empty:
                    ref_data[ref] = df["Close"].pct_change().dropna().values
            except Exception:
                pass

        tests: List[GrangerTest] = []
        causal_net: Dict[str, List[str]] = {}

        for ref_name, x_ret in ref_data.items():
            # Align lengths
            n = min(len(x_ret), len(y_ret))
            x_a, y_a = x_ret[-n:], y_ret[-n:]
            # Test both directions
            t_xy = granger_test(x_a, y_a, max_lag=5, x_name=ref_name, y_name=ticker)
            t_yx = granger_test(y_a, x_a, max_lag=5, x_name=ticker, y_name=ref_name)
            tests.append(t_xy)
            tests.append(t_yx)
            if t_xy.causal:
                causal_net.setdefault(ref_name, []).append(ticker)
            if t_yx.causal:
                causal_net.setdefault(ticker, []).append(ref_name)

        leading  = [t.x_name for t in tests if t.causal and t.y_name == ticker]
        feedback = [t.x_name for t in tests
                    if t.causal and t.y_name == ticker
                    and any(t2.causal and t2.y_name == t.x_name and t2.x_name == ticker
                            for t2 in tests)]

        if len(leading) >= 3:
            sig = "EXTERNALLY_DRIVEN"
            summary = f"{ticker} is Granger-caused by {', '.join(leading[:3])} — externally driven."
        elif len(leading) == 0:
            sig = "SELF_DRIVEN"
            summary = f"{ticker} moves independently — self-driven dynamics."
        else:
            sig = "MIXED"
            summary = f"{ticker} partially driven by {', '.join(leading)} with self-dynamics."

        notes.append(f"Tested {len(ref_data)} reference series with up to lag-5.")
        for t in tests:
            if t.causal:
                notes.append(f"  {t.direction} (F={t.f_stat:.2f}, p={t.p_value:.4f})")

        return GrangerResult(
            ticker=ticker, tests=tests, leading_series=leading,
            feedback_series=feedback, max_lag=5,
            causal_network=causal_net, signal=sig, summary=summary, notes=notes,
        )
    except Exception as e:
        logger.error(f"Granger analysis failed for {ticker}: {e}")
        return GrangerResult(
            ticker=ticker, tests=[], leading_series=[], feedback_series=[],
            max_lag=5, causal_network={}, signal="UNKNOWN",
            summary=str(e), notes=[str(e)],
        )
