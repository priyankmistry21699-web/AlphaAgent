"""
AlphaAgent — Multifractal Volatility Models

Implements:
  1. Markov-Switching Multifractal (MSM) — Calvet & Fisher (2004)
     k volatility components, each switching independently
     σ_t = σ₀ · √(M₁_t · M₂_t · … · M_k_t)
     M_i switches with prob γ_i = 1-(1-γ_k)^{b^{i-k}}

  2. Multiscale correlation analysis
  3. Hurst exponent via detrended fluctuation analysis (DFA)
  4. Multifractal spectrum via wavelet leaders
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class MSMResult:
    k_components:    int
    b_parameter:     float        # multiplier base
    m0:              float        # high-volatility multiplier value
    sigma_bar:       float        # baseline daily vol
    gamma_k:         float        # highest-freq switch prob
    forecast_vol_1d: float
    forecast_vol_5d: float
    regime_probs:    List[float]  # prob of each vol state (sorted high→low)
    current_vol:     float        # current estimated vol
    vol_persistence: float        # Hurst exponent (DFA)
    multifractal_width: float     # width of singularity spectrum
    signal:          str          # HIGH_VOL_REGIME / NORMAL / LOW_VOL_REGIME
    notes:           List[str]


# ─── DFA (Detrended Fluctuation Analysis) ────────────────────────────────────

def dfa_hurst(x: np.ndarray, min_box: int = 4, max_box: int = None) -> float:
    """
    DFA Hurst exponent. H > 0.5 = persistent, H < 0.5 = anti-persistent.
    """
    n = len(x)
    if max_box is None:
        max_box = n // 4
    max_box = min(max_box, n // 4)

    y      = np.cumsum(x - np.mean(x))
    boxes  = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 20).astype(int))
    F      = []
    for box in boxes:
        n_full = n // box
        if n_full < 2:
            continue
        y_cut = y[:n_full * box].reshape(n_full, box)
        t_arr = np.arange(box)
        rms_sq = []
        for seg in y_cut:
            poly  = np.polyfit(t_arr, seg, 1)
            trend = np.polyval(poly, t_arr)
            rms_sq.append(np.mean((seg - trend)**2))
        F.append((box, np.sqrt(np.mean(rms_sq))))

    if len(F) < 4:
        return 0.5
    log_s = np.log([f[0] for f in F])
    log_f = np.log([f[1] for f in F])
    H, _  = np.polyfit(log_s, log_f, 1)
    return float(np.clip(H, 0.0, 1.0))


# ─── Multifractal spectrum (structure functions) ──────────────────────────────

def multifractal_spectrum(returns: np.ndarray,
                          q_range: List[float] = None) -> Tuple[np.ndarray, float]:
    """
    Compute multifractal spectrum width via structure functions.
    Returns (tau_q, alpha_range) where alpha_range = width of f(α).
    """
    if q_range is None:
        q_range = np.linspace(-4, 4, 17).tolist()
    q_arr = np.array(q_range)

    # Use absolute returns at multiple scales
    scales = [1, 2, 4, 8, 16, 32]
    tau    = []
    abs_r  = np.abs(returns)

    for q in q_arr:
        zeta_q = []
        for s in scales:
            if s >= len(abs_r):
                continue
            coarse = np.array([abs_r[i:i+s].mean() for i in range(0, len(abs_r)-s, s)])
            if len(coarse) < 2:
                continue
            moment = np.mean(np.maximum(coarse, 1e-10)**q)
            zeta_q.append((np.log(s), np.log(max(moment, 1e-30))))
        if len(zeta_q) >= 3:
            xs, ys = zip(*zeta_q)
            slope, _ = np.polyfit(xs, ys, 1)
            tau.append(slope)
        else:
            tau.append(q * 0.5)  # monofractal fallback

    tau_arr = np.array(tau)
    # Hölder exponent α(q) = dτ/dq
    alpha = np.gradient(tau_arr, np.diff(q_arr).mean())
    alpha_range = float(alpha.max() - alpha.min())
    return tau_arr, alpha_range


# ─── MSM Model ───────────────────────────────────────────────────────────────

class MSMModel:
    """
    Markov-Switching Multifractal (k components).
    Simplified version suitable for real-time signal generation.
    """

    def __init__(self, k: int = 8, b: float = 2.5, m0: float = 1.6,
                 sigma: float = 0.01, gamma_k: float = 0.5):
        self.k       = k
        self.b       = b
        self.m0      = m0       # high-vol multiplier
        self.sigma   = sigma    # baseline daily vol
        self.gamma_k = gamma_k

    def _switch_probs(self) -> np.ndarray:
        """Transition probabilities γ_i for each component."""
        return np.array([1 - (1 - self.gamma_k) ** (self.b ** (i - self.k + 1))
                         for i in range(self.k)])

    def _state_vols(self) -> np.ndarray:
        """All 2^k possible vol states."""
        n_states = 2**self.k
        states   = np.zeros(n_states)
        for s in range(n_states):
            prod = 1.0
            for i in range(self.k):
                bit = (s >> i) & 1
                prod *= self.m0 if bit else (2 - self.m0)
            states[s] = prod
        return states * self.sigma

    def filter(self, returns: np.ndarray) -> np.ndarray:
        """
        Hamilton filter — returns filtered regime probabilities.
        Returns array of shape (T, 2^k).
        """
        n_states = 2**self.k
        gammas   = self._switch_probs()
        vols     = self._state_vols()

        # Transition matrix (independent switching across components)
        P = np.ones((n_states, n_states))
        for s in range(n_states):
            for s2 in range(n_states):
                for i in range(self.k):
                    b_s  = (s  >> i) & 1
                    b_s2 = (s2 >> i) & 1
                    if b_s == b_s2:
                        P[s, s2] *= (1 - gammas[i])
                    else:
                        P[s, s2] *= gammas[i] * 0.5

        pi    = np.full(n_states, 1.0 / n_states)
        probs = [pi]
        eps   = 1e-10
        for r in returns:
            # Likelihood
            lk  = (1 / (np.sqrt(2 * np.pi) * vols + eps)) * np.exp(-0.5 * (r / (vols + eps))**2)
            up  = lk * (P.T @ pi)
            pi  = up / max(up.sum(), eps)
            probs.append(pi.copy())

        return np.array(probs)

    def forecast_vol(self, returns: np.ndarray, horizon: int = 5) -> float:
        """Expected forward vol over `horizon` days."""
        vols     = self._state_vols()
        probs    = self.filter(returns)[-1]   # current state probs
        exp_var  = float(np.dot(probs, vols**2))
        return float(np.sqrt(exp_var * horizon / horizon))   # daily vol

    def fit(self, returns: np.ndarray, max_iter: int = 50) -> float:
        """Simple EM to fit sigma and m0 from data."""
        def _neg_ll(params):
            self.sigma, self.m0 = max(params[0], 1e-5), max(params[1], 1.001)
            probs = self.filter(returns)
            vols  = self._state_vols()
            ll    = 0.0
            for i, r in enumerate(returns):
                pi = probs[i]
                lk = np.sum(pi * (1 / (np.sqrt(2 * np.pi) * vols + 1e-10))
                            * np.exp(-0.5 * (r / (vols + 1e-10))**2))
                ll += np.log(max(lk, 1e-30))
            return -ll

        rv = float(np.std(returns))
        x0 = [rv, min(self.m0, 1.9)]
        res = minimize(_neg_ll, x0, method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4})
        self.sigma, self.m0 = float(res.x[0]), float(res.x[1])
        return float(-res.fun)


def analyze_multifractal(ticker: str, market_data) -> MSMResult:
    notes: List[str] = []
    try:
        ohlcv = market_data.get_ohlcv("2y")
        if ohlcv.empty or len(ohlcv) < 60:
            raise ValueError("Need ≥60 days")

        returns = ohlcv["Close"].pct_change().dropna().values
        rv      = float(np.std(returns) * np.sqrt(252))
        H       = dfa_hurst(returns)
        _, mf_width = multifractal_spectrum(returns[-200:] if len(returns) > 200 else returns)

        # Fit MSM
        k  = 6
        msm = MSMModel(k=k, sigma=rv/np.sqrt(252))
        try:
            msm.fit(returns[-252:] if len(returns) > 252 else returns, max_iter=30)
        except Exception:
            pass

        fv1d = msm.forecast_vol(returns[-60:], 1)
        fv5d = msm.forecast_vol(returns[-60:], 5)

        # State probs
        state_probs = msm.filter(returns[-60:])[-1]
        vols_states = msm._state_vols()
        high_idx    = np.argsort(vols_states)[::-1][:8]
        regime_probs = sorted(state_probs[high_idx].tolist(), reverse=True)[:4]

        current_vol = fv1d * np.sqrt(252)

        if current_vol > rv * 1.3:
            sig = "HIGH_VOL_REGIME"
            notes.append(f"MSM: current vol ({current_vol:.1%}) >> historical ({rv:.1%}) — elevated risk regime.")
        elif current_vol < rv * 0.7:
            sig = "LOW_VOL_REGIME"
            notes.append(f"MSM: low vol regime ({current_vol:.1%}) — potential for vol expansion.")
        else:
            sig = "NORMAL"

        notes.append(f"DFA Hurst H={H:.3f} ({'persistent' if H>0.5 else 'anti-persistent'} vol clustering).")
        if mf_width > 0.5:
            notes.append(f"Multifractal width={mf_width:.2f} — complex multiscale dynamics detected.")

        return MSMResult(
            k_components=k, b_parameter=msm.b, m0=round(msm.m0, 4),
            sigma_bar=round(msm.sigma, 6), gamma_k=round(msm.gamma_k, 4),
            forecast_vol_1d=round(fv1d, 4), forecast_vol_5d=round(fv5d, 4),
            regime_probs=[round(p, 4) for p in regime_probs],
            current_vol=round(current_vol, 4), vol_persistence=round(H, 4),
            multifractal_width=round(mf_width, 4), signal=sig, notes=notes,
        )
    except Exception as e:
        logger.error(f"Multifractal analysis failed for {ticker}: {e}")
        return MSMResult(k_components=6, b_parameter=2.5, m0=1.6, sigma_bar=0.01,
                         gamma_k=0.5, forecast_vol_1d=0.0, forecast_vol_5d=0.0,
                         regime_probs=[], current_vol=0.0, vol_persistence=0.5,
                         multifractal_width=0.0, signal="UNKNOWN", notes=[str(e)])
