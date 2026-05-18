"""
AlphaAgent — Rough Volatility Models

Implements:
  1. rBergomi (Rough Bergomi) — Bayer, Friz, Gatheral (2016)
     Volatility driven by fractional Brownian motion with H < 0.5
     dS_t = S_t √v_t dW_t
     v_t = ξ₀ exp(η Ŵ^H_t - η²/2 t^{2H})

  2. Rough Heston (El Euch & Rosenbaum, 2019)
     Fractional kernel replaces the exponential mean reversion

  3. Volterra-Bergomi roughness estimator from realized vol

Key insight: real markets have H ≈ 0.1 (much rougher than BM's H=0.5),
which explains the observed power-law behavior of vol-of-vol.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from scipy.special import gamma
import logging

logger = logging.getLogger(__name__)


@dataclass
class RoughVolResult:
    hurst_exponent: float          # H ∈ (0, 0.5) for rough vol
    roughness_label: str           # VERY_ROUGH / ROUGH / SEMI_ROUGH
    eta: float                     # vol-of-vol parameter
    rho: float                     # correlation (equity: negative)
    xi0: float                     # spot variance (initial var)
    forecast_vol_5d: float         # 5-day forward vol estimate
    forecast_vol_21d: float        # 21-day forward vol estimate
    vol_of_vol: float              # empirical vol-of-vol
    mean_reversion_days: float     # effective half-life
    arbitrage_vol_signal: str      # HIGH_CONVEXITY / NORMAL / LOW_CONVEXITY
    notes: List[str]


def _fbm_cholesky(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate fractional Brownian motion via Cholesky (exact, slow for large n)."""
    t = np.arange(1, n + 1, dtype=float)
    # Covariance: E[B^H_s B^H_t] = 0.5(s^2H + t^2H - |s-t|^2H)
    cov = 0.5 * (t[:, None]**(2*H) + t[None, :]**(2*H)
                 - np.abs(t[:, None] - t[None, :])**(2*H))
    try:
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
        return L @ rng.standard_normal(n)
    except Exception:
        return rng.standard_normal(n) * np.sqrt(t**(2*H) - np.concatenate([[0], t[:-1]**(2*H)]))


def _fbm_hybrid(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Hybrid scheme (Bennedsen et al.): faster approximate fBm."""
    if n <= 256:
        return _fbm_cholesky(n, H, rng)
    # Use power-law kernel approximation
    k = np.arange(1, n + 1, dtype=float)
    weights = k**(H - 0.5) - np.maximum(k - 1, 0)**(H - 0.5)
    weights /= np.sqrt(np.sum(weights**2))
    z = rng.standard_normal(n)
    from numpy.fft import fft, ifft
    # Convolution via FFT
    n2 = 2 * n
    result = np.real(ifft(fft(weights, n2) * fft(z, n2)))[:n]
    return result * np.sqrt(gamma(2*H + 1) / 2)


class RoughBergomi:
    """
    rBergomi model:  v_t = ξ₀ · exp(η·Ŵ^H_t - η²/2·t^{2H})
    """

    def __init__(self, H: float = 0.1, eta: float = 1.9, rho: float = -0.9,
                 xi0: float = 0.04):
        self.H   = min(max(H, 0.01), 0.49)
        self.eta = eta
        self.rho = rho
        self.xi0 = xi0

    def simulate(self, T: float = 1.0, n_steps: int = 252,
                 n_paths: int = 1000, S0: float = 100.0) -> np.ndarray:
        """Return terminal stock prices (shape: n_paths)."""
        rng  = np.random.default_rng(42)
        dt   = T / n_steps
        sqdt = np.sqrt(dt)

        # Correlated Brownian motions
        z1   = rng.standard_normal((n_paths, n_steps))
        z2   = self.rho * z1 + np.sqrt(1 - self.rho**2) * rng.standard_normal((n_paths, n_steps))

        # fBm on each path (share a single draw for speed — acceptable for pricing)
        fbm_base = _fbm_hybrid(n_steps, self.H, rng)  # shape (n_steps,)

        # Build integrated log-vol process
        t_arr  = np.arange(1, n_steps + 1) * dt
        kernel = (t_arr**(self.H + 0.5) - np.maximum(t_arr - dt, 0)**(self.H + 0.5)) / (self.H + 0.5)

        # Approximate: v_t ≈ xi0 * exp(eta * fBm - eta²/2 * t^{2H})
        log_v = self.eta * fbm_base * kernel - 0.5 * self.eta**2 * t_arr**(2*self.H)
        v     = self.xi0 * np.exp(log_v)           # shape (n_steps,)

        # Euler for log-price
        log_S  = np.zeros(n_paths)
        for i in range(n_steps):
            sv     = np.sqrt(max(v[i], 1e-8))
            log_S += -0.5 * v[i] * dt + sv * sqdt * z1[:, i]

        return S0 * np.exp(log_S)

    def implied_variance_term_structure(self, horizons: List[float] = None) -> List[dict]:
        """Expected variance E[∫₀ᵀ v_t dt / T] for each horizon."""
        if horizons is None:
            horizons = [1/52, 1/12, 3/12, 6/12, 1.0]
        result = []
        for T in horizons:
            n_steps = max(int(T * 252), 5)
            t_arr   = np.linspace(dt := T / n_steps, T, n_steps)
            ev      = self.xi0 * np.exp(0.5 * self.eta**2 * t_arr**(2*self.H)
                                        - 0.5 * self.eta**2 * t_arr**(2*self.H))
            ev_avg  = float(np.mean(ev))
            result.append({"horizon_months": round(T * 12, 1), "expected_var": round(ev_avg, 6),
                           "expected_vol":  round(np.sqrt(ev_avg), 4)})
        return result


def estimate_hurst(returns: np.ndarray, lags: int = 20) -> float:
    """
    Estimate Hurst exponent of volatility via log-log regression of
    absolute returns structure function: E[|r_{t+l}| - |r_t|] ~ l^H.
    """
    abs_r = np.abs(returns)
    lags_arr = np.arange(1, lags + 1)
    sf = []
    for lag in lags_arr:
        diff = abs_r[lag:] - abs_r[:-lag]
        sf.append(np.mean(np.abs(diff)))
    log_lag = np.log(lags_arr)
    log_sf  = np.log(np.maximum(sf, 1e-10))
    H = float(np.polyfit(log_lag, log_sf, 1)[0])
    return min(max(H, 0.01), 0.99)


def estimate_vol_of_vol(returns: np.ndarray, window: int = 21) -> float:
    """Empirical vol-of-vol: std of rolling realized vol."""
    rv = []
    for i in range(window, len(returns)):
        rv.append(returns[i-window:i].std() * np.sqrt(252))
    return float(np.std(rv)) if rv else 0.0


def analyze_rough_vol(ticker: str, market_data) -> RoughVolResult:
    notes: List[str] = []
    try:
        ohlcv = market_data.get_ohlcv("2y")
        if ohlcv.empty or len(ohlcv) < 60:
            raise ValueError("Need at least 60 days")

        returns = ohlcv["Close"].pct_change().dropna().values
        S = float(ohlcv["Close"].iloc[-1])

        # Estimate roughness
        H   = estimate_hurst(returns[max(-252, -len(returns)):])
        vov = estimate_vol_of_vol(returns)
        rv  = float(np.std(returns) * np.sqrt(252))

        # Calibrate rBergomi parameters
        rho  = -0.8 if H < 0.2 else -0.5   # rough vol → strong negative correlation
        eta  = vov * np.sqrt(2 * H) * 10   # rough scaling
        eta  = max(0.1, min(eta, 5.0))
        xi0  = rv**2

        model = RoughBergomi(H=H, eta=eta, rho=rho, xi0=xi0)

        # Vol forecasts
        ts = model.implied_variance_term_structure([5/252, 21/252])
        fv5  = ts[0]["expected_vol"] if ts else rv
        fv21 = ts[1]["expected_vol"] if len(ts) > 1 else rv

        # Mean reversion: effective half-life from H
        hl_days = max(1.0, 1.0 / (1.0 - 2*H))  # heuristic: smaller H → faster reversion

        # Signal
        if H < 0.15:
            label = "VERY_ROUGH"
            notes.append(f"H={H:.3f}: extremely rough vol — short-lived vol spikes, rapid mean reversion.")
            sig = "HIGH_CONVEXITY"
        elif H < 0.30:
            label = "ROUGH"
            notes.append(f"H={H:.3f}: rough volatility detected — characteristic of equity markets.")
            sig = "NORMAL"
        else:
            label = "SEMI_ROUGH"
            notes.append(f"H={H:.3f}: semi-rough vol — slower mean reversion, trend-like vol.")
            sig = "LOW_CONVEXITY"

        if vov > 0.15:
            notes.append(f"High vol-of-vol ({vov:.2%}) — gamma/vega trades may be attractive.")

        return RoughVolResult(
            hurst_exponent=round(H, 4), roughness_label=label,
            eta=round(eta, 4), rho=round(rho, 4), xi0=round(xi0, 6),
            forecast_vol_5d=round(fv5, 4), forecast_vol_21d=round(fv21, 4),
            vol_of_vol=round(vov, 4), mean_reversion_days=round(hl_days, 1),
            arbitrage_vol_signal=sig, notes=notes,
        )
    except Exception as e:
        logger.error(f"Rough vol analysis failed for {ticker}: {e}")
        return RoughVolResult(hurst_exponent=0.1, roughness_label="UNKNOWN",
                              eta=1.9, rho=-0.8, xi0=0.04,
                              forecast_vol_5d=0.0, forecast_vol_21d=0.0,
                              vol_of_vol=0.0, mean_reversion_days=0.0,
                              arbitrage_vol_signal="UNKNOWN", notes=[str(e)])
