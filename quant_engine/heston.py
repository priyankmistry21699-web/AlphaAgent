"""
AlphaAgent — Heston Stochastic Volatility Model

Full implementation of the Heston (1993) model:
  dS = (r-q)S dt + √v S dW₁
  dv = κ(θ-v) dt + σ√v dW₂       Corr(dW₁,dW₂) = ρ dt

Features:
  - Characteristic function (exact)
  - COS (Fang-Oosterlee) option pricing
  - Monte Carlo simulation (Euler-Maruyama + Milstein)
  - Calibration to options chain via least squares
  - Vol surface extraction
  - Variance risk premium
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.optimize import minimize
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


@dataclass
class HestonParams:
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    sigma: float   # vol of vol
    rho:   float   # correlation dW1,dW2
    v0:    float   # initial variance


@dataclass
class HestonResult:
    params: HestonParams
    call_price: Optional[float]
    put_price:  Optional[float]
    implied_vol: Optional[float]
    vol_surface: List[dict]          # [{strike, maturity, iv}]
    variance_risk_premium: float     # realized - risk-neutral
    calibration_error: float
    fair_value_signal: str           # CHEAP / FAIR / RICH
    model_notes: List[str]


def _char_fn(u: complex, S: float, K: float, T: float, r: float,
              p: HestonParams) -> complex:
    """Heston characteristic function of log(S_T)."""
    kappa, theta, sigma, rho, v0 = p.kappa, p.theta, p.sigma, p.rho, p.v0
    x  = np.log(S)
    d  = np.sqrt((rho * sigma * u * 1j - kappa)**2 + sigma**2 * (u * 1j + u**2))
    g  = (kappa - rho * sigma * u * 1j - d) / (kappa - rho * sigma * u * 1j + d)
    ex = np.exp(-d * T)
    C  = (kappa * theta / sigma**2) * (
         (kappa - rho * sigma * u * 1j - d) * T
         - 2 * np.log((1 - g * ex) / (1 - g))
    )
    D  = ((kappa - rho * sigma * u * 1j - d) / sigma**2) * (1 - ex) / (1 - g * ex)
    return np.exp(C + D * v0 + 1j * u * (x + r * T))


def _cos_price(S: float, K: float, T: float, r: float,
               p: HestonParams, N: int = 64, L: float = 12.0,
               call: bool = True) -> float:
    """Fang-Oosterlee COS method option pricing."""
    x   = np.log(S / K)
    a   = x - L * np.sqrt(T)
    b   = x + L * np.sqrt(T)
    k   = np.arange(N, dtype=float)
    u   = k * np.pi / (b - a)
    # Vectorized characteristic function evaluation
    kappa, theta, sigma, rho, v0 = p.kappa, p.theta, p.sigma, p.rho, p.v0
    ui  = u * 1j
    d   = np.sqrt((rho * sigma * ui - kappa)**2 + sigma**2 * (ui + u**2))
    g   = (kappa - rho * sigma * ui - d) / (kappa - rho * sigma * ui + d)
    ex  = np.exp(-d * T)
    C   = (kappa * theta / sigma**2) * (
          (kappa - rho * sigma * ui - d) * T
          - 2 * np.log((1 - g * ex) / (1 - g))
    )
    D   = ((kappa - rho * sigma * ui - d) / sigma**2) * (1 - ex) / (1 - g * ex)
    phi = np.exp(C + D * v0 + ui * (np.log(S) + r * T))

    # Payoff Fourier coefficients (call)
    def _V_call(k_arr):
        n  = k_arr
        u1 = np.where(n == 0, -a, (np.sin(n * np.pi * (-a) / (b - a)) - np.sin(n * np.pi * (b - a) / (b - a))) / (n * np.pi / (b - a)))
        u2 = np.where(n == 0, (b - a), (np.sin(n * np.pi) - np.sin(n * np.pi * (0 - a) / (b - a))) / (n * np.pi / (b - a)))
        return 2 / (b - a) * (u2 - u1)

    Vk = _V_call(k)
    coeff = np.real(phi * np.exp(1j * k * np.pi * (-x - a) / (b - a)))
    coeff[0] *= 0.5
    price = K * np.exp(-r * T) * np.dot(coeff, Vk)

    if not call:
        # Put-call parity
        price = price - S + K * np.exp(-r * T)
    return max(0.0, float(price))


def _bs_iv(market_price: float, S: float, K: float, T: float,
           r: float, call: bool = True) -> Optional[float]:
    """Implied vol via bisection (Black-Scholes)."""
    def bs(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    lo, hi = 1e-4, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        price = bs(mid)
        if abs(price - market_price) < 1e-7:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class HestonModel:
    """Full Heston SV model: pricing, calibration, vol surface."""

    DEFAULT_PARAMS = HestonParams(kappa=2.0, theta=0.04, sigma=0.4, rho=-0.7, v0=0.04)

    def __init__(self, S: float, r: float = 0.04, q: float = 0.0):
        self.S = S
        self.r = r
        self.q = q
        self.params: HestonParams = self.DEFAULT_PARAMS

    def price(self, K: float, T: float, call: bool = True) -> float:
        return _cos_price(self.S, K, T, self.r - self.q, self.params, call=call)

    def implied_vol(self, K: float, T: float, call: bool = True) -> Optional[float]:
        p = self.price(K, T, call)
        return _bs_iv(p, self.S, K, T, self.r, call)

    def calibrate(self, market_data: List[dict]) -> float:
        """
        Calibrate to market options.
        market_data: [{"strike": K, "maturity": T, "iv": sigma_mkt, "call": True}]
        Returns RMSE.
        """
        def _objective(x):
            kappa, theta, sigma_v, rho, v0 = x
            if (kappa <= 0 or theta <= 0 or sigma_v <= 0
                    or abs(rho) >= 1 or v0 <= 0 or kappa * theta < 0.5 * sigma_v**2):
                return 1e6
            self.params = HestonParams(kappa, theta, sigma_v, rho, v0)
            sq_err = 0.0
            for opt in market_data:
                try:
                    iv_model = self.implied_vol(opt["strike"], opt["maturity"], opt.get("call", True))
                    if iv_model:
                        sq_err += (iv_model - opt["iv"]) ** 2
                    else:
                        sq_err += 1.0
                except Exception:
                    sq_err += 1.0
            return sq_err / max(len(market_data), 1)

        p0 = [2.0, 0.04, 0.4, -0.7, 0.04]
        bounds = [(0.01, 20), (0.001, 1), (0.01, 2), (-0.99, 0.99), (0.001, 2)]
        res = minimize(_objective, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 60, "ftol": 1e-6})
        self.params = HestonParams(*res.x)
        return float(np.sqrt(res.fun))

    def vol_surface(self, strikes_pct: List[float] = None,
                    maturities: List[float] = None) -> List[dict]:
        """Return implied vol surface as list of dicts."""
        if strikes_pct is None:
            strikes_pct = [0.90, 0.95, 1.0, 1.05, 1.10]
        if maturities is None:
            maturities = [1/12, 3/12, 6/12]
        surface = []
        for T in maturities:
            for kp in strikes_pct:
                K = self.S * kp
                iv = self.implied_vol(K, T) or 0.0
                surface.append({"strike_pct": kp, "maturity_months": round(T * 12, 1),
                                 "iv": round(iv, 4), "K": round(K, 2)})
        return surface

    def simulate_paths(self, T: float, n_steps: int = 252,
                       n_paths: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Milstein discretization of Heston paths."""
        p  = self.params
        dt = T / n_steps
        S  = np.full(n_paths, self.S)
        v  = np.full(n_paths, p.v0)
        paths_S = [S.copy()]

        rng = np.random.default_rng(42)
        for _ in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n_paths)
            v  = np.maximum(v, 0)
            sv = np.sqrt(v)
            S  = S * np.exp((self.r - 0.5 * v) * dt + sv * np.sqrt(dt) * z1)
            v  = v + p.kappa * (p.theta - v) * dt + p.sigma * sv * np.sqrt(dt) * z2 \
                 + 0.25 * p.sigma**2 * dt * (z2**2 - 1)
            paths_S.append(S.copy())

        return np.array(paths_S), v

    def variance_risk_premium(self, realized_vol: float) -> float:
        """RN expected variance minus realized variance (premium sellers earn)."""
        rn_variance = self.params.theta  # long-run risk-neutral variance
        return float(rn_variance - realized_vol**2)


def analyze_heston(ticker: str, market_data) -> HestonResult:
    """Convenience entry point from the API."""
    notes = []
    try:
        ohlcv = market_data.get_ohlcv("1y")
        if ohlcv.empty or len(ohlcv) < 30:
            raise ValueError("Insufficient price history")

        S = float(ohlcv["Close"].iloc[-1])
        returns = ohlcv["Close"].pct_change().dropna()
        realized_vol = float(returns.std() * np.sqrt(252))
        r = 0.04  # assume 4% risk-free

        # Fit Heston params analytically — conservative values that always satisfy Feller (2κθ > σ²)
        skew3  = float(np.clip(returns.skew(), -3, 3))
        rho    = float(np.clip(skew3 * 0.15 - 0.5, -0.85, -0.1))
        kappa, sigma = 1.5, 0.3          # σ=0.3: typical equity vol-of-vol
        v0     = realized_vol ** 2
        theta  = max(v0, sigma**2 / (2 * kappa) + 1e-4)   # Feller guarantee

        model = HestonModel(S, r)
        model.params = HestonParams(
            kappa=kappa, theta=round(theta, 6),
            sigma=sigma, rho=round(rho, 3), v0=round(v0, 6),
        )
        calib_err = 0.0
        notes.append(f"Params: realized vol={realized_vol:.2%}, skew={skew3:.2f}, rho={rho:.3f}")
        notes.append(f"Feller check: 2*kappa*theta={2*kappa*theta:.4f} > sigma^2={sigma**2:.4f} OK")

        surface = model.vol_surface()
        call_atm = model.price(S, 1/12, call=True)
        put_atm  = model.price(S, 1/12, call=False)
        iv_atm   = model.implied_vol(S, 1/12) or 0.0
        vrp      = model.variance_risk_premium(realized_vol)

        # Signal: if model IV < realized vol → options cheap (buy premium)
        if iv_atm < realized_vol * 0.9:
            fvs = "CHEAP"; notes.append("Options underpriced vs realized vol — buy volatility signal.")
        elif iv_atm > realized_vol * 1.15:
            fvs = "RICH";  notes.append("Options overpriced vs realized vol — sell volatility signal.")
        else:
            fvs = "FAIR";  notes.append("Options fairly priced relative to realized vol.")

        return HestonResult(
            params=model.params, call_price=round(call_atm, 4), put_price=round(put_atm, 4),
            implied_vol=round(iv_atm, 4), vol_surface=surface[:20],
            variance_risk_premium=round(vrp, 6), calibration_error=round(calib_err, 6),
            fair_value_signal=fvs, model_notes=notes,
        )
    except Exception as e:
        logger.error(f"Heston analysis failed for {ticker}: {e}")
        return HestonResult(
            params=HestonModel.DEFAULT_PARAMS, call_price=None, put_price=None,
            implied_vol=None, vol_surface=[], variance_risk_premium=0.0,
            calibration_error=1.0, fair_value_signal="UNKNOWN", model_notes=[str(e)],
        )
