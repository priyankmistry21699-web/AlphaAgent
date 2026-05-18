"""
AlphaAgent — SABR Stochastic Volatility Model

SABR (Stochastic Alpha Beta Rho) by Hagan et al. (2002).
  dF = α F^β dW₁
  dα = ν α dW₂        Corr(dW₁,dW₂) = ρ

The Hagan approximation gives a closed-form for implied vol:
  σ_B(K,F) ≈ α·φ(ζ)/(A(F,K))·[1 + correction terms]·T^0

Includes:
  - Hagan implied vol formula
  - Vol smile fitting via calibration
  - SABR-based delta and vega
  - ATM vol and skew/convexity decomposition
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class SABRParams:
    alpha: float   # initial vol (≈ ATM vol for β=1)
    beta:  float   # CEV exponent [0,1]; 0=normal, 1=lognormal
    rho:   float   # correlation
    nu:    float   # vol of vol


@dataclass
class SABRResult:
    params:        SABRParams
    atm_vol:       float
    skew:          float        # dσ/dK at ATM (negative for equity smirk)
    convexity:     float        # d²σ/dK² at ATM
    smile_data:    List[dict]   # [{strike_pct, iv}]
    calib_rmse:    float
    signal:        str          # SKEW_HEAVY / NORMAL / FLAT
    notes:         List[str]


def sabr_iv(F: float, K: float, T: float, p: SABRParams) -> float:
    """
    Hagan et al. (2002) SABR implied vol approximation.
    Returns Black-Scholes implied vol.
    """
    alpha, beta, rho, nu = p.alpha, p.beta, p.rho, p.nu
    if T <= 0 or alpha <= 0 or nu <= 0:
        return alpha

    if abs(F - K) < 1e-8 * F:
        # ATM formula
        FK_mid = F
        term1  = ((1 - beta)**2 / 24) * alpha**2 / FK_mid**(2 - 2 * beta)
        term2  = 0.25 * rho * beta * nu * alpha / FK_mid**(1 - beta)
        term3  = (2 - 3 * rho**2) / 24 * nu**2
        iv     = (alpha / FK_mid**(1 - beta)) * (1 + (term1 + term2 + term3) * T)
    else:
        log_FK = np.log(F / K)
        FK_pow = (F * K) ** ((1 - beta) / 2)
        z      = (nu / alpha) * FK_pow * log_FK
        xi     = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        if abs(xi) < 1e-8:
            xi_ratio = 1.0
        else:
            xi_ratio = z / xi
        A  = alpha / (FK_pow * (1 + ((1-beta)**2/24)*log_FK**2
                                  + ((1-beta)**4/1920)*log_FK**4))
        B  = (1 + (((1-beta)**2/24) * alpha**2 / FK_pow**2
                   + (0.25 * rho * beta * nu * alpha / FK_pow)
                   + ((2 - 3*rho**2)/24) * nu**2) * T)
        iv = A * xi_ratio * B

    return max(1e-4, float(iv))


def _smile_grid(F: float, T: float, p: SABRParams,
                pcts: List[float] = None) -> List[dict]:
    if pcts is None:
        pcts = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.0,
                1.025, 1.05, 1.075, 1.10, 1.15, 1.20]
    return [{"strike_pct": kp, "strike": round(F*kp, 2),
             "iv": round(sabr_iv(F, F*kp, T, p), 4)} for kp in pcts]


class SABRModel:
    """SABR model with calibration and smile analytics."""

    DEFAULT_PARAMS = SABRParams(alpha=0.3, beta=0.7, rho=-0.3, nu=0.4)

    def __init__(self, F: float, T: float = 1.0):
        self.F = F        # forward price (≈ spot for near-zero rates)
        self.T = T
        self.params = self.DEFAULT_PARAMS

    def iv(self, K: float) -> float:
        return sabr_iv(self.F, K, self.T, self.params)

    def calibrate(self, market_vols: List[Tuple[float, float]]) -> float:
        """
        market_vols: [(K, sigma_mkt), ...]
        Returns RMSE.
        """
        def _obj(x):
            alpha, rho, nu = x
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                return 1e6
            p = SABRParams(alpha=alpha, beta=self.params.beta, rho=rho, nu=nu)
            return np.mean([(sabr_iv(self.F, K, self.T, p) - sig)**2
                            for K, sig in market_vols])

        x0 = [self.params.alpha, self.params.rho, self.params.nu]
        bounds = [(0.001, 3), (-0.99, 0.99), (0.001, 5)]
        res = minimize(_obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300})
        alpha, rho, nu = res.x
        self.params = SABRParams(alpha=alpha, beta=self.params.beta, rho=rho, nu=nu)
        return float(np.sqrt(res.fun))

    def atm_vol(self) -> float:
        return self.iv(self.F)

    def skew(self, dK: float = None) -> float:
        """Numerical derivative dσ/dK at ATM."""
        h = dK or self.F * 0.01
        return (self.iv(self.F + h) - self.iv(self.F - h)) / (2 * h)

    def convexity(self, dK: float = None) -> float:
        """Numerical second derivative d²σ/dK²."""
        h = dK or self.F * 0.01
        return (self.iv(self.F + h) - 2*self.iv(self.F) + self.iv(self.F - h)) / h**2

    def smile(self) -> List[dict]:
        return _smile_grid(self.F, self.T, self.params)


def analyze_sabr(ticker: str, market_data) -> SABRResult:
    """Convenience entry for the API layer."""
    notes: List[str] = []
    try:
        ohlcv = market_data.get_ohlcv("1y")
        if ohlcv.empty or len(ohlcv) < 30:
            raise ValueError("Insufficient price data")

        S  = float(ohlcv["Close"].iloc[-1])
        rv = float(ohlcv["Close"].pct_change().dropna().std() * np.sqrt(252))
        T  = 1/12   # 1-month default

        model = SABRModel(S, T)
        model.params = SABRParams(alpha=rv, beta=0.7, rho=-0.3, nu=0.5)

        # Try calibrate to options data
        calib_err = 0.0
        try:
            yf_ticker = market_data.get_yfinance_ticker()
            exps = yf_ticker.options or []
            mkt_vols = []
            for exp in exps[:2]:
                chain = yf_ticker.option_chain(exp)
                from datetime import datetime
                T_exp = max((datetime.strptime(exp, "%Y-%m-%d") - datetime.utcnow()).days / 365, 1/365)
                model.T = T_exp
                for _, row in chain.calls.iterrows():
                    if 0.75 < row.get("strike", 0) / S < 1.25 and row.get("impliedVolatility", 0) > 0.01:
                        mkt_vols.append((row["strike"], row["impliedVolatility"]))
            if len(mkt_vols) >= 4:
                calib_err = model.calibrate(mkt_vols[:20])
                notes.append(f"Calibrated SABR to {len(mkt_vols[:20])} strikes (RMSE {calib_err:.4f})")
            else:
                notes.append("Using default SABR params (insufficient options data)")
        except Exception as ex:
            notes.append(f"Options calibration skipped: {ex}")

        atm = model.atm_vol()
        sk  = model.skew()
        cvx = model.convexity()
        smile_data = model.smile()

        # Interpret skew
        if sk < -0.02:
            sig = "SKEW_HEAVY"; notes.append("Heavy put skew — market pricing tail-down risk.")
        elif sk > 0.01:
            sig = "CALL_SKEW";  notes.append("Positive skew — speculative call buying detected.")
        else:
            sig = "NORMAL";     notes.append("Normal SABR skew — balanced options flow.")

        return SABRResult(params=model.params, atm_vol=round(atm, 4),
                          skew=round(sk, 6), convexity=round(cvx, 8),
                          smile_data=smile_data, calib_rmse=round(calib_err, 6),
                          signal=sig, notes=notes)
    except Exception as e:
        logger.error(f"SABR analysis failed for {ticker}: {e}")
        return SABRResult(params=SABRModel.DEFAULT_PARAMS, atm_vol=0.0,
                          skew=0.0, convexity=0.0, smile_data=[], calib_rmse=1.0,
                          signal="UNKNOWN", notes=[str(e)])
