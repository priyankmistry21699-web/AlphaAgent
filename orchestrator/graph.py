"""
AlphaAgent — LangGraph Orchestrator

The main workflow engine. Uses LangGraph to create a strict, typed,
and observable execution graph.

Workflow:
  1. START -> fetch market data
  2. run_agents -> all agents execute in parallel via ThreadPoolExecutor
  3. portfolio_manager -> Bayesian fusion, override checks, SignalPacket
  4. END
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

import pandas as pd

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from data.market import MarketData
from agents.registry import AgentRegistry
from agents.state import AgentResult, SignalPacket, Direction, Confidence, RegimeResult, Regime, HoldingPeriod
from quant_engine.bayesian import BayesianFusion

logger = logging.getLogger(__name__)


# ─── 1. Graph State Definition ──────────────────────────────────────────

class AlphaGraphState(BaseModel):
    """The central state passed between nodes in the LangGraph."""
    ticker: str

    # Injected data/services
    market_data: Any = None
    registry: Any = None

    # Agent Outputs
    agent_results: Dict[str, AgentResult] = Field(default_factory=dict)

    # Final Output
    final_signal: Any = None

    class Config:
        arbitrary_types_allowed = True


# ─── 2. Node Functions ──────────────────────────────────────────────────

def data_ingestion_node(state: AlphaGraphState) -> AlphaGraphState:
    """Node 1: Pre-warms the data cache in parallel to prevent redundant API calls."""
    logger.info(f"[{state.ticker}] Node: Data Ingestion (parallel)")
    md = state.market_data

    def _fetch_ohlcv():
        try: md.get_ohlcv("1y")
        except Exception: pass

    def _fetch_financials():
        try: md.get_financials()
        except Exception: pass

    def _fetch_info():
        try: md.get_info()
        except Exception: pass

    def _fetch_news():
        try: md.get_yfinance_ticker().news
        except Exception: pass

    def _fetch_options():
        try:
            t = md.get_yfinance_ticker()
            if t.options:
                t.option_chain(t.options[0])   # pre-warm nearest expiry
        except Exception: pass

    def _fetch_returns():
        try: md.get_returns("1y")
        except Exception: pass

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [
            pool.submit(_fetch_ohlcv),
            pool.submit(_fetch_financials),
            pool.submit(_fetch_info),
            pool.submit(_fetch_news),
            pool.submit(_fetch_options),
            pool.submit(_fetch_returns),
        ]
        for f in futs:
            try: f.result(timeout=20)
            except Exception: pass

    return state


def run_agents_node(state: AlphaGraphState) -> AlphaGraphState:
    """Node 2: Runs all registered agents in parallel via ThreadPoolExecutor."""
    logger.info(f"[{state.ticker}] Node: Running Agents (parallel)")

    active_agents = state.registry.get_active_agents()
    results: Dict[str, AgentResult] = {}

    def _run_one(name: str, agent) -> tuple:
        try:
            return name, agent.analyze(state.ticker, state.market_data)
        except Exception as e:
            logger.error(f"[{state.ticker}] Agent '{name}' raised uncaught exception: {e}", exc_info=True)
            from agents.state import AgentResult, Direction
            return name, AgentResult(
                agent_name=name,
                vote=Direction.HOLD,
                probability_up=0.5,
                confidence=0.0,
                reasoning=f"AGENT FAILURE: {e}",
                warnings=[f"Agent '{name}' failed: {e}"],
            )

    AGENT_TIMEOUT = 35   # seconds per agent before neutral fallback (relaxed)
    OUTER_TIMEOUT = AGENT_TIMEOUT + 10   # 45s outer cap on the whole batch

    from agents.state import AgentResult, Direction
    pool = ThreadPoolExecutor(max_workers=min(len(active_agents), 9))
    futures = {pool.submit(_run_one, n, a): n for n, a in active_agents.items()}

    try:
        for future in as_completed(futures, timeout=OUTER_TIMEOUT):
            try:
                name, res = future.result(timeout=AGENT_TIMEOUT)
            except Exception as e:
                name = futures[future]
                logger.warning(f"[{state.ticker}] Agent '{name}' timed out or errored: {e}")
                res = AgentResult(
                    agent_name=name, vote=Direction.HOLD,
                    probability_up=0.5, confidence=0.0,
                    reasoning=f"TIMEOUT: {e}", warnings=[f"Agent '{name}' timed out"],
                )
            results[name] = res
            logger.info(f"[{state.ticker}] Agent '{name}' finished — P(up)={res.probability_up:.3f}")
    except TimeoutError as _toe:
        # The outer as_completed timed out — some futures still pending. Mark
        # all uncompleted agents as neutral instead of crashing the request.
        logger.warning(
            f"[{state.ticker}] Outer agent batch timeout ({OUTER_TIMEOUT}s) — "
            f"marking {len(futures) - len(results)} unfinished agents as neutral"
        )
        for fut, name in futures.items():
            if name in results:
                continue
            if fut.done():
                try:
                    n, r = fut.result(timeout=0.1)
                    results[n] = r
                    logger.info(f"[{state.ticker}] Late-completed agent '{n}' captured")
                    continue
                except Exception:
                    pass
            fut.cancel()
            results[name] = AgentResult(
                agent_name=name, vote=Direction.HOLD,
                probability_up=0.5, confidence=0.0,
                reasoning="BATCH_TIMEOUT: agent did not complete within budget",
                warnings=[f"Agent '{name}' did not complete (batch timeout)"],
            )
    finally:
        # Don't wait for stragglers — let them die in the background.
        pool.shutdown(wait=False, cancel_futures=True)

    state.agent_results = results
    return state


_PRIOR_CACHE: dict = {}
_PRIOR_TTL = 3600   # recompute prior at most once per hour


def _dynamic_prior(ticker: str) -> float:
    """
    Compute a market-regime-aware Bayesian prior instead of a flat 50%.

    Logic:
      - If SPY is above its 50-day SMA  → slight bullish bias  (0.53)
      - If SPY is below its 50-day SMA  → slight bearish bias  (0.47)
      - For non-US assets use the ticker itself vs its own 50d SMA
      - Falls back to 0.50 on any error
    """
    import time as _t
    import yfinance as _yf
    import numpy as _np

    cache_key = f"{ticker}_prior"
    entry = _PRIOR_CACHE.get(cache_key)
    if entry and _t.time() < entry[1]:
        return entry[0]

    try:
        ref = ticker if not ticker.startswith("^") else "SPY"
        df = _yf.download(ref, period="3mo", interval="1d",
                          auto_adjust=True)
        closes = df["Close"].squeeze().dropna()
        if len(closes) >= 50:
            sma50 = float(closes.iloc[-50:].mean())
            price = float(closes.iloc[-1])
            if price > sma50 * 1.005:       # clearly above SMA50
                prior = 0.53
            elif price < sma50 * 0.995:     # clearly below SMA50
                prior = 0.47
            else:
                prior = 0.50                # near SMA50 — neutral
        else:
            prior = 0.50
    except Exception:
        prior = 0.50

    _PRIOR_CACHE[cache_key] = (prior, _t.time() + _PRIOR_TTL)
    return prior


def portfolio_manager_node(state: AlphaGraphState) -> AlphaGraphState:
    """
    Node 3: Aggregates agent results and makes the final trading decision.
    """
    logger.info(f"[{state.ticker}] Node: Portfolio Manager (Aggregator)")

    results = state.agent_results

    # ── 1. Risk Circuit Breakers ─────────────────────────────────────────
    is_crisis = False
    is_halted = False
    multiplier = 1.0
    override_reason = ""

    risk_res = results.get("risk")
    if risk_res:
        risk_text = risk_res.reasoning
        if any(kw in risk_text for kw in ("BLACK SWAN", "FLASH CRASH")):
            is_halted = True
            is_crisis = True
            multiplier = 0.0
            override_reason = next(
                (kw for kw in ("BLACK SWAN", "FLASH CRASH") if kw in risk_text),
                "HALT"
            )
        elif "CRITICAL RISK" in risk_text:
            is_crisis = True
            is_halted = False          # signal allowed through, just scaled down
            multiplier = 0.25
            override_reason = "CRITICAL RISK: size 25%"
        elif "HIGH RISK" in risk_text:
            is_crisis = True
            multiplier = 0.5
            override_reason = "HIGH RISK: reduced size"
        elif any(kw in risk_text for kw in ("GEO_SHOCK", "WAR/GEO SHOCK")):
            is_crisis = True
            multiplier = min(multiplier, 0.35)
            override_reason = "GEO SHOCK: capped at 35%"
        elif any(kw in risk_text for kw in ("CARRY_UNWIND", "CARRY TRADE UNWIND")):
            is_crisis = True
            multiplier = min(multiplier, 0.5)
            override_reason = "CARRY UNWIND: reduced to 50%"

    # Geopolitical hard-cap
    geo_res = results.get("geopolitical")
    if geo_res and any("GEOPOLITICAL OVERRIDE" in w for w in geo_res.warnings):
        multiplier = min(multiplier, 0.35)
        is_crisis = True
        if not override_reason:
            override_reason = "GEOPOLITICAL OVERRIDE: capped at 35%"

    # ── 2. Market Regime Detection ───────────────────────────────────────
    # Determines regime-conditional correlation penalties and agent weights.
    # Falls back to static defaults if detection fails.
    _STATIC_CORR = {
        "sentiment":    0.20,
        "volatility":   0.40,
        "geopolitical": 0.25,
        "currency":     0.20,
    }
    CORRELATION_MAP = dict(_STATIC_CORR)
    REGIME_WEIGHTS  = {}
    market_regime   = None

    try:
        from quant_engine.regime_weights import (
            detect_regime, get_regime_correlations, get_regime_weights,
            MarketRegime,
        )
        market_regime   = detect_regime()
        CORRELATION_MAP = get_regime_correlations(market_regime)
        REGIME_WEIGHTS  = get_regime_weights(market_regime)
        logger.info(f"[{state.ticker}] Regime: {market_regime.value}")

        # ── Soft Regime Blending (continuous instead of discrete) ──────────────
        # Blend regime weights based on HMM probabilities for smoother transitions
        try:
            from quant_engine.hmm import RegimeDetector
            import yfinance as _yf_blend
            _spy_blend = _yf_blend.download("SPY", period="1y", interval="1d",
                                            auto_adjust=True, progress=False)
            if not _spy_blend.empty:
                _spy_r = _spy_blend["Close"].squeeze().pct_change().dropna()
                _hmm_b = RegimeDetector(n_states=3).fit_predict(_spy_r)
                if _hmm_b is not None:
                    _p_bull = _hmm_b.probabilities.get("bull", 0.5)
                    _p_bear = _hmm_b.probabilities.get("bear", 0.3)
                    _p_crisis = _hmm_b.probabilities.get("crisis", 0.2)
                    # Map HMM probabilities to regime weight tables
                    _w_bull = get_regime_weights(MarketRegime.BULL_TREND)
                    _w_chop = get_regime_weights(MarketRegime.BULL_CHOPPY)
                    _w_bear = get_regime_weights(MarketRegime.BEAR)
                    _w_cris = get_regime_weights(MarketRegime.CRISIS)
                    # Soft-blend: high bull_prob → bull weights dominate
                    _blended = {}
                    for _ag in _w_bull.keys():
                        _blended[_ag] = (_p_bull * _w_bull.get(_ag, 0) +
                                         max(0, _p_bull - 0.3) * _w_chop.get(_ag, 0) +
                                         _p_bear * _w_bear.get(_ag, 0) +
                                         _p_crisis * _w_cris.get(_ag, 0))
                    _tot = sum(_blended.values())
                    if _tot > 0:
                        REGIME_WEIGHTS = {k: v / _tot for k, v in _blended.items()}
                        logger.info(f"[{state.ticker}] Soft-blended weights: bull={_p_bull:.2f} bear={_p_bear:.2f} crisis={_p_crisis:.2f}")
        except Exception as _blend_e:
            logger.debug(f"[{state.ticker}] Soft blending fallback: {_blend_e}")
    except Exception as _re:
        logger.warning(f"[{state.ticker}] Regime detection failed: {_re}")

    _N_VOTING_AGENTS = max(1, len([n for n in results if n != "risk"]))
    _AVG_WEIGHT      = 1.0 / _N_VOTING_AGENTS

    # ── 3. Bayesian Fusion — dynamic prior from SPY vs 50-day SMA ───────
    _prior = _dynamic_prior(state.ticker)
    from config.settings_manager import settings as _cfg
    _sensitivity = _cfg.get("agent_defaults.bayesian_sensitivity", 1.4)
    fusion = BayesianFusion(prior=_prior, sensitivity=_sensitivity)

    for agent_name, res in results.items():
        if agent_name == "risk":
            continue  # risk is circuit breaker, not a Bayesian voter
        corr = CORRELATION_MAP.get(agent_name, 0.0)
        # Scale confidence by regime weight; cap at 3x to avoid extremes
        regime_w     = REGIME_WEIGHTS.get(agent_name, _AVG_WEIGHT)
        regime_scale = max(0.3, min(3.0, regime_w / _AVG_WEIGHT)) if _AVG_WEIGHT > 0 else 1.0
        fusion.update(
            agent_prob=res.probability_up,
            confidence=res.confidence * regime_scale,
            correlation=corr,
            agent_name=agent_name,
        )

    final_prob = fusion.posterior

    # ── 4. Direction & Conviction ────────────────────────────────────────
    if is_halted:
        final_prob = 0.5
        direction  = Direction.HOLD
        conviction = 0.0
    else:
        conviction = abs(final_prob - 0.5) * 2.0   # 0 = neutral, 1 = max

        # Entropy-adaptive gate: widen thresholds when agents disagree (high entropy),
        # tighten when they strongly agree (low entropy = act on smaller edge)
        _entropy = fusion.entropy
        if _entropy > 0.85:
            _long_gate, _short_gate = 0.56, 0.44   # high disagreement → conservative
        elif _entropy > 0.70:
            _long_gate, _short_gate = 0.545, 0.455
        elif _entropy < 0.40:
            _long_gate, _short_gate = 0.515, 0.485  # strong consensus → act on small edge
        else:
            _long_gate, _short_gate = 0.53, 0.47    # default

        if final_prob > _long_gate:
            direction = Direction.LONG
        elif final_prob < _short_gate:
            direction = Direction.SHORT
        else:
            direction = Direction.HOLD

    # ── 4.5. Meta-Learner Stacking ───────────────────────────────────────
    # Blends LightGBM prediction with Bayesian posterior using learned weights.
    # Falls back silently to pure Bayesian if model not yet trained.
    try:
        from quant_engine.meta_learner import get_meta_learner
        _ml     = get_meta_learner()
        _a_prbs = {n: r.probability_up for n, r in results.items()}
        _ml_res = _ml.predict(
            agent_probs    = _a_prbs,
            bayesian_prob  = final_prob,
            conviction     = conviction,
            entropy        = fusion.entropy,
            agreement_score= fusion.agreement_score(),
        )
        if _ml_res is not None:
            logger.info(
                f"[{state.ticker}] MetaLearner: lgbm={_ml_res.lgbm_prob:.3f} "
                f"w={_ml_res.blend_weight:.2f} → blended={_ml_res.blended_prob:.3f}"
            )
            final_prob = _ml_res.blended_prob
            # Recompute conviction and direction after meta-learner blend
            conviction = abs(final_prob - 0.5) * 2.0
            if not is_halted:
                if final_prob > _long_gate:
                    direction = Direction.LONG
                elif final_prob < _short_gate:
                    direction = Direction.SHORT
                else:
                    direction = Direction.HOLD
    except Exception as _mle:
        logger.warning(f"[{state.ticker}] Meta-learner failed: {_mle}")

    # ── 5. HMM Regime Detection ──────────────────────────────────────────
    regime_result_obj = None
    try:
        from quant_engine.hmm import RegimeDetector
        returns = state.market_data.get_returns("1y")
        if len(returns) > 100:
            detector = RegimeDetector(n_states=3)
            hmm_res = detector.fit_predict(returns)
            regime_result_obj = RegimeResult(
                current_regime=Regime(hmm_res.current_regime),
                regime_probabilities=hmm_res.probabilities,
                transition_risk=hmm_res.transition_risk,
                regime_duration_days=hmm_res.regime_duration_days,
            )
    except Exception as e:
        logger.warning(f"HMM regime detection failed: {e}")

    # ── 6. Signal Decay / Holding Period ────────────────────────────────
    holding_period_obj = None
    try:
        from quant_engine.signal_decay import SignalDecay
        prices = state.market_data.get_ohlcv("1y")["Close"]
        if len(prices) > 30:
            decay = SignalDecay(prices)
            decay_res = decay.calculate_half_life()
            holding_period_obj = HoldingPeriod(
                half_life_days=decay_res.half_life_days,
                optimal_hold_min=decay_res.optimal_hold_min,
                optimal_hold_max=decay_res.optimal_hold_max,
                signal_strength=float(min(1.0, conviction)),
            )
    except Exception as e:
        logger.warning(f"Signal decay calculation failed: {e}")

    # ── 7. Compile the Final Signal Packet ──────────────────────────────
    signal = SignalPacket(
        ticker=state.ticker,
        direction=direction,
        conviction_pct=conviction * 100,
        confidence=(
            Confidence.HIGH   if conviction > 0.4 else
            Confidence.MEDIUM if conviction > 0.2 else
            Confidence.LOW
        ),
        agent_results=list(results.values()),
        regime=regime_result_obj,
        holding_period=holding_period_obj,
        risk_metrics=None,
        override_active=bool(override_reason),
        override_reason=override_reason,
    )

    # Append only the structured warning strings from risk (not the full reasoning blob)
    if risk_res and risk_res.warnings:
        signal.warnings.extend(risk_res.warnings)

    if fusion.entropy > 0.9:
        signal.warnings.append(
            f"High System Entropy ({fusion.entropy:.2f}): Agents strongly disagree."
        )

    # ── 8. Portfolio-level risk metrics (per-signal portfolio_risk view) ────
    portfolio_risk_obj = None
    try:
        from quant_engine.portfolio_risk import PortfolioRisk
        import yfinance as _yf_pr
        # Single-position "portfolio" of ticker + SPY hedge proxy for marginal risk
        _peers = [state.ticker, "SPY"]
        _hist = _yf_pr.download(_peers, period="6mo", interval="1d",
                                auto_adjust=True, progress=False, group_by="ticker")
        if not _hist.empty:
            _rets = {}
            for _p in _peers:
                try:
                    _c = _hist[_p]["Close"].dropna() if len(_peers) > 1 else _hist["Close"].dropna()
                    _rets[_p] = _c.pct_change().dropna()
                except Exception:
                    continue
            if len(_rets) == 2:
                _df_pr = pd.DataFrame(_rets).dropna()
                if len(_df_pr) >= 30:
                    _pr = PortfolioRisk().analyze(
                        weights={state.ticker: 1.0, "SPY": 0.0},
                        returns=_df_pr,
                    )
                    if _pr is not None:
                        portfolio_risk_obj = {
                            "var_95": _pr.var_95,
                            "var_99": _pr.var_99,
                            "cvar_95": _pr.cvar_95,
                            "portfolio_vol_ann": _pr.portfolio_vol_ann,
                            "stress_2008": _pr.stress_scenarios.get("GFC_2008"),
                            "stress_covid": _pr.stress_scenarios.get("COVID_Mar_2020"),
                        }
    except Exception as _pre:
        logger.debug(f"[{state.ticker}] Portfolio risk computation skipped: {_pre}")

    state.final_signal = {
        "packet":          signal,
        "probability_up":  final_prob,
        "multiplier":      multiplier,
        "entropy":         fusion.entropy,
        "risk_res":        risk_res,
        "council":         fusion.council_summary(),
        "agreement_score": fusion.agreement_score(),
        "market_regime":   market_regime.value if market_regime else "UNKNOWN",
        "portfolio_risk":  portfolio_risk_obj,
    }
    return state


# ─── 3. Graph Builder ───────────────────────────────────────────────────

def build_alpha_graph() -> Any:
    """Builds and compiles the LangGraph."""
    workflow = StateGraph(AlphaGraphState)

    workflow.add_node("data_ingestion", data_ingestion_node)
    workflow.add_node("run_agents", run_agents_node)
    workflow.add_node("portfolio_manager", portfolio_manager_node)

    workflow.add_edge(START, "data_ingestion")
    workflow.add_edge("data_ingestion", "run_agents")
    workflow.add_edge("run_agents", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    return workflow.compile()
