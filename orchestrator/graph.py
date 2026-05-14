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
    """Node 1: Pre-warms the data cache to prevent redundant API calls."""
    logger.info(f"[{state.ticker}] Node: Data Ingestion")
    md = state.market_data
    md.get_ohlcv("1y")
    md.get_financials()
    md.get_info()
    try:
        md.get_yfinance_ticker().news
    except Exception:
        pass
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

    with ThreadPoolExecutor(max_workers=min(len(active_agents), 9)) as pool:
        futures = {pool.submit(_run_one, n, a): n for n, a in active_agents.items()}
        for future in as_completed(futures):
            name, res = future.result()
            results[name] = res
            logger.info(f"[{state.ticker}] Agent '{name}' finished — P(up)={res.probability_up:.3f}")

    state.agent_results = results
    return state


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
        if any(kw in risk_text for kw in ("CRITICAL RISK", "BLACK SWAN", "FLASH CRASH")):
            is_halted = True
            is_crisis = True
            multiplier = 0.0
            override_reason = next(
                (kw for kw in ("BLACK SWAN", "FLASH CRASH", "CRITICAL RISK") if kw in risk_text),
                "HALT"
            )
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

    # ── 2. Bayesian Fusion ───────────────────────────────────────────────
    fusion = BayesianFusion(prior=0.5)

    # Correlation penalties — prevents double-counting overlapping signals
    CORRELATION_MAP = {
        "sentiment":    0.20,   # correlated with news/narrative
        "volatility":   0.40,   # highly correlated with risk agent
        "geopolitical": 0.25,   # correlated with macro
        "currency":     0.20,   # correlated with macro
    }

    for agent_name, res in results.items():
        if agent_name == "risk":
            continue  # risk is circuit breaker, not a Bayesian voter
        corr = CORRELATION_MAP.get(agent_name, 0.0)
        fusion.update(
            agent_prob=res.probability_up,
            confidence=res.confidence,
            correlation=corr,
        )

    final_prob = fusion.posterior

    # ── 3. Direction & Conviction ────────────────────────────────────────
    if is_halted:
        final_prob = 0.5          # neutral — no directional bet when halted
        direction = Direction.HOLD
        conviction = 0.0
    else:
        conviction = abs(final_prob - 0.5) * 2.0   # 0 = neutral, 1 = max
        if final_prob > 0.55:
            direction = Direction.LONG
        elif final_prob < 0.45:
            direction = Direction.SHORT
        else:
            direction = Direction.HOLD

    # ── 4. HMM Regime Detection ──────────────────────────────────────────
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

    # ── 5. Signal Decay / Holding Period ────────────────────────────────
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

    # ── 6. Compile the Final Signal Packet ──────────────────────────────
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

    state.final_signal = {
        "packet": signal,
        "probability_up": final_prob,
        "multiplier": multiplier,
        "entropy": fusion.entropy,
        "risk_res": risk_res,
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
