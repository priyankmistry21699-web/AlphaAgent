# AlphaAgent — Detailed Function Tree (simple wording)

NOTE: I scanned the repository and produced a simplified, beginner-friendly function tree. It lists files, classes, and functions with: purpose, inputs (parameters and data sources), outputs, methods, and short description of calculations. Some files were inferred from imports; if you want more detail for any file I will open it and expand.

Repository: priyankmistry21699-web/AlphaAgent

---

## How to read this file
- File: path relative to repo root
- Node: function or class name
- Purpose: one-line description
- Inputs: parameters the function takes and external data sources it reads
- Outputs: what it returns or emits
- Methods: for classes, list of important methods
- Notes: simple extra notes for beginners

---

# 1) cli.py
- Entry point: run_analysis(ticker, output_json=False, agent_filter=None)
  - Purpose: run full analysis for a ticker and print results
  - Inputs: ticker (string), flags from CLI; it creates MarketData and AgentRegistry and builds the orchestrator graph
  - Outputs: prints final signal or JSON; returns nothing
  - Notes: uses rich for nice console output; handles errors and shows spinner

# 2) data/market.py — MarketData
Class: MarketData
- Purpose: central place to get market data (prices, info, financials, options, holders)
- Main attributes:
  - ticker (str)
  - cache (DataCache) — persistent cache
  - _yf_ticker (yfinance.Ticker)
  - _ohlcv_cache (in-memory cache)
- Important properties (read TTL values from settings): OHLCV_TTL, INFO_TTL, FINANCIALS_TTL, OPTIONS_TTL, INSIDER_TTL, HOLDERS_TTL

- Methods:
  - get_ohlcv(period='1y', interval='1d') -> pd.DataFrame
    - Inputs: period, interval
    - Data sources: DataCache (persistent), yfinance.history
    - Outputs: validated OHLCV DataFrame (Open, High, Low, Close, Volume)
    - Calculation: loads cached data if available; otherwise fetches from yfinance and validates via validate_ohlcv
  - get_returns(period='1y', method='log') -> pd.Series
    - Inputs: period, method
    - Data sources: get_ohlcv
    - Outputs: return series (log or simple)
  - get_current_price() -> float
    - Returns last Close price or 0.0
  - get_info() -> dict
    - Inputs: none
    - Data sources: cache or yfinance.Ticker.info
    - Outputs: company metadata (marketCap, sector, trailingPE, etc.)
  - get_financials() -> dict
    - Returns quarterly income, balance, cashflow as DataFrames
  - get_options_chain(expiry_index=0) -> Optional[dict]
    - Returns calls, puts DataFrames and expiry
  - get_insider_transactions(), get_institutional_holders(), get_major_holders()
    - Return DataFrames from yfinance attributes
  - has_sufficient_history(min_days=200) -> bool
    - Uses ensure_minimum_history (data.validation)
  - summary() -> dict
    - Simple summary: current_price, company_name, sector, market_cap, pe_ratio

- Notes: This class centralizes network calls and caching to avoid repeated API hits.

# 3) data/cache.py (DataCache) — (concept)
- Purpose: persist and retrieve cached API responses (likely SQLite)
- Main methods:
  - get(namespace, key, subkey) -> cached object or None
  - set(namespace, key, subkey, value, ttl_seconds)
- Inputs: namespace and keys used by MarketData
- Outputs: Python structures stored in cache
- Notes: TTL controls freshness; MarketData uses DataCache extensively

# 4) data/validation.py (helpers)
- Purpose: clean and validate data coming from yfinance
- Important functions used by MarketData:
  - validate_ohlcv(df, ticker) -> pd.DataFrame (clean, ensure correct columns)
  - validate_financials(info, ticker) -> dict
  - compute_returns(ohlcv, method) -> pd.Series
  - ensure_minimum_history(ohlcv, min_days, ticker) -> bool
- Notes: validation prevents downstream errors when data is missing or malformed

# 5) config/settings_manager.py — SettingsManager
Class: SettingsManager (singleton)
- Purpose: load config/settings.yaml and provide get() and get_section()
- Methods:
  - get(key, default=None) -> value
  - get_section(section) -> dict
  - reload(), save()
- Inputs: disk file config/settings.yaml
- Outputs: config values used across the app
- Notes: thread-safe with locks; agents and MarketData call settings.get

# 6) agents/base.py — BaseAgent
Class: BaseAgent (abstract)
- Purpose: standard interface and wrapper for all agents
- Important methods:
  - analyze(ticker, data, **kwargs) -> AgentResult
    - Wraps _run_analysis with timing and exception handling
  - _run_analysis(...) -> AgentResult (abstract — implemented by agents)
  - _build_error_result(reason) -> AgentResult (returns neutral 50% with 0 confidence)
- Notes: ensures agents return consistent AgentResult and do not crash the whole pipeline

# 7) agents/state.py — data models
- Purpose: pydantic models for AgentResult, VolatilityResult, MonteCarloResult, FactorScore, and enums (Direction, Confidence, Regime)
- Main model: AgentResult
  - Fields: agent_name, vote (Direction), probability_up (0-1), confidence (0-1), reasoning, factor_scores (dict of FactorScore), warnings, computation_time_ms
- Notes: Agents return AgentResult objects; orchestrator uses them to fuse signals

# 8) agents/registry.py — AgentRegistry
Class: AgentRegistry
- Purpose: create and hold instances of enabled agents
- Behavior:
  - Reads config/agents.yaml
  - register(agent_instance) — adds agent if enabled
  - get_active_agents() -> dict[name -> agent]
  - get_agent(name) -> agent
  - get_agent_weight(name, regime='normal') -> float (reads weights from config and allows regime overrides)
- Notes: CLI and API create this registry and orchestrator reads it to run agents

# 9) orchestrator/graph.py — AlphaGraph (core)
Main components:
- AlphaGraphState (Pydantic model)
  - Fields: ticker, market_data, registry, agent_results, final_signal

- Nodes (functions executed in sequence by LangGraph):
  - data_ingestion_node(state)
    - Purpose: pre-warm MarketData (get_ohlcv, get_financials, get_info) to avoid repeated fetches
    - Inputs: state.market_data
    - Outputs: same state
  - run_agents_node(state)
    - Purpose: run all agents in parallel using ThreadPoolExecutor
    - Inputs: state.registry.get_active_agents(), state.ticker, state.market_data
    - Outputs: state.agent_results (dict of AgentResult)
  - portfolio_manager_node(state)
    - Purpose: fuse agent results into a final SignalPacket
    - Important steps:
      - risk agent is checked for overrides (halt trading)
      - BayesianFusion used to combine probabilities of voting agents
      - HMM regime detection via quant_engine.hmm
      - Signal decay, holding period, and final conviction calculation
    - Outputs: state.final_signal (contains packet, probability_up, entropy, multiplier)

- Usage: build_alpha_graph() constructs the workflow; graph.invoke(state) runs it and returns final state

- Notes: This is the "brain" that decides how agent votes are combined

# 10) Agents (who they are and what they do)
For each agent: simple bullets — inputs, main calculations, outputs, callers

- agents/technical.py — TechnicalAgent
  - Purpose: compute technical indicators and momentum
  - Inputs: MarketData.get_ohlcv (prices)
  - Uses: quant_engine.technical.compute_indicators, quant_engine.momentum.MomentumEngine, quant_engine.hmm.RegimeDetector, compute_seasonality_score
  - Outputs: AgentResult with factor_scores (rsi, macd, bollinger, etc.), probability_up and confidence
  - Callers: orchestrator (run_agents_node), tests, CLI

- agents/fundamental.py — FundamentalAgent (expected)
  - Purpose: analyze financial statements and valuation
  - Inputs: MarketData.get_financials, MarketData.get_info
  - Uses: quant_engine.scoring.compute_fundamental_scores
  - Outputs: AgentResult with valuation and health scores
  - Callers: orchestrator, tests, CLI

- agents/sentiment.py — SentimentAgent
  - Purpose: sentiment from news and LLM (or simulated)
  - Inputs: news (MarketData or external news), embeddings, LLM API key (optional)
  - Outputs: AgentResult (probability / reasoning)
  - Callers: orchestrator, tests

- agents/macro.py — MacroAgent
  - Purpose: macro environment signals (recession, yield curve, VIX)
  - Inputs: macro snapshots (FRED or cached data)
  - Uses: quant_engine.macro.analyze_macro_environment
  - Outputs: AgentResult
  - Callers: orchestrator

- agents/insider.py — InsiderAgent
  - Purpose: insider buying/selling and institutional ownership
  - Inputs: MarketData.get_insider_transactions, get_major_holders
  - Uses: quant_engine.insider.analyze_insider_data
  - Outputs: AgentResult (insider sentiment score, warnings)
  - Callers: orchestrator, tests

- agents/volatility.py — VolatilityAgent
  - Purpose: forecast volatility regime and volatility-driven signals
  - Inputs: MarketData.get_returns, options chain (MarketData.get_options_chain)
  - Uses: quant_engine.garch.GARCHModel, quant_engine.options_intel.analyze_options, quant_engine.kalman.KalmanBeta
  - Outputs: AgentResult with vol metrics and vote adjustments
  - Callers: orchestrator

- agents/risk.py — RiskAgent
  - Purpose: circuit breaker and position sizing
  - Inputs: prices, returns from MarketData
  - Uses: quant_engine.garch.GARCHModel, quant_engine.monte_carlo.MonteCarloEngine, quant_engine.evt.ExtremeValueModel, quant_engine.kelly.KellyCriterion
  - Outputs: AgentResult used as override (not fused like others) and risk metrics
  - Callers: orchestrator (portfolio_manager_node treats risk specially)

- agents/geopolitical.py — GeopoliticalAgent
  - Purpose: geopolitical event risk (news-driven)
  - Inputs: news feeds, external sources
  - Outputs: AgentResult (risk-adjusted probability)
  - Callers: orchestrator

- agents/currency.py — CurrencyAgent
  - Purpose: FX moves' impact on a US stock
  - Inputs: FX snapshot (data.currency.CurrencyData), MarketData.get_info (sector)
  - Outputs: AgentResult adjusting probability based on USD strength
  - Callers: orchestrator

# 11) quant_engine (math kernels)
- quant_engine/garch.py — GARCHModel
  - Purpose: forecast future volatility and classify vol regime
  - Inputs: returns series
  - Outputs: GARCHResult (vol_1day, vol_5day, vol_regime, params)
  - Callers: VolatilityAgent, RiskAgent, backtest

- quant_engine/hmm.py — RegimeDetector
  - Purpose: fit HMM to returns and label regime (BULL/BEAR/CRISIS)
  - Inputs: returns series
  - Outputs: RegimeResult (current_regime, probabilities)
  - Callers: TechnicalAgent, orchestrator

- quant_engine/momentum.py — MomentumEngine
  - Purpose: compute Hurst exponent and 12M-1M momentum
  - Inputs: price series
  - Outputs: MomentumResult (hurst_exponent, regime_type, momentum_12m_1m)
  - Callers: TechnicalAgent

- quant_engine/evt.py — ExtremeValueModel
  - Purpose: model tails (VaR/CVaR using GPD/GEV)
  - Inputs: returns series
  - Outputs: EVTResult (var_95, var_99, cvar_95, cvar_99, tail_index)
  - Callers: RiskAgent

- quant_engine/kelly.py — KellyCriterion
  - Purpose: compute position sizing from probability and win/loss
  - Inputs: prob_win, expected_win_pct, expected_loss_pct
  - Outputs: KellyResult (full_kelly, half_kelly, vol_adjusted_kelly)
  - Callers: RiskAgent

- quant_engine/bayesian.py — BayesianFusion
  - Purpose: fuse independent agent probabilities into posterior probability
  - Inputs: agent_prob, confidence, correlation
  - Outputs: posterior probability (0-1)
  - Callers: orchestrator.portfolio_manager_node, backtest fallback

- quant_engine/monte_carlo.py — MonteCarloEngine (or quasi_mc)
  - Purpose: simulate price paths for confidence intervals and stress testing
  - Inputs: current_price, drift, vol forecasts, number of paths
  - Outputs: Monte Carlo results (confidence intervals, path outcomes)
  - Callers: RiskAgent, tests

- quant_engine/scoring.py — Fundamental scoring
  - Purpose: compute Piotroski F-score, Altman Z, Beneish M-score, valuation metrics
  - Inputs: income, balance, cashflow DataFrames and info
  - Outputs: FundamentalScores dataclass
  - Callers: FundamentalAgent

- quant_engine/options_intel.py — options analysis
  - Purpose: compute put/call ratio, IV vs RV, ATM straddle move
  - Inputs: options chain DataFrames
  - Outputs: option metrics used by VolatilityAgent
  - Callers: VolatilityAgent

- quant_engine/kalman.py — KalmanBeta
  - Purpose: dynamic beta estimation to market (SPY)
  - Inputs: price series for ticker and market
  - Outputs: dynamic beta and rolling correlation
  - Callers: VolatilityAgent

- quant_engine/hawkes.py — HawkesProcess
  - Purpose: detect event clustering and cascade risk
  - Inputs: return exceedance events
  - Outputs: HawkesResult (branching ratio, cascade risk)
  - Callers: API endpoints and backtests

- quant_engine/portfolio_optimizer.py — PortfolioOptimizer
  - Purpose: compute portfolio weights (max_sharpe, min_variance, risk_parity)
  - Inputs: tickers list, returns data, optional signals
  - Outputs: OptimizationResult (weights, expected return, volatility)
  - Callers: CLI run_optimize, API

# 12) backtesting/engine.py — BacktestEngine
- Purpose: simulate historical trading given signals and compute performance
- Important methods:
  - run_fast_backtest(ticker, ohlcv_df, signal_series) -> BacktestResult
    - Inputs: price history DataFrame, signal series (1 buy, -1 sell, 0 hold)
    - Outputs: BacktestResult (trades list, win_rate, total_return_pct, max_drawdown)
- Notes: Uses simple position sizing and trade rules for speed

# 13) database/manager.py — DatabaseManager
- Purpose: persist trades, portfolios, and agent logs using SQLAlchemy
- Important methods:
  - record_signal(ticker, signal_data, agents) -> trade_id
  - add_position(ticker, shares, avg_price, current_price=None) -> Portfolio
  - get_db() -> session generator
- Inputs: signal dict and AgentResult-like objects
- Outputs: DB rows and IDs

# 14) api/main.py — FastAPI server
- Purpose: run an HTTP server that exposes endpoints for status, signals, leaderboards, backtests, etc.
- Behavior:
  - Creates global registry = AgentRegistry() and graph = build_alpha_graph() on startup
  - Endpoints call graph.invoke or other helpers to return JSON
- Notes: useful for UI or programmatic access

# 15) data/universe.py — UniverseManager
- Purpose: run the pipeline across many tickers (bulk scan)
- Important method: scan(universe, top_n, registry=None, direction_filter=None)
  - Calls build_alpha_graph and registry (per ticker) and returns highest conviction results

---

If you want this as a file in the repository I have created `docs/function_tree_detailed.md` with this content. You can view and edit it.

Next steps I can take (pick one):
- Expand any single file with line-by-line explanation (e.g., orchestrator/graph.py) in the same simple style.
- Produce a JSON or DOT file representing this function tree for visualization.
- Run a full scan and add missing agent-specific detail lines if you want exhaustive coverage.

Tell me which next step you prefer.
