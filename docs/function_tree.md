# AlphaAgent — Function Tree (initial scan)

NOTE: This file was generated from an automated scan of a subset of the repository (top code search results). The search results may be incomplete. To view the full repository, see: https://github.com/priyankmistry21699-web/AlphaAgent

Summary: a hierarchical function/class tree for the scanned modules. Each node lists: variables/state, methods/functions, key calculations, purpose, data sources (where it gets inputs), and outputs (what it returns/emits).

---

## Top-level flow (high-level)
- CLI (cli.py) → MarketData (data/market.py) + AgentRegistry (agents/registry.py) → Agents (agents/*.py) → Orchestrator/Graph (orchestrator.graph — not fully scanned) → Outputs: AgentResult, VolatilityResult, BacktestResult, DB records

---

## cli.py
- Purpose: Rich CLI to run analysis, backtests, scans, leaderboard, and optimizations.
- Key functions:
  - run_analysis(ticker: str, output_json: bool = False, agent_filter: Optional[list] = None)
    - Variables: ticker, market_data, registry, graph, state, result_state, final
    - Calls/Dependencies: data.market.MarketData, agents.registry.AgentRegistry, orchestrator.graph.build_alpha_graph, AlphaGraphState
    - Calculations: invokes graph.invoke(state) to run pipeline (delegates computation to agents and orchestrator)
    - Data sources: yfinance via MarketData, local config via settings in underlying modules
    - Outputs: final signal (from graph state), printed rich UI
- Notes: Uses rich.Progress for UX and handles exceptions to print errors and exit.

---

## data/market.py (class MarketData)
- Purpose: Unified market data wrapper around yfinance with caching, validation, and helper accessors.
- Variables / state:
  - ticker (str), cache (DataCache), _yf_ticker (yfinance.Ticker), _ohlcv_cache (dict), TTL properties (OHLCV_TTL, INFO_TTL, etc.)
- Key methods:
  - get_ohlcv(period: str = "1y", interval: str = "1d") -> pd.DataFrame
    - Inputs: period, interval
    - Flow: check in-memory _ohlcv_cache → check SQLite cache via DataCache → fetch from yfinance if needed → validate via data.validation.validate_ohlcv
    - Outputs: validated DataFrame with Open/High/Low/Close/Volume
  - get_info(), get_financials(), get_options(), get_insiders(), get_holders(), get_returns(), etc. (implied by docstring)
- Calculations: computes returns via compute_returns in data.validation, ensures minimum history, may convert/clean raw yfinance outputs
- Data sources: yfinance API, SQLite cache (DataCache), settings for TTLs
- Outputs: pandas DataFrames, dicts (info), time series for downstream agents

---

## config/settings_manager.py (SettingsManager singleton)
- Purpose: Thread-safe settings loader with hot-reload and save.
- Variables:
  - _data (dict), _rw_lock (RLock), _instance singleton
- Key methods:
  - get(key: str, default: Any = None) -> Any
    - Reads nested key via dot notation under lock and returns value or default
  - get_section(section: str) -> dict
  - reload(), save(), all()
- Data sources: config/settings.yaml file on disk
- Outputs: configuration values used by agents and data layers

---

## agents/registry.py (AgentRegistry)
- Purpose: Auto-discover and manage lifecycle of analysis agents per config/config/agents.yaml.
- Variables:
  - config_path, config (dict), _agents (dict[str, BaseAgent])
- Key methods:
  - __init__(config_path: str = None): loads config and registers agent instances
  - _load_config() -> dict: reads YAML config
  - register(agent: BaseAgent): registers agent if enabled in config
  - get_agent(name: str) -> BaseAgent
  - get_active_agents() -> Dict[str, BaseAgent]
  - get_agent_weight(name: str, regime: str = 'normal') -> float: returns base weight or regime override
- Data sources: config/agents.yaml, agent classes (imported modules)
- Outputs: live agent instances, weights for voting/fusion

---

## agents/state.py (Pydantic models)
- Purpose: Typed schemas for inter-agent communication and outputs.
- Key models:
  - Direction (Enum): LONG/SHORT/HOLD
  - Confidence (Enum): HIGH/MEDIUM/LOW
  - Regime / VolRegime (Enums)
  - FactorScore (BaseModel): name, value, score (0-100), interpretation
  - AgentResult (BaseModel): agent_name, vote (Direction), probability_up (0-1), confidence (0-1), reasoning (str), factor_scores (dict[str, FactorScore]), warnings, computation_time_ms, timestamp
  - VolatilityResult, MonteCarloResult, ConfidenceInterval (not fully displayed)
- Data sources: constructed by agents after computations
- Outputs: standardized JSON-serializable objects passed through orchestrator/graph and persisted to DB or presented in UI

---

## agents/technical.py (TechnicalAgent : BaseAgent)
- Purpose: Compute technical indicators, momentum, HMM regime, seasonality and return structured AgentResult.
- Variables:
  - name = "technical"
  - period, min_days, min_momentum_days (from settings)
  - ohlcv (DataFrame)
- Key methods:
  - _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult
    - Steps:
      1. Load OHLCV from data.get_ohlcv(period)
      2. Check minimum data days
      3. compute_indicators(ohlcv) → indicators (RSI, MACD, Bollinger, SMA/EMA, ADX, OBV, VWAP, Stochastic)
      4. MomentumEngine(ohlcv['Close']).analyze() → momentum_result (Hurst, Jegadeesh-Titman)
      5. RegimeDetector.fit_predict(returns) → regime_result (HMM states)
      6. compute_seasonality_score(date) → seasonality_scores
      7. Build FactorScore dict and assemble AgentResult (vote, probability_up, confidence, reasoning)
- Calculations: indicator math (delegated to quant_engine.technical), momentum statistics, HMM regime inference, seasonality heuristic
- Data sources: MarketData (ohlcv via yfinance), settings manager, quant_engine modules
- Outputs: AgentResult with factor_scores, vote, probability, confidence

---

## agents/currency.py (CurrencyAgent)
- Purpose: Evaluate FX market dynamics and estimate impact on US-listed stocks.
- Variables:
  - name = "currency"
  - fx_data = CurrencyData() (in __init__)
  - SECTOR_FX_EXPOSURE mapping (module-level)
- Key methods:
  - _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult
    - Steps:
      1. Fetch FX snapshot: snap = self.fx_data.get_snapshot()
      2. Get company sector via data.get_info()
      3. Compute fx_exposure from SECTOR_FX_EXPOSURE
      4. Compute DXY regime impacts (dxy_1m, dxy_3m) and adjust prob_up, confidence
      5. Build factor_scores, reasoning strings, warnings as applicable
- Calculations: linear impact approximations (e.g., impact = -dxy_1m * fx_exposure * 0.3), thresholds for strong/weak USD
- Data sources: data.currency.CurrencyData (likely uses external FX data providers), company info via MarketData
- Outputs: AgentResult

---

## backtesting/engine.py (BacktestEngine)
- Purpose: Run fast/vectorized historical backtests using pre-computed signals and compute PnL metrics.
- Variables/state:
  - initial_capital, current_capital, trades (list[Trade]), equity_curve
- Key methods:
  - run_fast_backtest(ticker: str, ohlcv_df: pd.DataFrame, signal_series: pd.Series) -> BacktestResult
    - Loop over historical days using yesterday's signal to act today
    - Enter logic: if signal == 1 → open LONG with position_size = current_capital * 0.10
    - Exit logic: if active_trade and signal flips to -1
    - Calculates daily pnl for closed trades, updates current_capital and equity_curve
    - After loop compute win_rate, total_return_pct, max_drawdown_pct
- Data sources: OHLCV DataFrame, pre-computed signal series (1, -1, 0)
- Outputs: BacktestResult dataclass with trades and aggregate metrics

---

## database/manager.py (DatabaseManager)
- Purpose: SQLite DB manager using SQLAlchemy for persistence of trades, portfolios, and agent logs.
- Variables:
  - engine, SessionLocal, DB_URL, file paths
- Key methods:
  - get_db() generator that yields DB session
  - record_signal(ticker: str, signal_data: dict, agents: list) -> int
    - Persists Trade and AgentLog records and returns new_trade.id
  - add_position(ticker, shares, avg_price, current_price=None) -> Portfolio
    - Upserts position, blends shares and avg_price, updates allocation and timestamps
- Data sources: models in database.models, inputs from orchestrator/CLI/backtest
- Outputs: DB records (Trade, AgentLog, Portfolio)

---

## api/monitoring.py (Monitor)
- Purpose: Thread-safe telemetry for requests and agent runs (latency, errors, counts)
- Variables:
  - APIMetrics dataclass instance, _lock threading.Lock
- Key methods:
  - record_request(latency_ms: float, success: bool)
  - record_agent_run(agent_name: str, latency_ms: float, success: bool)
  - get_summary() -> dict
  - reset()
- Calculations: rolling averages for latency, per-agent avg latency and error counters
- Data sources: callers instrumenting Monitor.record_*
- Outputs: runtime metrics dictionary

---

## Missing / partially scanned modules
The automated search returned a limited set of files (ten results). There are additional modules referenced but not fully analyzed here, e.g.:
- orchestrator/graph.py (build_alpha_graph, AlphaGraphState) — crucial for how AgentResults are fused into final signal
- agents/fundamental.py, sentiment.py, macro.py, volatility.py, risk.py, insider.py, geopolitical.py — other agents listed in the registry were not scanned in this run
- quant_engine.* modules (technical, momentum, hmm) — implementations of many calculations were not fully scanned
- data/cache.py, data/validation.py, data/currency.py — data helpers and sources were only referenced
- database/models.py — DB schema was not scanned

To perform a complete function tree I should scan the rest of the repo. The code search tool is limited to 10 results per request; I can iterate further or run targeted searches for the modules above and then update/expand this file.

---

If you want, I will:
1. Expand the scan to cover all agents and the orchestrator and quant_engine modules, and then update this function tree to be complete.
2. Or generate a visual (DOT/graphviz) function tree or JSON file suitable for programmatic consumption.

Tell me which you prefer and I'll continue.
