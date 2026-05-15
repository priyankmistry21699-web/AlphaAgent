# AlphaAgent — Detailed Function Tree (simple wording)

NOTE: I scanned the repository and produced a simplified, beginner-friendly function tree. It lists files, classes, and functions with: purpose, inputs (parameters and data sources), outputs, methods, and short notes. I updated this file to include a clear "call map" that shows which files call which files and which functions call which functions (function-level callers) in simple language.

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

## New: Call Map (which files call which files and which functions call which functions)
This section lists the main files and shows: who calls them (files) and which specific function(s) call which function(s). Wording is simple.

- cli.py
  - Who runs this file: You (run `python cli.py ...`) or the system (scripts/tests) start here.
  - Files that import or call cli.py: none (it is an entry point).
  - Functions called by cli.py:
    - `MarketData(...)` → constructs MarketData (calls MarketData.__init__ in data/market.py)
    - `AgentRegistry()` → constructs AgentRegistry (agents/registry.py)
    - `build_alpha_graph()` and `AlphaGraphState(...)` → from orchestrator/graph.py
    - `graph.invoke(state)` → runs orchestrator which calls many agents

- data/market.py (MarketData)
  - Who calls/get uses MarketData:
    - orchestrator/graph.py (data_ingestion_node calls `md.get_ohlcv`, `md.get_financials`, `md.get_info`)
    - agents/*.py (TechnicalAgent, FundamentalAgent, VolatilityAgent, RiskAgent, InsiderAgent, CurrencyAgent, etc.) call methods like `get_ohlcv`, `get_info`, `get_returns`, `get_options_chain`.
    - tests and scripts (tests/*.py) call MarketData directly in examples.
  - Function-level calls inside MarketData:
    - `get_ohlcv()` calls `self.cache.get()` and `self._yf_ticker.history(...)` and `validate_ohlcv()` (data/validation.py)
    - `get_info()` calls `self.cache.get()` and `self._yf_ticker.info` and `validate_financials()`
    - `get_financials()` calls `.quarterly_financials` attributes and `self.cache.set()`

- data/cache.py (DataCache)
  - Who calls DataCache methods: MarketData (get/set), and possibly other data modules that cache API results.
  - Functions called: `DataCache.get(namespace, key, subkey)` and `DataCache.set(namespace, key, subkey, value, ttl_seconds)`.

- data/validation.py
  - Who calls validation helpers: MarketData (validate_ohlcv, validate_financials, compute_returns, ensure_minimum_history), tests, backtester.
  - Function-level mapping:
    - `validate_ohlcv(df, ticker)` is called by `MarketData.get_ohlcv()`
    - `compute_returns()` is called by `MarketData.get_returns()` and by agents/backtest code that needs return series

- config/settings_manager.py (SettingsManager)
  - Who calls settings: virtually every module (agents, MarketData, quant_engine) call `settings.get()` or `settings.get_section()`.
  - Function-level mapping:
    - `settings.get_section('cache')` is called by `MarketData._cache_cfg()` to obtain TTLs
    - Agents read `settings.get('agent_defaults.*')` inside their logic

- agents/registry.py (AgentRegistry)
  - Who calls AgentRegistry:
    - cli.py (constructs registry)
    - api/main.py (creates a global registry on startup)
    - data/universe.py (scan() builds registry if not provided)
    - tests/scripts that need to run agents directly
  - Function-level mapping:
    - `AgentRegistry.__init__()` calls `self.register()` repeatedly, which instantiates each Agent class (e.g., `TechnicalAgent()`)
    - `register(agent)` checks `self.config` (from config/agents.yaml) to decide whether to store the agent in `_agents`
    - Other modules call `registry.get_active_agents()` and iterate the returned dict to run agents

- orchestrator/graph.py (AlphaGraph)
  - Who calls orchestrator:
    - cli.py (`build_alpha_graph()` and `graph.invoke(state)`)
    - api/main.py (build graph at startup and call in endpoints)
    - data/universe.py (bulk scan calls graph.invoke for each ticker)
    - tests (test_day7.py, etc.)
  - Function-level mapping (important functions calling other functions):
    - `build_alpha_graph()` returns a graph object with nodes.
    - `graph.invoke(state)` executes nodes in sequence. Nodes call these functions:
      - `data_ingestion_node(state)` → calls `md.get_ohlcv()`, `md.get_financials()`, `md.get_info()`
      - `run_agents_node(state)` → calls `state.registry.get_active_agents()` then, for each agent, submits `agent.analyze(state.ticker, state.market_data)` to a ThreadPoolExecutor. `agent.analyze()` calls the agent's `_run_analysis()` internally.
      - `portfolio_manager_node(state)` → reads `state.agent_results` and calls `BayesianFusion.update()` (quant_engine.bayesian) and possibly `RegimeDetector.fit_predict()` (quant_engine.hmm) and `SignalDecay` (quant_engine.signal_decay)
    - The `run_agents_node` maps each agent name → AgentResult by calling `agent.analyze()`; `analyze()` measures time and wraps `_run_analysis()`.

- agents/*.py (each Agent file)
  - Who calls each agent (files):
    - agents are instantiated by AgentRegistry and executed by orchestrator.run_agents_node (via agent.analyze())
    - tests also call agents directly (e.g., `registry.get_agent('technical').analyze(ticker, md)`)
  - Example function-level calls inside agents:
    - `TechnicalAgent._run_analysis()` calls `data.get_ohlcv()`, `compute_indicators()` (quant_engine.technical), `MomentumEngine.analyze()` (quant_engine.momentum), and `RegimeDetector.fit_predict()` (quant_engine.hmm).
    - `VolatilityAgent._run_analysis()` calls `data.get_returns()`, `GARCHModel.fit_and_forecast()` (quant_engine.garch), `analyze_options()` (quant_engine.options_intel), and `KalmanBeta` (quant_engine.kalman).
    - `RiskAgent._run_analysis()` calls `GARCHModel`, `MonteCarloEngine.simulate_garch()`, `ExtremeValueModel.fit_and_calculate()`, then `KellyCriterion.calculate()`.
    - `InsiderAgent._run_analysis()` calls `data.get_insider_transactions()` and `analyze_insider_data()` (quant_engine.insider).

- quant_engine/* (math kernels)
  - Who calls quant_engine modules:
    - Agents (technical, fundamental, volatility, risk, insider) call specific quant_engine functions/classes as listed in each agent's section.
    - orchestrator.portfolio_manager_node may call `RegimeDetector` or `BayesianFusion` for final processing.
  - Example function-level mapping:
    - `quant_engine.garch.GARCHModel.fit_and_forecast()` is called by `VolatilityAgent._run_analysis()` and `RiskAgent._run_analysis()`
    - `quant_engine.bayesian.BayesianFusion.update(agent_prob, confidence, correlation)` is called inside `orchestrator.portfolio_manager_node` to combine probabilities.

- backtesting/engine.py (BacktestEngine)
  - Who calls BacktestEngine:
    - cli.py (when --backtest flag used)
    - tests/backtest scripts
  - Function-level mapping:
    - `BacktestEngine.run_fast_backtest(ticker, ohlcv_df, signal_series)` expects precomputed `signal_series` (1,0,-1) and may call quant_engine functions in its scoring logic (e.g., uses GARCH, HMM, Bayesian fallback inside the backtest math)

- database/manager.py (DatabaseManager)
  - Who calls DatabaseManager:
    - orchestrator/graph.py or api/main.py may call `record_signal()` to persist final signals
    - paper_trader.py and other trading modules call add_position/record_signal
  - Function-level mapping:
    - `DatabaseManager.record_signal(ticker, signal_data, agents)` creates `Trade` and `AgentLog` rows using AgentResult data

- api/main.py (FastAPI server)
  - Who calls API main: run uvicorn to start server; it creates global objects on startup
  - Function-level mapping:
    - On startup: `registry = AgentRegistry()`, `graph = build_alpha_graph()`, `db_manager = DatabaseManager()`
    - Endpoints call `graph.invoke()` or helpers that run the orchestrator which triggers agents

- data/universe.py (UniverseManager.scan)
  - Who calls UniverseManager.scan:
    - CLI `--scan` command and API endpoints for bulk scanning
  - Function-level mapping:
    - For each ticker in the universe: it creates `MarketData(ticker)` and calls `graph.invoke({"ticker": ticker, "market_data": md, "registry": reg})`

- tests/ and scripts/
  - Who runs them: developer/test runner; they call specific agents, orchestrator, or MarketData to validate behavior.
  - Example mappings found in tests:
    - `tests/test_day5.py` creates `MarketData`, `AgentRegistry`, and runs `TechnicalAgent.analyze()` directly.
    - `tests/test_day7.py` uses `build_alpha_graph()` and streams the graph to see node-level outputs.

---

The above call map adds both file-level and function-level caller/callee links for the main parts of the system. If you want, I will now:

1) Expand this call map into a new section that lists, for every file in the repo, an exact list of files that import it (and line examples) — this will be exhaustive but longer.
2) Produce a machine-readable JSON file that contains the call graph (edges from caller -> callee both at file and function level), saved as `docs/call_map.json` so you can visualize it with tools.
3) Generate a DOT/Graphviz file `docs/call_map.dot` and render it to PNG for a visual map.

Which option do you prefer? If you want the exhaustive import list (1), I will run a repo-wide search and update the docs with exact import lines. 

---

(I'm ready to run the chosen job and commit the results into the repo.)
