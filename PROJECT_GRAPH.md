# AlphaAgent — Project Graph Memory

> **Visual companion to `PROJECT_GRAPH.json`.** Same knowledge graph in Mermaid form, for humans and for any agent that renders markdown.
>
> Pair this with `PROJECT_GRAPH.json` for machine queries, `PROJECT_AUDIT.md` for architecture deep-dive, and `COMPLETE_INVENTORY.md` for the full factor list.

---

## 1. High-Level System Flow

```mermaid
flowchart TB
    User([User / Frontend])
    API[FastAPI Server<br/>api/main.py]
    Orch[LangGraph Orchestrator<br/>orchestrator/graph.py]

    subgraph Voters["8 Voting Agents (parallel)"]
        T[Technical<br/>60 factors]
        F[Fundamental<br/>54 factors]
        M[Macro<br/>43 factors]
        S[Sentiment<br/>25 factors]
        I[Insider<br/>19 factors]
        G[Geopolitical<br/>23 factors]
        C[Currency<br/>13 factors]
        V[Volatility<br/>12 factors]
    end

    R[Risk Agent<br/>Circuit Breaker<br/>24 factors]

    BF[Bayesian Fusion<br/>+ Meta-Learner<br/>+ Soft Regime Blend<br/>+ Entropy-Adaptive Gate]
    PR[Portfolio Risk<br/>VaR/CVaR/Stress]
    SD[Signal Decay<br/>OU Half-life]

    Out([SignalPacket])

    User --> API
    API --> Orch
    Orch --> Voters
    Orch --> R
    Voters --> BF
    R -.OVERRIDES.-> BF
    BF --> PR
    BF --> SD
    PR --> Out
    SD --> Out
    Out --> API
    API --> User

    classDef voter fill:#3b82f6,stroke:#1e40af,color:#fff
    classDef risk fill:#ef4444,stroke:#991b1b,color:#fff
    classDef fusion fill:#22c55e,stroke:#14532d,color:#fff
    class T,F,M,S,I,G,C,V voter
    class R risk
    class BF,PR,SD fusion
```

---

## 2. Agent → Quant Module Dependencies

```mermaid
flowchart LR
    subgraph Agents
        T[Technical]
        F[Fundamental]
        M[Macro]
        Sen[Sentiment]
        R[Risk]
        I[Insider]
        Geo[Geopolitical]
        V[Volatility]
    end

    subgraph QuantModules
        garch[garch.py]
        evt[evt.py]
        mc[monte_carlo.py]
        qmc[quasi_mc.py]
        hawkes[hawkes.py]
        kelly[kelly.py]
        dcc[dcc_garch.py]
        rmt[rmt.py]
        velo[vol_estimators.py]
        cusum[structural_break.py]
        copula[copula.py]
        scoring[scoring.py]
        pead[pead.py]
        sec_nlp[sec_nlp.py]
        ff5[factor_exposure.py]
        macro_m[macro.py]
        fred[fred_nowcast.py]
        hmm[hmm.py]
        tda[tda_signal.py]
        vpin[vpin.py]
        opts[options_intel.py]
        zdte[zero_dte.py]
        cot[cot_data.py]
        weather[weather_factor.py]
        eia[eia_petroleum.py]
        gtrends[google_trends.py]
        etfprem[etf_premium.py]
        roll[commodity_roll_yield.py]
        insider_m[insider.py]
        tech_m[technical.py]
        heston[heston.py]
        sabr[sabr.py]
        rvol[rough_vol.py]
        kalman[kalman.py]
        varb[vol_arbitrage.py]
    end

    R --> garch & evt & mc & qmc & hawkes & kelly & dcc & velo & cusum & copula
    dcc --> rmt
    F --> scoring & pead & sec_nlp & ff5
    M --> macro_m & fred
    T --> tech_m & hmm & tda & vpin & opts & zdte & cot & weather & eia & gtrends & etfprem & roll
    I --> insider_m
    V --> garch & heston & sabr & rvol & kalman & velo & varb & opts
```

---

## 3. Bayesian Fusion Hierarchy

```mermaid
flowchart TB
    subgraph Inputs["8 Agent Probability Outputs"]
        a1[Technical]
        a2[Fundamental]
        a3[Macro]
        a4[Sentiment]
        a5[Insider]
        a6[Geopolitical]
        a7[Currency]
        a8[Volatility]
    end

    subgraph RiskOverrides["Risk Circuit Breakers (Priority)"]
        cb1["1. BLACK_SWAN<br/>|Z|>5σ"]
        cb2["2. FLASH_CRASH<br/>3× daily vol"]
        cb3["3. CRITICAL_RISK<br/>EVT VaR99<-8%"]
        cb4["4. HIGH_RISK<br/>EVT VaR95<-5%"]
        cb5["5. GEO_SHOCK<br/>VIX>35 + commodity surge"]
        cb6["6. CARRY_UNWIND<br/>USD/JPY<125"]
    end

    Prior[Dynamic Prior<br/>SPY vs SMA50<br/>→ 0.47-0.53]

    BF[Bayesian Log-Odds Fusion<br/>+ Regime-Conditional Weights<br/>+ Correlation Penalty<br/>+ Soft HMM Blend]

    MetaLearner[LightGBM Meta-Learner<br/>+ Blend with Bayesian]

    Gate{Entropy-Adaptive<br/>Direction Gate<br/>0.485-0.515 to 0.44-0.56}

    Output["SignalPacket:<br/>LONG / SHORT / NEUTRAL<br/>+ probability<br/>+ conviction<br/>+ multiplier"]

    Inputs --> BF
    Prior --> BF
    BF --> MetaLearner
    MetaLearner --> Gate
    Gate --> Output

    RiskOverrides -.HALT/SCALE.-> Output

    classDef cb fill:#ef4444,stroke:#991b1b,color:#fff
    classDef fusion fill:#22c55e,stroke:#14532d,color:#fff
    class cb1,cb2,cb3,cb4,cb5,cb6 cb
    class BF,MetaLearner,Gate fusion
```

---

## 4. Theory → Module Implementation Map

```mermaid
flowchart LR
    subgraph VolTheory["Volatility Theories"]
        bollerslev[Bollerslev 1986 GARCH]
        engle[Engle 2002 DCC-GARCH]
        heston_t[Heston 1993]
        hagan[Hagan SABR 2002]
        bfg[Bayer-Friz-Gatheral 2016]
        yz[Yang-Zhang 2000]
    end

    subgraph RiskTheory["Risk Theories"]
        pbdh[Pickands-Balkema-de Haan]
        sobol[Sobol 1967]
        hawkes_t[Hawkes 1971]
        kelly_t[Kelly 1956]
        almgren[Almgren-Chriss 2000]
    end

    subgraph PortTheory["Portfolio Theories"]
        markow[Markowitz 1952]
        bl[Black-Litterman 1992]
        prado_hrp[Prado HRP 2016]
        prado_afml[Prado AFML 2018]
    end

    subgraph StatTheory["Statistical Theories"]
        mp[Marchenko-Pastur 1967]
        bp[Bailey-Prado 2014]
        bh[Benjamini-Hochberg 1995]
        kb[Koenker-Bassett 1978]
    end

    subgraph BehavTheory["Behavioural Theories"]
        ang[Ang 2006 Idio Vol]
        bali[Bali 2011 MAX]
        jeg[Jegadeesh 1990]
        dt[DeBondt-Thaler 1985]
        gh[George-Hwang 2004]
        dm[Daniel-Moskowitz 2016]
    end

    bollerslev --> garch[garch.py]
    engle --> dccg[dcc_garch.py]
    heston_t --> heston_m[heston.py]
    hagan --> sabr_m[sabr.py]
    bfg --> rvol_m[rough_vol.py]
    yz --> velo[vol_estimators.py]

    pbdh --> evt_m[evt.py]
    sobol --> qmc_m[quasi_mc.py]
    hawkes_t --> hk[hawkes.py]
    kelly_t --> kl[kelly.py]
    almgren --> tcm[transaction_costs.py]

    markow --> po[portfolio_optimizer.py]
    bl --> blit[black_litterman.py]
    prado_hrp --> hrp_m[hrp.py]
    prado_afml --> ml[ml_finance.py]

    mp --> rmt_m[rmt.py]
    bp --> ds[deflated_sharpe.py]
    bh --> ds
    kb --> qr[quantile_regression.py]

    ang & bali & jeg & dt & gh & dm --> tech_agent[Technical Agent]
```

---

## 5. Data Source → Module Map

```mermaid
flowchart LR
    subgraph FreeData["Free Data Sources"]
        yf[yfinance]
        fred[FRED]
        cftc[CFTC]
        noaa[NOAA]
        sec[SEC EDGAR]
        reddit[Reddit PRAW]
        gem[Gemini API]
        pyt[pytrends]
        finra[FINRA]
    end

    subgraph QuantModules
        eia_m[eia_petroleum.py]
        pead_m[pead.py]
        opts_m[options_intel.py]
        zdte_m[zero_dte.py]
        macro_m[macro.py]
        fred_n[fred_nowcast.py]
        cot_m[cot_data.py]
        weather_m[weather_factor.py]
        secnlp[sec_nlp.py]
        gtrend[google_trends.py]
    end

    subgraph Agents
        tech_a[Technical]
        sent_a[Sentiment]
        geo_a[Geopolitical]
        ins_a[Insider]
        fund_a[Fundamental]
        macro_a[Macro]
    end

    yf --> eia_m & pead_m & opts_m & zdte_m
    fred --> macro_m & fred_n
    cftc --> cot_m
    noaa --> weather_m
    sec --> secnlp
    pyt --> gtrend

    yf --> tech_a & sent_a & geo_a & macro_a
    reddit --> sent_a
    gem --> sent_a & geo_a & fund_a
    sec --> ins_a & fund_a
    finra --> ins_a
```

---

## 6. Module Categories (37 Modules in 11 Buckets)

```mermaid
flowchart TB
    subgraph Vol["Volatility (8)"]
        v1[garch.py]
        v2[heston.py]
        v3[sabr.py]
        v4[rough_vol.py]
        v5[vol_estimators.py]
        v6[vol_arbitrage.py]
        v7[dcc_garch.py]
        v8[quasi_mc.py]
    end
    subgraph Risk["Risk / Tail (5)"]
        r1[evt.py]
        r2[monte_carlo.py]
        r3[hawkes.py]
        r4[copula.py]
        r5[portfolio_risk.py]
    end
    subgraph Regime["Regime (4)"]
        rg1[hmm.py]
        rg2[markov_regime.py]
        rg3[regime_weights.py]
        rg4[structural_break.py]
    end
    subgraph Fusion["Fusion (4)"]
        fu1[bayesian.py]
        fu2[meta_learner.py]
        fu3[calibration.py]
        fu4[leaderboard.py]
    end
    subgraph Portfolio["Portfolio (3)"]
        p1[portfolio_optimizer.py]
        p2[black_litterman.py]
        p3[hrp.py]
    end
    subgraph Stats["Statistical (5)"]
        s1[rmt.py]
        s2[factor_orthogonalization.py]
        s3[deflated_sharpe.py]
        s4[quantile_regression.py]
        s5[ml_finance.py]
    end
    subgraph Micro["Microstructure (5)"]
        m1[vpin.py]
        m2[lob.py]
        m3[kalman.py]
        m4[granger.py]
        m5[causal_engine.py]
    end
    subgraph Topo["Topology (3)"]
        t1[tda_signal.py]
        t2[multifractal.py]
        t3[quantum_finance.py]
    end
    subgraph Exec["Execution (1)"]
        e1[transaction_costs.py]
    end
    subgraph Life["Lifecycle (4)"]
        l1[signal_decay.py]
        l2[kelly.py]
        l3[momentum.py]
        l4[factor_exposure.py]
    end
    subgraph Domain["Domain (12)"]
        d1[options_intel.py]
        d2[zero_dte.py]
        d3[pead.py]
        d4[sec_nlp.py]
        d5[scoring.py]
        d6[technical.py]
        d7[macro.py]
        d8[insider.py]
        d9[etf_premium.py]
        d10[commodity_roll_yield.py]
        d11[cot_data.py]
        d12[weather_factor.py]
    end
```

---

## 7. Implementation Lineage (4 Passes)

```mermaid
gantt
    title AlphaAgent Implementation Passes
    dateFormat YYYY-MM-DD
    section Pass 1
    Core gaps + dynamic params      :p1, 2026-06-14, 1d
    MOVE/TED/VIX3M to Macro         :p1a, 2026-06-14, 1d
    4 fundamental factors           :p1b, 2026-06-14, 1d
    DCC-GARCH module                :p1c, 2026-06-14, 1d
    Dynamic WACC/Bollinger/halflife :p1d, 2026-06-14, 1d
    section Pass 2
    Behavioural + methodology       :p2, 2026-06-14, 1d
    7 behavioural factors           :p2a, 2026-06-14, 1d
    R&D + QMJ                        :p2b, 2026-06-14, 1d
    Nelson-Siegel + Business Cycle  :p2c, 2026-06-14, 1d
    Black-Litterman + HRP + CUSUM   :p2d, 2026-06-14, 1d
    section Pass 3-4
    Vol estimators + TCM             :p3, 2026-06-15, 1d
    RMT + Portfolio VaR              :p3a, 2026-06-15, 1d
    Deflated Sharpe + Prado ML       :p3b, 2026-06-15, 1d
    Soft regime blending             :p3c, 2026-06-15, 1d
    section Final
    Free data integration            :p4, 2026-06-15, 1d
    COT + Weather + EIA              :p4a, 2026-06-15, 1d
    FRED Nowcast + Google Trends     :p4b, 2026-06-15, 1d
    Vol Arbitrage + Quantile Reg     :p4c, 2026-06-15, 1d
    Factor Orthogonalization         :p4d, 2026-06-15, 1d
```

---

## 8. End-to-End Request Flow

```mermaid
sequenceDiagram
    participant U as User/Frontend
    participant API as FastAPI
    participant G as LangGraph
    participant D as Data Ingestion
    participant A as 9 Agents (parallel)
    participant PM as Portfolio Manager
    participant DB as Database

    U->>API: GET /api/signal/AAPL
    API->>G: build_alpha_graph().invoke(state)
    G->>D: Pre-warm OHLCV/financials/news/options
    D-->>G: Cached data ready
    G->>A: Run 9 agents in parallel (ThreadPoolExecutor)

    par Technical
        A->>A: 60 factors
    and Fundamental
        A->>A: 54 factors
    and Macro
        A->>A: 43 factors
    and Sentiment
        A->>A: 25 factors
    and Risk
        A->>A: 24 factors + circuit breakers
    and Insider
        A->>A: 19 factors
    and Geopolitical
        A->>A: 23 factors
    and Currency
        A->>A: 13 factors
    and Volatility
        A->>A: 12 factors
    end

    A-->>G: AgentResults dict
    G->>PM: portfolio_manager_node
    PM->>PM: Risk circuit breakers
    PM->>PM: HMM regime + soft blend
    PM->>PM: Bayesian fusion + correlation penalty
    PM->>PM: Meta-learner blend
    PM->>PM: Entropy-adaptive direction gate
    PM->>PM: HMM regime obj + Signal decay
    PM->>PM: Portfolio VaR (single-position)
    PM-->>G: SignalPacket
    G-->>API: state.final_signal
    API->>DB: Persist SignalHistory
    API-->>U: SignalPacket JSON
```

---

## 9. Query Examples (For Other Agents)

Other agents can ask these questions of `PROJECT_GRAPH.json`:

| Question | JSON Query |
|----------|-----------|
| **What modules does Risk agent use?** | `edges WHERE source='agent_risk' AND type='USES'` |
| **What theory does DCC-GARCH implement?** | `edges WHERE source='qm_dcc_garch' AND type='IMPLEMENTS'` |
| **What agents read from yfinance?** | `edges WHERE target='ds_yfinance' AND type='READS_FROM'` |
| **Which circuit breakers halt trading?** | `nodes WHERE type='circuit_breaker' AND priority<=2` |
| **What was added in Pass 4?** | `nodes WHERE added_in_pass=4` |
| **What feeds into Bayesian fusion?** | `edges WHERE target='qm_bayesian' AND type='FEEDS_INTO'` |
| **Which modules in 'volatility' category?** | `nodes WHERE type='quant_module' AND category='volatility'` |
| **What does the orchestrator coordinate?** | `edges WHERE source='orchestrator' AND type='COORDINATES'` |
| **Which API endpoints exist?** | `nodes WHERE type='api_endpoint'` |
| **Theories by López de Prado?** | `nodes WHERE type='theory' AND author CONTAINS 'Prado'` |

---

## 10. Quick Stats

| Metric | Count |
|--------|-------|
| Agents (voters) | 8 |
| Agents (circuit breakers) | 1 |
| Quant engine modules | 37 |
| Academic theories implemented | 60+ |
| Free data sources | 10 |
| Factors across all agents | 271 |
| Circuit breakers | 6 |
| Dynamic parameters | 6 |
| API endpoints (REST + WS) | 10 |
| Frontend tabs | 6 |
| Database tables | 6 |
| Implementation passes | 4 |

---

## 11. How to Share This Memory

**For another Claude Code instance:**
```
1. Copy these 4 files to the new project:
   - PROJECT_GRAPH.json       (machine-readable graph)
   - PROJECT_GRAPH.md          (this file — visual)
   - PROJECT_AUDIT.md          (architecture deep-dive)
   - COMPLETE_INVENTORY.md    (every factor enumerated)

2. Tell the new agent: "Read PROJECT_GRAPH.json + PROJECT_AUDIT.md first.
   That's the system. Use COMPLETE_INVENTORY.md as the factor reference."

3. The new agent can query the JSON graph to answer questions about
   dependencies, theory mapping, data sources without re-reading source code.
```

**For another agent framework (LangChain/AutoGen/CrewAI):**
```python
# Load the graph
import json
with open("PROJECT_GRAPH.json") as f:
    project_graph = json.load(f)

# Find all modules that implement a Prado theory
prado_theories = [n["id"] for n in project_graph["nodes"]
                  if n["type"] == "theory" and "Prado" in n.get("author", "")]
prado_modules = [e["source"] for e in project_graph["edges"]
                 if e["target"] in prado_theories and e["type"] == "IMPLEMENTS"]
print(prado_modules)
# → ["qm_hrp", "qm_ml_finance", "qm_deflated_sharpe"]
```

**For a non-technical reader:**
Just open this file (`PROJECT_GRAPH.md`) in a Markdown viewer with Mermaid support
(GitHub, VS Code with Mermaid extension, or any Mermaid-aware tool). The diagrams
render visually.

---

**End of Project Graph Memory.** Pair with `PROJECT_GRAPH.json` for machine queries.
