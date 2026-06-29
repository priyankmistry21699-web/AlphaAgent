import { useState } from 'react';

const AGENTS = [
  {
    name: 'Technical',
    color: 'var(--blue)',
    theory: 'Price Action & Chart Patterns',
    factors: ['RSI (14) — adaptive thresholds by asset class', 'MACD 12/26/9 + crossover detection', 'Bollinger Bands %B + bandwidth', 'ATR (14) — volatility proxy', 'OBV trend & accumulation', 'EMA 9/21 + crossover', 'SMA 50/200 golden/death cross', 'ADX (14) trend strength', 'Stochastic %K/%D (14/3)', 'VWAP deviation', 'Hurst exponent (long-memory)', 'Jegadeesh-Titman 12M-1M momentum', 'HMM market regime (3 states)', 'Seasonality / calendar factor', 'Fibonacci Retracement (52W range — 0.236/0.382/0.5/0.618/0.786)', 'Pivot Points (Classic Daily + Weekly R1/R2/S1/S2)', 'Multi-Timeframe RSI+MACD confluence (Daily vs Weekly)', 'Chart Pattern Recognition (Double Top / Double Bottom)', 'Volume Profile — Point of Control (POC) + Value Area', 'Order Flow Imbalance (CLV — close location value, buying pressure)', 'Cross-Sectional Momentum Rank vs sector peers (22d)', 'Signal Backtest Calibration (conditional hit rate, 126-day lookback)', 'Ichimoku Cloud (Tenkan / Kijun / Senkou A+B / cloud position)', 'Chaikin Money Flow (CMF-14) — accumulation vs distribution', 'TRIX Oscillator (15-period triple-smoothed EMA momentum)', 'Parabolic SAR — trailing stop & trend reversal detector', 'Supply / Demand Zones (volume-spike pivot high/low detection)', 'VPIN — Volume-Synchronized Probability of Informed Trading (Easley et al. 2012)', '0DTE Options Flow — same-day expiry net call/put OI + dealer GEX'],
  },
  {
    name: 'Fundamental',
    color: 'var(--green)',
    theory: 'Value & Quality Investing',
    factors: ['P/E TTM + Forward P/E', 'Price/Book ratio', 'EV/EBITDA', 'ROE / ROIC', 'Debt/Equity ratio', 'EPS growth YoY', 'Revenue growth', 'Free cash flow yield', 'Analyst consensus rating', 'Earnings surprise history', 'Piotroski F-Score (9-pt quality)', 'Altman Z-Score (bankruptcy risk)', 'Beneish M-Score (earnings manipulation)', 'Fama-French 5-Factor exposure', "CAPM Jensen's Alpha", 'Accruals ratio (earnings quality)', 'M&A Activity (EDGAR 8-K, 180d)', 'Earnings Revision Momentum (EPS estimate trend 30/60d)', 'Graham Number — √(22.5 × EPS × BVPS) intrinsic floor', 'Earnings Call NLP — quality, guidance tone, red flags (Gemini)', 'Net Debt / EBITDA leverage ratio', 'Dividend Yield + Payout Ratio sustainability check', 'PEAD — Post-Earnings Announcement Drift (Bernard & Thomas 1989, SUE-based)', 'Sector-Conditional Valuation Thresholds (P/E, P/B benchmarks by sector)'],
  },
  {
    name: 'Momentum',
    color: 'var(--cyan)',
    theory: 'Trend Following & Relative Strength',
    factors: ['1M / 3M / 6M / 12M return', 'Cross-sectional RS vs sector', 'Price vs 52W high/low', 'Momentum Z-score', 'Rate of change (ROC)', 'Trend acceleration', 'Volume-confirmed momentum'],
  },
  {
    name: 'Macro',
    color: 'var(--yellow)',
    theory: 'Macroeconomic Regime Analysis',
    factors: ['Yield curve 10Y-2Y spread', 'Fed Funds Rate level', 'VIX — rolling percentile score (dynamic)', 'Unemployment rate (FRED)', 'CPI inflation YoY (FRED)', 'PCE inflation YoY — Fed preferred gauge (FRED PCEPI)', 'Credit spreads HYG/LQD', 'Copper/Gold ratio (growth signal)', 'BTC risk-on proxy', 'ACWI vs SPY global breadth', 'DXY dollar momentum', 'M2 money supply trend', 'ISM Manufacturing PMI', 'NAIRU — natural unemployment (FRED NROU)', 'r* neutral rate (DFII10+2.0)', 'Fed balance sheet trend (WALCL)', 'SOX/SPY semiconductor relative strength', 'Sector relative strength vs SPY', 'Contagion correlation spike (SPY/TLT/GLD/HYG)', 'Real interest rate (DGS10−T10YIE)', 'Retail Sales MoM ex-auto (FRED RSXFS — consumer health)', 'Leading Economic Index trend (FRED USSLIND — forward indicator)', 'Financial Conditions Index (FRED NFCI — tightening/easing signal)', 'HY Credit Spread OAS (FRED BAMLH0A0HYM2 — credit conditions)', 'Equity Risk Premium E/P − Real Rate (equity vs bond attractiveness)'],
  },
  {
    name: 'Volatility',
    color: 'var(--orange)',
    theory: 'GARCH & Vol Surface Analysis',
    factors: ['GARCH(1,1) forecast 1d/5d/10d', 'EGARCH asymmetry (leverage effect)', 'Vol regime: LOW/NORM/HIGH/EXTREME', 'Vol percentile (1Y rolling)', 'VVIX — vol of vol', 'VIX3M term structure (contango/backwardation)', 'IV vs realized vol spread', 'SABR vol smile fit', 'Heston stochastic vol params', 'Rough vol Hurst exponent', 'Hawkes self-exciting process (event clustering)', 'Multifractal MF-DFA scaling exponent'],
  },
  {
    name: 'Sentiment',
    color: 'var(--purple)',
    theory: 'Behavioral & Crowd Psychology',
    factors: ['News NLP sentiment score', 'Social media buzz index', 'Fear & Greed index', 'Put/call ratio', 'Short interest %', 'Analyst rating consensus', 'Insider buy/sell ratio', 'Options skew (put/call IV spread)', 'Transfer entropy (news→price causality)', 'Shannon entropy (signal disagreement)', 'AAII bull-bear survey spread', 'Price target consensus upside %', 'Source credibility weighting (tier scoring)', 'Headline/body alignment (Gemini LLM)', 'Unusual Options Activity (vol/OI > 2x sweep detection)', 'FinBERT-Style NLP — sentiment, guidance, tone, catalyst (Gemini)', 'Reddit Sentiment — WSB + r/stocks + r/investing post scanning', 'News Decay Model — exponential age-weighted sentiment (half-life 4.6d)', 'Short-Squeeze Score — short float % + days-to-cover (squeeze risk)', 'Earnings Whisper — options implied move vs historical post-earnings move'],
  },
  {
    name: 'Risk',
    color: 'var(--red)',
    theory: 'Monte Carlo VaR & Tail Risk',
    factors: ['MC VaR 95%/99% (paths scale with GARCH regime)', 'CVaR / Expected Shortfall', 'Max drawdown (simulated paths)', 'Black Swan Z-score (>5σ detector)', 'Flash crash detector (>7% 1d)', 'Carry trade unwind signal', 'War/geo shock VIX spike', 'EVT GPD tail fit', 'Kelly position size (regime-adaptive cap)', 'Rolling Sharpe ratio (63-day annualized)', 'Rolling Sortino ratio (63-day downside σ)', 'Vanna/Charm exposure (BSM second-order greeks, OI-weighted)', 'Return Skewness & Excess Kurtosis (fat-tail distribution risk)', 'Options GEX Wall — net dealer gamma exposure (pinning / vol amplification)', 'Pairs Trading Spread Z-score (cointegrated sector pair mean-reversion)', 'CDS Spread Proxy — HYG/LQD ratio (credit leads equity, Moody\'s methodology)'],
  },
  {
    name: 'Insider',
    color: 'var(--dim2)',
    theory: 'Smart Money & Flow Analysis',
    factors: ['SEC Form 4 insider transactions', 'Institutional holder changes QoQ', 'Congressional STOCK Act filings', 'Mutual fund flow changes', 'Major holder % outstanding', 'Cluster buying/selling (3+ insiders)', "Kyle's Lambda (market impact cost)", 'Amihud illiquidity ratio', 'FINRA short sale volume % (weekly RegSHO)', 'ETF sector flow impact signal', 'Activist 13D / SC 13G filings (EDGAR, 90d)', 'Top-10 holder concentration (HHI)', 'Dark Pool Print Ratio (FINRA ADF off-exchange volume %)'],
  },
  {
    name: 'Geopolitical',
    color: '#f87171',
    theory: 'Event Risk & Macro Shock',
    factors: ['Geopolitical risk index (GPR — Caldara & Iacoviello)', 'VIX event spike detector (+5pt/1d)', 'Currency stress index', 'Commodity disruption signals', 'Central bank surprise score', 'Active conflict score (keyword NLP)', 'Sanctions / embargo risk (keyword NLP)', 'Tariff & regulatory risk (keyword NLP)'],
  },
];

const QUANT_THEORIES = [
  {
    name: 'Bayesian Fusion',
    category: 'ENSEMBLE',
    color: 'var(--blue)',
    usedBy: 'Orchestrator — combines all 8 voting agents',
    desc: 'Sequential prior→posterior updating: each agent shifts P(Up) based on its probability and correlation penalty. Final posterior = ensemble signal.',
    role: 'Directly produces final P(Up)',
  },
  {
    name: 'GARCH(1,1) + EGARCH',
    category: 'VOLATILITY',
    color: 'var(--orange)',
    usedBy: 'Volatility Agent, Risk Agent',
    desc: 'Models time-varying volatility: σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1). EGARCH adds asymmetry — downside shocks increase vol more than upside.',
    role: 'Vol regime → probability score + MC path vol input',
  },
  {
    name: 'Monte Carlo GBM',
    category: 'SIMULATION',
    color: 'var(--cyan)',
    usedBy: 'Risk Agent',
    desc: 'Simulates price paths via Geometric Brownian Motion. Paths scale with GARCH regime: LOW=3k, NORMAL=5k, HIGH=8k, EXTREME=10k paths.',
    role: 'P(paths above current) → Risk Agent probability',
  },
  {
    name: 'Hidden Markov Model',
    category: 'REGIME',
    color: 'var(--purple)',
    usedBy: 'Technical Agent',
    desc: '3-state HMM (Bull / Bear / Crisis) fitted on daily returns. Viterbi decoding gives current regime. Regime state probability contributes directly.',
    role: 'Regime state probability → Technical Agent factor score',
  },
  {
    name: 'Kalman Filter',
    category: 'STATE EST.',
    color: 'var(--green)',
    usedBy: 'Technical Agent',
    desc: 'Optimal state-space estimator for extracting trend signal from noisy price data. Adaptive gain adjusts to measurement uncertainty.',
    role: 'Filtered trend direction → momentum factor score',
  },
  {
    name: 'Extreme Value Theory (EVT)',
    category: 'TAIL RISK',
    color: 'var(--red)',
    usedBy: 'Risk Agent',
    desc: 'Generalized Pareto Distribution fitted to return tails. Extrapolates 99th percentile VaR/CVaR beyond the historical sample.',
    role: 'VaR99 score → Risk Agent probability adjustment',
  },
  {
    name: 'Kelly Criterion',
    category: 'SIZING',
    color: 'var(--green)',
    usedBy: 'Risk Agent',
    desc: 'Optimal fraction f* = p - (1-p)/b. Vol-adaptive: cap reduces in HIGH/EXTREME GARCH regimes (LOW=20%, NORMAL=15%, HIGH=10%, EXTREME=5%).',
    role: 'Kelly score → position sizing & risk agent factor',
  },
  {
    name: 'Topological Data Analysis',
    category: 'TOPOLOGY',
    color: 'var(--yellow)',
    usedBy: 'Technical Agent',
    desc: 'Persistent homology on sliding price windows. H0 (connectivity) and H1 (cycle) Betti numbers detect market structure not visible in price.',
    role: 'Persistence score → Technical Agent structural factor',
  },
  {
    name: 'Hawkes Process',
    category: 'EVENT RISK',
    color: 'var(--orange)',
    usedBy: 'Volatility Agent',
    desc: 'Self-exciting point process λ(t)=μ+∫α·e^(-β(t-s))dN(s). Detects volatility clustering: bursts of jumps predict more jumps.',
    role: 'Event intensity → Volatility Agent factor score',
  },
  {
    name: 'Momentum Engine',
    category: 'MOMENTUM',
    color: 'var(--cyan)',
    usedBy: 'Technical Agent',
    desc: 'Jegadeesh-Titman 12M-1M cross-sectional momentum + Hurst exponent (long-memory test). Combines into a momentum factor score.',
    role: 'Momentum Z-score → Technical Agent factor score',
  },
  {
    name: 'Quasi-Monte Carlo (Sobol)',
    category: 'SIMULATION',
    color: 'var(--blue)',
    usedBy: 'Risk Agent',
    desc: 'Low-discrepancy Sobol/Halton sequences replace pseudo-random numbers in MC. Faster convergence for the same path count.',
    role: 'More accurate MC estimates → refined probability',
  },
  {
    name: 'Options Intelligence',
    category: 'OPTIONS',
    color: 'var(--purple)',
    usedBy: 'Sentiment Agent',
    desc: 'Implied volatility, put/call ratio, options skew, and gamma exposure from yfinance options chains.',
    role: 'IV/skew signal → Sentiment Agent factor score',
  },
  {
    name: 'Signal Decay & Half-Life',
    category: 'POST-PROCESS',
    color: 'var(--dim2)',
    usedBy: 'Orchestrator — post-fusion scaling',
    desc: 'Exponential decay model: signal strength decays by half-life. Horizon-scaled: shorter horizons use faster decay. Probability pulled toward 0.5 over time.',
    role: 'Scales final P(Up) toward 0.5 by holding horizon',
  },
  {
    name: 'Probability Calibration',
    category: 'CALIBRATION',
    color: 'var(--green)',
    usedBy: 'Orchestrator — final output step',
    desc: 'Isotonic regression / Platt scaling to ensure raw Bayesian posteriors are empirically calibrated (60% confidence → 60% actual hit rate).',
    role: 'Calibrated P(Up) = final output probability',
  },
  {
    name: 'Factor Exposure',
    category: 'FACTORS',
    color: 'var(--yellow)',
    usedBy: 'Fundamental Agent',
    desc: 'Multi-factor alpha/beta decomposition. Separates systematic (market β) exposure from stock-specific alpha across value, quality, growth.',
    role: 'Factor alpha → Fundamental Agent composite score',
  },
  {
    name: 'SABR Volatility Model',
    category: 'VOL SURFACE',
    color: 'var(--orange)',
    usedBy: 'Volatility Agent',
    desc: 'Stochastic Alpha Beta Rho: dF=α·Fᵝ·dW, dα=ν·α·dZ. Calibrates to options vol smile. β adapts to asset class (equity=0.5, rates=0, FX=1).',
    role: 'Vol smile fit quality → Volatility Agent score',
  },
  {
    name: 'Heston Stochastic Vol',
    category: 'STOCH VOL',
    color: 'var(--purple)',
    usedBy: 'Volatility Agent',
    desc: 'Mean-reverting vol: dv=κ(θ-v)dt+σ√v·dW_v. Correlated with price. Captures vol clustering without GARCH assumption of return normality.',
    role: 'Heston params → vol regime confirmation score',
  },
  {
    name: 'Rough Volatility (fBm)',
    category: 'STOCH VOL',
    color: 'var(--orange)',
    usedBy: 'Volatility Agent',
    desc: 'Fractional Brownian motion with Hurst H≈0.1 (rough). Empirically matches volatility roughness in equity & crypto markets.',
    role: 'Hurst H → vol model quality + regime signal',
  },
  {
    name: 'Copula Dependency',
    category: 'CORRELATION',
    color: 'var(--cyan)',
    usedBy: 'Risk Agent (pair/portfolio signals)',
    desc: 'Gaussian + Clayton copulas model non-linear tail co-dependence. Captures correlation breakdown in stress (tail dependence λ).',
    role: 'Tail dependence → Risk Agent correlation factor',
  },
  {
    name: 'Multifractal (MF-DFA)',
    category: 'FRACTAL',
    color: 'var(--yellow)',
    usedBy: 'Technical Agent',
    desc: 'Multi-fractal detrended fluctuation analysis. Decomposes scaling exponents across moments q to detect anomalous long-memory.',
    role: 'Fractal spectrum width → Technical Agent anomaly factor',
  },
  {
    name: 'Granger Causality',
    category: 'CAUSALITY',
    color: 'var(--green)',
    usedBy: 'Macro Agent',
    desc: 'Tests if macro series X (yield curve, M2, DXY) Granger-causes asset returns. VAR-based F-test at p<0.05.',
    role: 'Causal predictors identified → Macro Agent factor boost',
  },
  {
    name: 'Causal Engine (DAG)',
    category: 'CAUSALITY',
    color: 'var(--blue)',
    usedBy: 'Macro Agent',
    desc: 'Directed Acyclic Graph causal model. Do-calculus intervention estimates: P(Return|do(VIX=x)) vs P(Return|VIX=x).',
    role: 'Causal intervention effect → Macro Agent signal',
  },
  {
    name: 'Limit Order Book',
    category: 'MICROSTRUCTURE',
    color: 'var(--red)',
    usedBy: 'Technical Agent',
    desc: 'Bid-ask spread, depth imbalance, price impact estimation. Order flow toxicity proxy from L2 data.',
    role: 'Order imbalance → Technical Agent microstructure factor',
  },
  {
    name: 'Quantum Finance (QAOA)',
    category: 'QUANTUM',
    color: 'var(--purple)',
    usedBy: 'Portfolio optimizer',
    desc: 'Quantum Approximate Optimization Algorithm for portfolio construction. Uses amplitude estimation for risk-return optimization.',
    role: 'Optimal weights → portfolio-level signal adjustment',
  },
  {
    name: 'Fisher Equation / Real Rate',
    category: 'MACRO',
    color: 'var(--yellow)',
    usedBy: 'Macro Agent',
    desc: 'Real Interest Rate = Nominal (DGS10) − Breakeven Inflation (T10YIE). High real rates suppress equity multiples; negative real rates fuel risk-asset expansion.',
    role: 'Real rate level → Macro Agent probability adjustment',
  },
  {
    name: 'Fama-French 5F + Quality Stack',
    category: 'FACTORS',
    color: 'var(--green)',
    usedBy: 'Fundamental Agent',
    desc: 'Fama-French 5-Factor (Mkt, SMB, HML, RMW, CMA) + Piotroski F-Score (9pt), Altman Z-Score (bankruptcy), Beneish M-Score (manipulation). Isolates idiosyncratic alpha from factor beta.',
    role: 'Quality score + factor alpha → Fundamental Agent composite',
  },
  {
    name: 'Contagion Correlation',
    category: 'SYSTEMIC RISK',
    color: 'var(--red)',
    usedBy: 'Macro Agent',
    desc: 'Rolling 20-day avg pairwise correlation across SPY/TLT/GLD/HYG vs full-history baseline. Spike >0.20 signals cross-asset contagion and systemic risk-off de-risking.',
    role: 'Correlation spike → Macro Agent bearish adjustment',
  },
  {
    name: 'Herfindahl-Hirschman Index',
    category: 'CONCENTRATION',
    color: 'var(--cyan)',
    usedBy: 'Insider Agent',
    desc: 'HHI = Σ(shareholding_fraction²) for top-10 institutional holders. High HHI signals concentrated ownership — coordinated selling risk. Moderate HHI = stable institutional base.',
    role: 'Ownership concentration → Insider Agent factor score',
  },
  {
    name: 'FINRA Short Volume Proxy',
    category: 'DARK POOL',
    color: 'var(--dim2)',
    usedBy: 'Insider Agent',
    desc: 'FINRA RegSHO weekly short sale volume as % of total volume. ≥55% = bearish conviction; 45–55% = neutral; <45% = light short covering. Parsed from public FINRA weekly CSV.',
    role: 'Short vol % → Insider Agent bearish/bullish factor',
  },
  {
    name: 'Sector Relative Strength',
    category: 'MOMENTUM',
    color: 'var(--yellow)',
    usedBy: 'Macro Agent',
    desc: "Ticker's SPDR sector ETF (XLK, XLF, XLE…) 22-day return vs SPY. RS > 1.0 signals sector leadership — macro tailwind when a stock's sector outperforms the broad market.",
    role: 'Sector RS vs SPY → Macro Agent factor score',
  },
  {
    name: 'Source Credibility + LLM Alignment',
    category: 'NLP / AI',
    color: 'var(--purple)',
    usedBy: 'Sentiment Agent',
    desc: 'Two-layer NLP: (1) Tier-weighted publisher credibility (WSJ/Bloomberg=1.0 → Benzinga=0.55). (2) Gemini LLM dual-pass — compares headline bullish score vs body bearish score. Flags spin.',
    role: 'Credibility-adjusted NLP + LLM alignment → Sentiment score',
  },
  {
    name: 'Activist Smart Money (13D)',
    category: 'INSIDER',
    color: 'var(--dim2)',
    usedBy: 'Insider Agent',
    desc: 'Scans SEC EDGAR full-text search for SC 13D / SC 13G filings mentioning the ticker in last 90 days. Activist entry (13D) signals intent to influence management — historically a bullish catalyst.',
    role: 'Activist filings count → Insider Agent bullish factor',
  },
  {
    name: 'Geopolitical Signal Stack',
    category: 'EVENT RISK',
    color: '#f87171',
    usedBy: 'Geopolitical Agent',
    desc: 'Three-layer keyword NLP on ticker news: (1) Active Conflict (war/invasion/missile), (2) Sanctions Risk (embargo/OFAC/blacklist), (3) Tariff+Regulatory Risk (antitrust/FTC/DOJ). Each reduces P(Up) on trigger.',
    role: 'Keyword hits → Geopolitical Agent bearish adjustments',
  },
  {
    name: 'Information Theory Stack',
    category: 'INFORMATION',
    color: 'var(--blue)',
    usedBy: 'Sentiment Agent, Orchestrator',
    desc: 'Shannon entropy measures agent signal diversity (high = disagreement = wider CI). Transfer entropy tests if news Granger-causes price. KL divergence flags return distribution regime shifts.',
    role: 'Entropy + causality → signal quality scaling in Bayesian fusion',
  },
];

export default function QuantPanel() {
  const [tab, setTab] = useState('agents');
  const [expandedAgent, setExpandedAgent] = useState(null);

  const TABS = [
    { id: 'agents', label: `🤖 Agents (${AGENTS.length})` },
    { id: 'quant',  label: `⚙️ Probability Engine (${QUANT_THEORIES.length})` },
  ];

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="card-header">
        <span className="card-title">🔬 Quant Theory Index</span>
        <span style={{ fontSize: 10, color: 'var(--dim)' }}>{AGENTS.length} agents · {QUANT_THEORIES.length} quant theories · 226+ factors</span>
      </div>

      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--border)' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 14px', fontSize: 11, border: 'none', cursor: 'pointer',
            background: 'transparent', whiteSpace: 'nowrap',
            color: tab === t.id ? 'var(--text)' : 'var(--dim)',
            fontWeight: tab === t.id ? 700 : 400,
            borderBottom: tab === t.id ? '2px solid var(--blue)' : '2px solid transparent',
          }}>{t.label}</button>
        ))}
      </div>

      <div className="card-body" style={{ padding: '14px 16px' }}>

        {/* ── AGENTS TAB ── */}
        {tab === 'agents' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {AGENTS.map(a => (
              <div key={a.name} style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>
                <div
                  onClick={() => setExpandedAgent(expandedAgent === a.name ? null : a.name)}
                  style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 12px', cursor: 'pointer', background: 'var(--card2)' }}
                >
                  <span style={{ fontSize: 9, color: 'var(--dim)' }}>{expandedAgent === a.name ? '▼' : '▶'}</span>
                  <span style={{ width: 4, height: 16, borderRadius: 2, background: a.color, flexShrink: 0 }} />
                  <strong style={{ fontSize: 12, color: 'var(--text)', flexShrink: 0 }}>{a.name}</strong>
                  <span style={{ fontSize: 10, color: 'var(--dim2)', flex: 1 }}>{a.theory}</span>
                  <span style={{ fontSize: 9, color: 'var(--dim)', background: 'var(--surface)', padding: '2px 7px', borderRadius: 4, flexShrink: 0 }}>
                    {a.factors.length} factors
                  </span>
                </div>
                {expandedAgent === a.name && (
                  <div style={{ padding: '10px 14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '4px 16px' }}>
                    {a.factors.map(f => (
                      <div key={f} style={{ fontSize: 10, color: 'var(--dim2)', display: 'flex', alignItems: 'flex-start', gap: 5 }}>
                        <span style={{ color: a.color, flexShrink: 0, marginTop: 1 }}>·</span>
                        {f}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* ── PROBABILITY ENGINE TAB ── */}
        {tab === 'quant' && (
          <>
            <div style={{ fontSize: 10, color: 'var(--dim)', marginBottom: 12, padding: '7px 10px', background: 'var(--card2)', borderRadius: 6, borderLeft: '3px solid var(--blue)' }}>
              These are the mathematical theories powering the probability calculation. Each feeds directly or indirectly into the final P(Up) via Bayesian fusion.
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 8 }}>
              {QUANT_THEORIES.map(m => (
                <div key={m.name} style={{ background: 'var(--card2)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', borderLeft: `3px solid ${m.color}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
                    <strong style={{ fontSize: 11, color: 'var(--text)' }}>{m.name}</strong>
                    <span style={{ fontSize: 8, fontWeight: 700, color: m.color, background: `${m.color}18`, border: `1px solid ${m.color}30`, padding: '1px 6px', borderRadius: 4, flexShrink: 0, marginLeft: 6 }}>{m.category}</span>
                  </div>
                  <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 5 }}>📍 {m.usedBy}</div>
                  <div style={{ fontSize: 10, color: 'var(--dim2)', lineHeight: 1.55, marginBottom: 6 }}>{m.desc}</div>
                  <div style={{ fontSize: 9, padding: '4px 7px', background: `${m.color}0d`, borderRadius: 4, border: `1px solid ${m.color}20`, color: m.color }}>
                    → {m.role}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
