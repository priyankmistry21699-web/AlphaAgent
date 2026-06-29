export const HZ_LABELS = {
  '1d': '1 Day', '1w': '1 Week', '1m': '1 Month',
  '3m': '3 Months', '6m': '6 Months', '1y': '1 Year',
};

export const TOP_TICKERS = [
  { sym: 'AAPL', name: 'Apple' }, { sym: 'NVDA', name: 'NVIDIA' }, { sym: 'MSFT', name: 'Microsoft' },
  { sym: 'AMZN', name: 'Amazon' }, { sym: 'GOOGL', name: 'Alphabet' }, { sym: 'META', name: 'Meta' },
  { sym: 'TSLA', name: 'Tesla' }, { sym: 'BRK-B', name: 'Berkshire' }, { sym: 'JPM', name: 'JPMorgan' },
  { sym: 'LLY', name: 'Eli Lilly' }, { sym: 'UNH', name: 'UnitedHealth' }, { sym: 'V', name: 'Visa' },
  { sym: 'XOM', name: 'ExxonMobil' }, { sym: 'MA', name: 'Mastercard' }, { sym: 'HD', name: 'Home Depot' },
  { sym: 'PG', name: 'P&G' }, { sym: 'COST', name: 'Costco' }, { sym: 'AVGO', name: 'Broadcom' },
  { sym: 'MRK', name: 'Merck' }, { sym: 'ABBV', name: 'AbbVie' }, { sym: 'AMD', name: 'AMD' },
  { sym: 'NFLX', name: 'Netflix' }, { sym: 'CRM', name: 'Salesforce' }, { sym: 'BAC', name: 'Bank of America' },
  { sym: 'ORCL', name: 'Oracle' }, { sym: 'KO', name: 'Coca-Cola' }, { sym: 'PFE', name: 'Pfizer' },
  { sym: 'DIS', name: 'Disney' }, { sym: 'INTC', name: 'Intel' }, { sym: 'GS', name: 'Goldman Sachs' },
  { sym: 'SPY', name: 'S&P 500 ETF' }, { sym: 'QQQ', name: 'Nasdaq ETF' }, { sym: 'IWM', name: 'Russell 2000' },
  { sym: 'TLT', name: '20Y Treasury' }, { sym: 'GLD', name: 'Gold ETF' }, { sym: 'BTC-USD', name: 'Bitcoin' },
  { sym: 'ETH-USD', name: 'Ethereum' }, { sym: 'COIN', name: 'Coinbase' }, { sym: 'PLTR', name: 'Palantir' },
  { sym: 'ARM', name: 'ARM Holdings' }, { sym: 'SMCI', name: 'Super Micro' }, { sym: 'SNOW', name: 'Snowflake' },
  { sym: 'ZM', name: 'Zoom' }, { sym: 'UBER', name: 'Uber' }, { sym: 'ABNB', name: 'Airbnb' },
  { sym: 'SQ', name: 'Block' }, { sym: 'SHOP', name: 'Shopify' }, { sym: 'MELI', name: 'MercadoLibre' },
  { sym: 'BABA', name: 'Alibaba' },
];

export const AGENT_THEORIES = {
  Technical: {
    color: 'tag-blue',
    tags: ['RSI','MACD','ADX','Bollinger','EMA 9/21','Hurst','HMM','TDA','GEX','IV Skew','Max Pain','VRP','OBV','VWAP','VPIN','0DTE Flow','PCA'],
  },
  Fundamental: {
    color: 'tag-green',
    tags: ['Piotroski','Altman Z','Beneish M','P/E','P/B','FCF Yield','ROE','FF5','CAPM Alpha','Accruals','M&A Activity','PEAD Drift','PCA'],
  },
  Macro: {
    color: 'tag-purple',
    tags: ['VIX','Yield Curve','DXY','ISM','CPI','Amihud','SOFR','Fed BS','SOX','Sector RS','Contagion Corr','Real Rate','PCA'],
  },
  Sentiment: {
    color: 'tag-orange',
    tags: ['NLP News','Short Int','Fear/Greed','P/C Ratio','Transfer Entropy','Shannon Entropy','AAII','PT Upside','Source Cred','HL/Body Align','Earnings Whisper'],
  },
  Insider: {
    color: 'tag-cyan',
    tags: ['Form 4','Cluster Buys','Kyle λ','Float','Inst Ownership','FINRA Short Vol','ETF Flow','Activist 13D','Top-10 HHI'],
  },
  Risk: {
    color: 'tag-red',
    tags: ['Kelly','GARCH','EVT VaR','Quasi-MC','KL Div','Hawkes','Black Swan','Carry','CDS Proxy'],
  },
  Geopolitical: {
    color: 'tag-yellow',
    tags: ['Oil','GPR Index','Election Cycle','Transport','Commodity','EM Stress','Active Conflict','Sanctions','Tariff Risk'],
  },
  Volatility: {
    color: 'tag-orange',
    tags: ['GARCH Regime','IV/RV','VIX Term','VVIX','Skew Index'],
  },
  Currency: {
    color: 'tag-purple',
    tags: ['DXY','EUR/USD','USD/JPY Carry','EM FX','Real IR'],
  },
};

export const FACTOR_ENC = {
  rsi: { fullName: 'RSI (14)', category: 'Momentum', formula: 'RSI = 100 − 100/(1+RS)  |  RS = Avg(14d gains)/Avg(14d losses)', valueUnit: '0–100 oscillator', great: '>65 — strong bullish momentum', good: '50–65 — positive trend', neutral: '~50 — balanced', bad: '30–50 — bearish bias', terrible: '<30 — oversold / capitulation', note: 'Divergence from price is a leading reversal signal. Readings >70 warn of near-term pullback.' },
  macd: { fullName: 'MACD Histogram (12/26/9)', category: 'Momentum / Trend', formula: 'MACD = EMA(12)−EMA(26)  |  Signal = EMA(9) of MACD  |  Histogram = MACD−Signal', valueUnit: '$ price difference', great: 'Positive & widening — accelerating bullish momentum', good: 'Positive histogram', neutral: 'Near zero, flat', bad: 'Negative histogram', terrible: 'Bearish crossover below zero line', note: 'Histogram slope matters more than level. Zero-line crossovers are high-significance signals.' },
  trend: { fullName: 'ADX Trend Strength', category: 'Trend', formula: 'ADX = 14-period smoothed avg of DX  |  DX = |+DI−−DI|/(+DI+−DI)×100', valueUnit: '0–100', great: '>50 with +DI>−DI — very strong trend', good: '25–50 bullish alignment', neutral: '20–25 — developing trend', bad: '<20 — no clear trend (choppy)', terrible: '<20 with −DI>+DI — weak bearish chop', note: 'ADX measures trend STRENGTH not direction. Check +DI vs −DI for direction.' },
  moving_averages: { fullName: 'Golden/Death Cross (50/200 SMA)', category: 'Trend', formula: 'Golden Cross: SMA(50)>SMA(200)  |  Death Cross: SMA(50)<SMA(200)', valueUnit: 'Crossover state', great: 'Recent Golden Cross — major bullish structural shift', good: 'SMA(50)>SMA(200) — bull structure', neutral: 'Lines converging within 1%', bad: 'SMA(50)<SMA(200) — bear structure', terrible: 'Recent Death Cross — major bearish shift', note: 'Lagging but historically reliable for long-term trend changes.' },
  bollinger: { fullName: 'Bollinger %B', category: 'Volatility / Mean Reversion', formula: '%B = (Price−Lower)/(Upper−Lower)  |  Bands = SMA(20)±2σ', valueUnit: '0–1 scale', great: '0.8–1.0 in uptrend — riding upper band', good: '0.5–0.8 above midline', neutral: '~0.5 at SMA(20)', bad: '0.2–0.5 below midline', terrible: '<0.2 near lower band', note: 'Context matters: riding upper band in uptrend = bullish continuation.' },
  volume_obv: { fullName: 'OBV Trend (On-Balance Volume)', category: 'Volume', formula: 'OBV += Volume on up-days  |  OBV −= Volume on down-days', valueUnit: 'Cumulative volume', great: 'OBV rising strongly — institutional accumulation', good: 'OBV uptrend', neutral: 'OBV flat', bad: 'OBV diverging from price (price up, OBV flat)', terrible: 'OBV downtrend — distribution phase', note: 'OBV divergence from price is one of the strongest leading indicators.' },
  vwap: { fullName: 'VWAP Position', category: 'Institutional', formula: 'VWAP = Σ(Price×Volume)/Σ(Volume)', valueUnit: '% above/below VWAP', great: 'Significantly above VWAP — strong institutional buying', good: 'Above VWAP', neutral: 'At VWAP', bad: 'Below VWAP', terrible: 'Far below VWAP — institutional selling', note: 'Critical institutional benchmark. Reclaiming VWAP is bullish.' },
  stochastic: { fullName: 'Stochastic %K (14,3,3)', category: 'Momentum', formula: '%K = (Close−Low₁₄)/(High₁₄−Low₁₄)×100', valueUnit: '0–100 oscillator', great: '>70 & rising in uptrend, or bullish crossover from oversold', good: '>50 above midline', neutral: '~50', bad: '<50 below midline', terrible: '<20 oversold in downtrend', note: 'Most reliable when %K crosses %D (signal line).' },
  range_52w: { fullName: '52-Week Range Position', category: 'Price Strength', formula: 'Position = (Price−52W Low)/(52W High−52W Low)×100', valueUnit: '% of 52W range', great: '>80% near 52W high', good: '60–80%', neutral: '40–60%', bad: '20–40%', terrible: '<20% near 52W low', note: 'Stocks near 52W highs tend to outperform. Near 52W lows often signal ongoing weakness.' },
  atr_momentum: { fullName: 'ATR-Normalized Momentum', category: 'Momentum / Volatility', formula: 'ATR Mom = (Price−Price[N])/ATR(14) — moves in ATR units', valueUnit: 'ATR multiples', great: '>2× ATR positive move', good: '1–2× ATR positive', neutral: '<0.5× ATR', bad: '−1 to −2× ATR', terrible: '<−2× ATR', note: 'Normalizes by volatility. A 2% move on a 5%-ATR stock is weak; same on 1%-ATR is strong.' },
  ema200_trend: { fullName: 'Price vs EMA-200', category: 'Long-Term Trend', formula: 'Signal = (Price−EMA(200))/EMA(200)×100', valueUnit: '% above/below EMA-200', great: '>5% above EMA-200 & rising', good: '0–5% above EMA-200', neutral: 'At EMA-200', bad: '0–5% below EMA-200', terrible: '>5% below EMA-200', note: 'The most-watched institutional trend filter. Below EMA-200 = bear market territory.' },
  volume_surge: { fullName: 'Volume Surge Ratio', category: 'Volume Confirmation', formula: 'Surge = Today Volume / 20-day Avg Volume', valueUnit: 'Ratio (1.0 = avg)', great: '>3× on up-day — institutional conviction', good: '1.5–3× on up-day', neutral: '0.8–1.2×', bad: '<0.8× on up-day (weak participation)', terrible: '>2× on down-day (distribution / panic)', note: 'Volume is the lie detector of price moves. High-volume breakouts are more reliable.' },
  williams_r: { fullName: 'Williams %R (14)', category: 'Momentum', formula: '%R = (High₁₄−Close)/(High₁₄−Low₁₄)×−100', valueUnit: '−100 to 0', great: '−20 to 0 in uptrend', good: '−50 to −20', neutral: '~−50', bad: '−80 to −50', terrible: '−100 to −80 in downtrend', note: 'Similar to Stochastic but more sensitive. Best to confirm RSI signals.' },
  ema_9_21: { fullName: 'EMA 9/21 Crossover', category: 'Short-Term Momentum', formula: 'Bullish: EMA(9)>EMA(21) widening  |  Bearish: EMA(9)<EMA(21)', valueUnit: 'Spread %', great: 'EMA(9) recently crossed above EMA(21)', good: 'EMA(9)>EMA(21) diverging', neutral: 'Lines converging', bad: 'EMA(9)<EMA(21)', terrible: 'EMA(9) recently crossed below EMA(21)', note: 'Fast signal for timing entries within a larger trend.' },
  bb_bandwidth: { fullName: 'Bollinger Bandwidth (Squeeze)', category: 'Volatility', formula: 'BW = (Upper−Lower)/SMA(20)  |  Low = Squeeze (impending breakout)', valueUnit: 'Fraction of SMA', great: 'Post-squeeze expansion with upside breakout', good: 'Moderate expansion', neutral: 'Normal bandwidth', bad: 'Contracting without clear direction', terrible: 'Extreme compression', note: 'Bollinger Squeeze precedes major moves. Direction set by breakout side.' },
  momentum_acceleration: { fullName: 'Momentum Acceleration', category: 'Momentum', formula: 'Accel = ROC(N, today)−ROC(N, yesterday) — is momentum speeding up?', valueUnit: 'Second derivative of price', great: 'Positive & increasing', good: 'Positive', neutral: 'Near zero', bad: 'Decelerating / mildly negative', terrible: 'Strongly negative — momentum collapsing', note: 'Leading indicator: deceleration often precedes reversals while price still rises.' },
  variance_risk_premium: { fullName: 'Variance Risk Premium (VRP)', category: 'Options / Volatility', formula: 'VRP = Implied Vol − Realized Vol', valueUnit: 'Vol points (%)', great: 'High VRP: options expensive, good for selling puts (bullish)', good: 'Moderate VRP 5–10 points', neutral: 'VRP near zero', bad: 'Low/negative VRP', terrible: 'Very negative VRP — extreme realized vol', note: 'High VRP means options sellers collect premium. Low VRP = cheap options, large moves ahead.' },
  hurst: { fullName: 'Hurst Exponent (Long-Memory)', category: 'Statistical', formula: 'H via rescaled range  |  H>0.5: trending  |  H=0.5: random walk  |  H<0.5: mean-reverting', valueUnit: '0–1 exponent', great: '>0.65 — strong persistent trend', good: '0.55–0.65 mild persistence', neutral: '~0.5 random walk', bad: '0.4–0.5 mild mean-reversion', terrible: '<0.4 — strong mean-reversion', note: 'H>0.6 is rare and indicates genuine trend persistence.' },
  momentum_12m_1m: { fullName: '12M-1M Cross-Sectional Momentum', category: 'Momentum', formula: 'Return(12M)−Return(1M)  |  Avoids short-term reversal contamination', valueUnit: '% return', great: '>30% — top momentum decile', good: '15–30%', neutral: '0–15%', bad: '−15 to 0%', terrible: '<−15% — strong downward momentum', note: 'Jegadeesh-Titman (1993). 12M-1M predicts 1–6M forward returns. Skip last month avoids reversal.' },
  hmm_regime: { fullName: 'HMM Market Regime', category: 'Regime Detection', formula: '3-state HMM (Bull/Bear/Volatile) fit to 252d returns + vol via Baum-Welch EM', valueUnit: 'Regime probability', great: 'Bull probability >0.75', good: 'Bull probable 0.55–0.75', neutral: 'Mixed signal', bad: 'Volatile/uncertain regime', terrible: 'Bear probability >0.75', note: 'Captures non-linear regime shifts that moving averages miss.' },
  tda_regime: { fullName: 'TDA Persistent Homology', category: 'Topology / Advanced', formula: 'H₁ persistent homology on price manifold. Betti≠0 indicates market cycles', valueUnit: 'Topological persistence', great: 'Low persistence — smooth monotone uptrend', good: 'Moderate persistence — mild oscillation in uptrend', neutral: 'Average persistence', bad: 'High persistence — choppy market', terrible: 'Extreme topological complexity — regime instability', note: 'TDA captures market structure invisible to linear methods.' },
  iv_skew: { fullName: 'IV Skew (25Δ Put-Call)', category: 'Options / Sentiment', formula: 'Skew = IV(25Δ Put)−IV(25Δ Call)  |  Measures demand for downside protection', valueUnit: 'Vol points', great: 'Falling skew — fear subsiding, bullish', good: 'Normal negative skew', neutral: 'Zero skew', bad: 'High put skew — elevated fear', terrible: 'Extreme put skew >10 vol pts — crash fears', note: 'Rising skew = smart money buying protection. Falling skew after fear = bullish contrarian.' },
  gamma_exposure: { fullName: 'Gamma Exposure (GEX)', category: 'Options / Microstructure', formula: 'GEX = Σ(Gamma × OI × 100 × Spot²) across all strikes', valueUnit: '$ billions', great: 'Strong positive GEX — dealers stabilize market (buy dips, sell rallies)', good: 'Moderate positive GEX', neutral: 'Near zero', bad: 'Negative GEX — dealers amplify moves', terrible: 'Large negative GEX — dealers sell into declines (acceleration)', note: 'The "gamma flip" where GEX crosses zero is a critical level. Below it, moves accelerate.' },
  max_pain: { fullName: 'Options Max Pain Strike', category: 'Options', formula: 'Strike where aggregate options holder losses are maximized (MM profit)', valueUnit: '% from max pain', great: 'Price well above max pain — bullish call momentum', good: 'Price slightly above max pain', neutral: 'Price near max pain (pin risk)', bad: 'Price below max pain — put pressure', terrible: 'Price far below — put buyers in control', note: 'Options market makers hedge dynamically. Price gravitates toward max pain near expiry.' },
  implied_correlation: { fullName: 'Implied Correlation Proxy', category: 'Market Stress', formula: 'Proxy via SPX vs stock IV spread. High correlation = risk-off environment', valueUnit: '0–1 correlation', great: '<0.4 — idiosyncratic behavior, alpha opportunity', good: '0.4–0.55 — moderate correlation', neutral: '~0.5', bad: '0.55–0.7 — high market-driven correlation', terrible: '>0.7 — crisis correlation: everything moves together', note: 'Low correlation = stock-pickers market. Spikes during crises.' },
  pca_signal_quality: { fullName: 'PCA Signal Quality', category: 'Statistical Quality', formula: 'PCA on agent factor scores. Quality = variance explained by PC1', valueUnit: '0–100 score', great: '>75 — high factor coherence', good: '60–75', neutral: '50–60', bad: '40–50 — noisy, factors disagree', terrible: '<40 — very noisy, low confidence', note: 'Low PCA quality = mixed signals, consider reducing position size.' },
  // Fundamental
  piotroski: { fullName: 'Piotroski F-Score (9-pt Health)', category: 'Fundamental Quality', formula: '9 binary signals: Profitability(4) + Leverage(3) + Operating Efficiency(2)', valueUnit: '0–9 score', great: '8–9 — excellent financial health', good: '6–7 — healthy', neutral: '5', bad: '3–4 — weak', terrible: '0–2 — distressed / bankruptcy risk', note: 'Piotroski (2000): top F-score longs beat bottom shorts by 23% annually.' },
  altman: { fullName: 'Altman Z-Score (Bankruptcy)', category: 'Fundamental / Risk', formula: 'Z = 1.2×WC/TA + 1.4×RE/TA + 3.3×EBIT/TA + 0.6×MVE/TL + 1.0×Sales/TA', valueUnit: 'Z-score', great: '>3.0 — safe zone', good: '2.6–3.0 — gray upper', neutral: '1.8–2.6 — gray zone (caution)', bad: '1.1–1.8 — distress zone', terrible: '<1.1 — high bankruptcy probability', note: 'Score <1.8 → 94% historical bankruptcy rate.' },
  beneish: { fullName: 'Beneish M-Score (Earnings Manipulation)', category: 'Earnings Quality', formula: 'M = −4.84 + 0.92×DSRI + 0.528×GMI + 0.404×AQI + ... (8 components)', valueUnit: 'M-score (more negative = cleaner)', great: '<−2.5 — unlikely manipulator', good: '−2.5 to −2.0', neutral: '−2.0 to −1.78', bad: '−1.78 to −1.0', terrible: '>−1.78 — strong manipulation indicator', note: 'Beneish (1999): >−1.78 = 76% sensitivity for accounting fraud detection.' },
  pe_ratio: { fullName: 'P/E Ratio (Trailing)', category: 'Valuation', formula: 'P/E = Market Price / Trailing 12M EPS', valueUnit: 'Multiple (×)', great: '<15× — undervalued vs historical avg', good: '15–20× — fair value', neutral: '20–25×', bad: '25–35× — expensive', terrible: '>35× — bubble risk', note: 'P/E varies by sector. Compare to sector peers and historical range.' },
  forward_pe: { fullName: 'Forward P/E', category: 'Valuation', formula: 'Fwd P/E = Current Price / Consensus NTM EPS Estimate', valueUnit: 'Multiple (×)', great: '<12× vs sector — deep value', good: '12–18×', neutral: '18–22×', bad: '22–30× — stretched expectations', terrible: '>30× — priced for perfection', note: 'Depends on estimate accuracy. Beat-and-raise cycles compress forward P/E.' },
  pb_ratio: { fullName: 'Price-to-Book Ratio', category: 'Valuation', formula: 'P/B = Market Cap / Book Value of Equity', valueUnit: 'Multiple (×)', great: '<1.0× — below book value', good: '1.0–2.0×', neutral: '2.0–3.0×', bad: '3.0–5.0×', terrible: '>5× — intangible-heavy or overvalued', note: 'P/B<1 can signal value OR distress. Best with high ROE.' },
  fcf_yield: { fullName: 'FCF Yield', category: 'Valuation / Quality', formula: 'FCF Yield = (Operating CF − CapEx) / Market Cap', valueUnit: '% yield', great: '>8% — cheap, strong cash generation', good: '5–8%', neutral: '3–5%', bad: '1–3%', terrible: '<0% — burning cash', note: "Buffett's preferred metric. FCF yield > 10Y Treasury yield = stock may beat bonds." },
  roe: { fullName: 'Return on Equity (ROE)', category: 'Profitability', formula: 'ROE = Net Income / Avg Shareholders Equity × 100%', valueUnit: '% annual', great: '>20% — exceptional capital efficiency', good: '15–20%', neutral: '10–15%', bad: '5–10%', terrible: '<5% or negative', note: "Buffett's favorite. Sustainable >20% ROE = durable competitive advantage." },
  accruals_ratio: { fullName: 'Accruals Ratio (Earnings Quality)', category: 'Earnings Quality', formula: 'Accruals = (Net Income − CFO) / Avg Net Assets  |  High = earnings not backed by cash', valueUnit: 'Ratio', great: '<−0.1 — earnings significantly cash-backed', good: '−0.1 to 0.0', neutral: '0.0 to 0.05', bad: '0.05 to 0.1', terrible: '>0.1 — accrual-based earnings, reversal risk', note: 'Richardson et al. (2005): high-accruals firms underperform by 14% annually.' },
  // Macro
  recession_risk: { fullName: 'Recession Risk Probability', category: 'Macro', formula: 'Composite: yield curve inversion + ISM PMI + unemployment claims + LEI index', valueUnit: '0–100 probability', great: '<15% — expansion phase', good: '15–30%', neutral: '30–50%', bad: '50–70%', terrible: '>70% — high recession probability', note: 'Recessions historically = −30%+ S&P 500 drawdowns.' },
  yield_curve: { fullName: 'Yield Curve (10Y−2Y)', category: 'Macro / Fixed Income', formula: 'Spread = 10Y Treasury Yield − 2Y Treasury Yield', valueUnit: 'Basis points (bps)', great: '>100 bps — steep curve, strong growth', good: '50–100 bps — positive, healthy', neutral: '0–50 bps — flat', bad: '−50 to 0 bps — inversion beginning', terrible: '<−50 bps — deeply inverted (recession signal, 12–18M lead)', note: 'Every U.S. recession since 1955 preceded by yield curve inversion.' },
  vix: { fullName: 'VIX (Fear Gauge)', category: 'Sentiment / Volatility', formula: 'VIX = 30-day expected S&P 500 vol from options prices across all strikes', valueUnit: 'Annualized vol %', great: '<15 — complacency, risk-on', good: '15–20 — normal low-vol', neutral: '20–25', bad: '25–35 — elevated fear', terrible: '>35 — extreme fear / crash', note: 'VIX>30 is historically a contrarian BUY signal for long-term investors.' },
  dxy_regime: { fullName: 'DXY Regime', category: 'Macro / Currency', formula: 'DXY trend vs 50d & 200d MAs. Strong dollar = headwind for commodities & EM', valueUnit: 'USD trend score', great: 'Weak/falling dollar — bullish for risk assets & EM', good: 'Mild weakness', neutral: 'Dollar stable', bad: 'Dollar strength — headwind for non-US earnings', terrible: 'Very strong dollar — EM crisis risk, commodity selloff', note: 'Strong USD hurts multinationals and EM allocations.' },
  kl_divergence: { fullName: 'KL Divergence (Return Distribution Shift)', category: 'Risk / Regime', formula: 'KL(P||Q) = Σ P(x)·log(P(x)/Q(x))  |  P = 20d recent returns, Q = 252d baseline', valueUnit: 'Nats (information units)', great: '<0.05 — stable distribution', good: '0.05–0.15 — minor shift', neutral: '0.15–0.30', bad: '0.30–0.60 — significant regime change', terrible: '>0.60 — major distribution shift', note: 'High KL divergence = recent behavior differs materially from history — regime alert.' },
  hawkes_intensity: { fullName: 'Hawkes Branching Ratio', category: 'Microstructure', formula: 'α/β ratio from Ozaki MLE. Measures self-excitation: how much each event triggers more', valueUnit: 'Ratio (0–1)', great: '<0.3 — mostly exogenous shocks', good: '0.3–0.5', neutral: '~0.5', bad: '0.5–0.7 — high clustering', terrible: '>0.7 — near-explosive (flash crash risk zone)', note: 'Branching ratio >0.9 precedes market microstructure instabilities.' },
  quasi_mc_var: { fullName: 'Quasi-MC VaR (Sobol, 4096 paths)', category: 'Risk', formula: 'Sobol low-discrepancy sequences (2¹²=4096 paths) for tail VaR at 95%', valueUnit: '% portfolio loss', great: '<2% VaR — very low tail risk', good: '2–5%', neutral: '5–8%', bad: '8–15%', terrible: '>15% VaR — extreme tail risk', note: 'Sobol sequences fill probability space ~10× more efficiently than pseudorandom.' },
  kelly: { fullName: 'Kelly Fraction (Optimal Position Size)', category: 'Risk / Portfolio', formula: 'f* = (p×b−q)/b  |  p=win prob, b=win/loss ratio (from MC simulation)', valueUnit: 'Fraction of capital', great: '>0.25 — strong edge, significant position', good: '0.15–0.25', neutral: '0.10–0.15', bad: '0.05–0.10', terrible: '<0.05 or negative — no edge', note: 'AlphaAgent uses half-Kelly. Full Kelly is theoretically optimal but too volatile.' },
  news_sentiment: { fullName: 'News Sentiment (NLP)', category: 'Sentiment', formula: 'Aggregates FinBERT scores across headlines. Weighted by source credibility & recency', valueUnit: '−1 to +1', great: '>0.5 — strongly positive news flow', good: '0.2–0.5', neutral: '−0.2 to 0.2', bad: '−0.5 to −0.2', terrible: '<−0.5 — very negative', note: 'News sentiment has ~1–3 day alpha decay.' },
  short_interest: { fullName: 'Short Interest / Float %', category: 'Sentiment / Short Squeeze', formula: 'SI % = Shares Short / Float Shares × 100%', valueUnit: '% of float', great: '<5% — low short, clean technical action', good: '5–10%', neutral: '10–15%', bad: '15–25% — heavy short headwind', terrible: '>25% — extreme (squeeze risk OR fundamental problem)', note: 'High SI is double-edged: headwind or fuel for squeeze on positive catalysts.' },
  insider_sentiment: { fullName: 'Net Insider Sentiment', category: 'Insider Activity', formula: 'Ratio = Insider Buys (value) / (Buys + Sells) from SEC Form 4', valueUnit: 'Ratio 0–1', great: '>0.7 — strong insider buying', good: '0.55–0.7', neutral: '0.4–0.55', bad: '0.25–0.4', terrible: '<0.25 — heavy insider selling', note: 'Open-market purchases are most bullish. 10b5-1 auto-sales less bearish.' },
  kyle_lambda: { fullName: "Kyle's Lambda (Market Impact)", category: 'Microstructure', formula: 'λ = Cov(ΔPrice, Signed Order Flow) / Var(Signed Order Flow)  |  Kyle (1985)', valueUnit: '$/share per $1M order', great: 'Very low λ — deep liquid market', good: 'Low-moderate λ', neutral: 'Average λ', bad: 'High λ — illiquid, large orders move price', terrible: 'Extreme λ — adverse selection risk', note: "Kyle's lambda is the definitive measure of market depth." },
  garch_vol: { fullName: 'GARCH(1,1) Volatility Forecast', category: 'Volatility', formula: 'σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁  |  Typical: α+β≈0.97', valueUnit: 'Annualized vol %', great: 'GARCH<15% — calm, trending', good: '15–25% — normal', neutral: '25–35%', bad: '35–50% — elevated uncertainty', terrible: '>50% — crisis-level vol', note: 'GARCH captures volatility clustering: calm follows calm, volatile follows volatile.' },
  oil_shock: { fullName: 'Oil Shock Impact (Brent Sensitivity)', category: 'Geopolitical / Macro', formula: '30d Brent return × stock-to-oil beta from rolling 60d regression', valueUnit: 'Impact score', great: 'Oil falling, negative oil beta stock (airline/consumer) — tailwind', good: 'Oil stable, low sensitivity', neutral: 'No significant exposure', bad: 'Oil rising, positive oil beta — cost pressure', terrible: 'Extreme oil spike, significant beta', note: 'Brent >$100 historically increases recession probability.' },
  oil_price_direct: { fullName: 'Brent Crude Direct Signal', category: 'Macro / Commodity', formula: 'Brent price trend (30d) vs 50d MA', valueUnit: 'Brent 30d % change', great: 'Oil rallying >10% — energy sector tailwind', good: 'Oil up 3–10%', neutral: 'Oil flat ±3%', bad: 'Oil down 3–10%', terrible: 'Oil crashing >−10%', note: 'WTI/Brent divergence signals supply disruptions.' },
  gpr_index: { fullName: 'GPR Index (Geopolitical Risk)', category: 'Geopolitical', formula: 'Text-based from 11 major newspapers. Counts geopolitical threat/act/risk words. Normalized 100=avg', valueUnit: 'Index (100=avg)', great: '<75 — low tensions, risk-on', good: '75–100', neutral: '100–125', bad: '125–175 — elevated tensions', terrible: '>200 — major geopolitical crisis', note: 'Caldara & Iacoviello (2022). GPR spikes precede equity selloffs and flight to safety.' },
  iv_vs_rv: { fullName: 'IV vs Realized Volatility Ratio', category: 'Volatility', formula: 'Ratio = 30d Implied Vol / 30d Realized Vol  |  >1 = options rich', valueUnit: 'Ratio', great: '<0.85 — options cheap, good to buy calls', good: '0.85–1.0', neutral: '~1.0', bad: '1.0–1.3 — options mildly expensive', terrible: '>1.3 — very expensive, event risk priced in', note: 'High IV/RV ratio = fear premium in options. Mean-reverts after events resolve.' },
  vix_term_structure: { fullName: 'VIX Term Structure (Contango/Backwardation)', category: 'Volatility', formula: 'VX1/VX2 ratio. Contango: VX1<VX2 (normal). Backwardation: VX1>VX2 (stress)', valueUnit: 'Front/back ratio', great: 'Deep contango <0.90 — calm, vol sellers winning', good: 'Normal contango 0.90–0.97', neutral: '~1.0 flat', bad: 'Backwardation beginning >1.0', terrible: '>1.10 backwardation — crisis conditions', note: 'VIX curve backwardation has historically been a short-term buy signal after fear peak.' },
  garch_regime: { fullName: 'GARCH Regime Probability', category: 'Volatility / Regime', formula: 'Regime state from GARCH volatility level thresholds', valueUnit: 'Regime (low/medium/high vol)', great: 'Low-vol regime — trending market', good: 'Transitioning to low-vol', neutral: 'Medium-vol regime', bad: 'High-vol regime — uncertainty', terrible: 'Extreme-vol regime — crisis state', note: 'Regime change from high to low vol is bullish. From low to high is bearish.' },
  // ── New factors (Phase 7+) ──────────────────────────────────────────────
  vpin: { fullName: 'VPIN — Volume-Sync. Prob. of Informed Trading', category: 'Microstructure', formula: 'VPIN = rolling mean |Buys−Sells|/BucketSize over last 50 vol-buckets (Easley, López de Prado & O\'Hara 2012)', valueUnit: '0–1 toxicity ratio', great: '<0.25 — low toxicity, safe liquidity', good: '0.25–0.50 — moderate order flow', neutral: '~0.50', bad: '0.50–0.65 — elevated informed trading', terrible: '>0.65 — HIGH toxicity, informed front-running likely', note: 'VPIN spikes precede volatility cascades. Rising VPIN trend = smart money entering.' },
  zero_dte_flow: { fullName: '0DTE Options Flow (Same-Day Expiry)', category: 'Options / Microstructure', formula: 'Net call OI − put OI (ATM ±5%) + Net dealer GEX = Σ(Gamma×OI×100) calls − puts', valueUnit: 'CP ratio + GEX score', great: 'CALL_DOMINATED: CP>1.3 & OI ratio>0.35 — aggressive call buying, dealers buy stock', good: 'CALL_LEAN: mild call flow', neutral: 'NEUTRAL — balanced put/call', bad: 'PUT_LEAN: mild put flow', terrible: 'PUT_DOMINATED: CP<0.75 & OI ratio<−0.35 — put hedging, dealers sell stock', note: '0DTE options are ~50% of all US equity options volume. Dealer hedging creates self-fulfilling momentum.' },
  pead_drift: { fullName: 'PEAD — Post-Earnings Announcement Drift', category: 'Earnings / Behavioral', formula: 'SUE = (Actual EPS − Estimated EPS) / StdDev(EPS surprise). Drift = sign(SUE) × decay × prob_adj', valueUnit: 'SUE (standardized units)', great: 'SUE>2.0 beat, <20d ago — strong drift continuation', good: 'SUE 0.5–2.0 beat', neutral: 'SUE near 0 or >45d ago', bad: 'SUE −0.5 to −2.0 miss', terrible: 'SUE<−2.0 miss, <20d ago — strong negative drift', note: 'Bernard & Thomas (1989): markets under-react to earnings for 30–60 days. SUE>1 stocks beat by ~4% over 60d.' },
  cds_proxy: { fullName: 'CDS Spread Proxy (HYG/LQD Ratio)', category: 'Credit / Risk', formula: 'Ratio = HYG/LQD price. Credit spread proxy: ratio_1m = (Ratio[t]/Ratio[t−20d] − 1)×100', valueUnit: '1-month % change in HYG/LQD', great: '>+2% — credit tightening, high-yield outperforms IG → bullish equity', good: '+0.5 to +2% — mild credit improvement', neutral: '±0.5% stable', bad: '−1.5 to −3% — credit stress starting', terrible: '<−3% — HYG underperforms LQD, credit spreads widening → bearish equity', note: 'Credit leads equity by days-to-weeks. HYG/LQD deterioration precedes equity selloffs.' },
  earnings_whisper: { fullName: 'Earnings Whisper — Implied vs Historical Move', category: 'Options / Event Risk', formula: 'Ratio = ATM Straddle Price / Current Price ÷ Avg Historical Post-Earnings Move (>4% days)', valueUnit: 'Implied/Historical ratio', great: 'Ratio<0.7 — options cheap vs history (complacency opportunity)', good: '0.7–1.0 — slightly below avg premium', neutral: '1.0–1.2 — normal premium', bad: '1.2–1.5 — above-average event pricing', terrible: '>1.5 — ELEVATED EVENT RISK, market pricing major catalyst', note: 'The "whisper" is what the options market implies beyond consensus. High ratio = event-driven volatility expected.' },
};

export const QUICK_CHIPS = [
  'Buy now?', 'Key risks', 'Why bullish?', 'Agent votes', 'Entry timing', 'Hold period',
];

export const TICKER_CATEGORIES = [
  { id: 'us', label: '🇺🇸 US', sub: [
    { id: 'stocks', label: 'Stocks', tickers: [
      {sym:'AAPL',name:'Apple'},{sym:'NVDA',name:'NVIDIA'},{sym:'MSFT',name:'Microsoft'},
      {sym:'AMZN',name:'Amazon'},{sym:'GOOGL',name:'Alphabet'},{sym:'META',name:'Meta'},
      {sym:'TSLA',name:'Tesla'},{sym:'BRK-B',name:'Berkshire'},{sym:'JPM',name:'JPMorgan'},
      {sym:'LLY',name:'Eli Lilly'},{sym:'UNH',name:'UnitedHealth'},{sym:'V',name:'Visa'},
      {sym:'XOM',name:'ExxonMobil'},{sym:'MA',name:'Mastercard'},{sym:'WMT',name:'Walmart'},
      {sym:'HD',name:'Home Depot'},{sym:'PG',name:'P&G'},{sym:'COST',name:'Costco'},
      {sym:'AVGO',name:'Broadcom'},{sym:'MRK',name:'Merck'},{sym:'ABBV',name:'AbbVie'},
      {sym:'AMD',name:'AMD'},{sym:'NFLX',name:'Netflix'},{sym:'CRM',name:'Salesforce'},
      {sym:'BAC',name:'Bank of America'},{sym:'ORCL',name:'Oracle'},{sym:'KO',name:'Coca-Cola'},
      {sym:'GS',name:'Goldman Sachs'},{sym:'PFE',name:'Pfizer'},{sym:'DIS',name:'Disney'},
      {sym:'INTC',name:'Intel'},{sym:'PLTR',name:'Palantir'},{sym:'COIN',name:'Coinbase'},
      {sym:'SHOP',name:'Shopify'},{sym:'UBER',name:'Uber'},{sym:'ABNB',name:'Airbnb'},
      {sym:'RIVN',name:'Rivian'},{sym:'MELI',name:'MercadoLibre'},{sym:'SNOW',name:'Snowflake'},
    ]},
    { id: 'etfs', label: 'ETFs & Funds', tickers: [
      {sym:'SPY',name:'S&P 500'},{sym:'QQQ',name:'NASDAQ 100'},{sym:'IWM',name:'Russell 2000'},
      {sym:'DIA',name:'Dow Jones'},{sym:'VTI',name:'Total US Mkt'},{sym:'VOO',name:'Vanguard S&P'},
      {sym:'SCHD',name:'Dividend'},{sym:'ARKK',name:'ARK Innovation'},{sym:'VNQ',name:'Real Estate'},
      {sym:'AGG',name:'US Bonds'},{sym:'TLT',name:'20Y Treasury'},{sym:'HYG',name:'High Yield Bonds'},
      {sym:'LQD',name:'IG Bonds'},{sym:'GLD',name:'Gold ETF'},{sym:'SLV',name:'Silver ETF'},
    ]},
    { id: 'sectors', label: 'Sectors', tickers: [
      {sym:'XLK',name:'Technology'},{sym:'XLF',name:'Financials'},{sym:'XLE',name:'Energy'},
      {sym:'XLV',name:'Healthcare'},{sym:'XLI',name:'Industrials'},{sym:'XLB',name:'Materials'},
      {sym:'XLU',name:'Utilities'},{sym:'XLP',name:'Consumer Staples'},{sym:'XLY',name:'Consumer Disc.'},
      {sym:'XLRE',name:'Real Estate'},{sym:'XLC',name:'Comm. Services'},{sym:'VGT',name:'Info Tech'},
    ]},
    { id: 'indices', label: 'Indices', tickers: [
      {sym:'^GSPC',name:'S&P 500'},{sym:'^IXIC',name:'NASDAQ Composite'},
      {sym:'^DJI',name:'Dow Jones'},{sym:'^RUT',name:'Russell 2000'},
      {sym:'^VIX',name:'VIX Fear Index'},
    ]},
  ]},

  { id: 'india', label: '🇮🇳 India', sub: [
    { id: 'stocks', label: 'Stocks', tickers: [
      {sym:'RELIANCE.NS',name:'Reliance Ind.'},{sym:'TCS.NS',name:'TCS'},
      {sym:'HDFCBANK.NS',name:'HDFC Bank'},{sym:'ICICIBANK.NS',name:'ICICI Bank'},
      {sym:'INFY.NS',name:'Infosys'},{sym:'WIPRO.NS',name:'Wipro'},
      {sym:'BAJFINANCE.NS',name:'Bajaj Finance'},{sym:'ITC.NS',name:'ITC'},
      {sym:'HINDUNILVR.NS',name:'HUL'},{sym:'MARUTI.NS',name:'Maruti Suzuki'},
      {sym:'TATAMOTORS.NS',name:'Tata Motors'},{sym:'SBIN.NS',name:'SBI'},
      {sym:'LT.NS',name:'L&T'},{sym:'KOTAKBANK.NS',name:'Kotak Bank'},
      {sym:'HCLTECH.NS',name:'HCL Tech'},{sym:'AXISBANK.NS',name:'Axis Bank'},
      {sym:'TITAN.NS',name:'Titan'},{sym:'ASIANPAINT.NS',name:'Asian Paints'},
      {sym:'SUNPHARMA.NS',name:'Sun Pharma'},{sym:'ADANIENT.NS',name:'Adani Ent.'},
      {sym:'ONGC.NS',name:'ONGC'},{sym:'NTPC.NS',name:'NTPC'},
      {sym:'POWERGRID.NS',name:'Power Grid'},{sym:'DRREDDY.NS',name:"Dr. Reddy's"},
    ]},
    { id: 'etfs', label: 'ETFs & Funds', tickers: [
      {sym:'NIFTYBEES.NS',name:'Nifty BeES'},{sym:'BANKBEES.NS',name:'Bank Nifty BeES'},
      {sym:'GOLDBEES.NS',name:'Gold BeES'},{sym:'JUNIORBEES.NS',name:'Junior BeES'},
      {sym:'ITBEES.NS',name:'IT BeES'},{sym:'INDA',name:'iShares India ETF'},
      {sym:'INDY',name:'Nifty 50 USD ETF'},
    ]},
    { id: 'indices', label: 'Indices', tickers: [
      {sym:'^NSEI',name:'Nifty 50'},{sym:'^BSESN',name:'BSE Sensex'},
    ]},
  ]},

  { id: 'china', label: '🇨🇳 China', sub: [
    { id: 'stocks', label: 'Stocks (ADR)', tickers: [
      {sym:'BABA',name:'Alibaba'},{sym:'TCEHY',name:'Tencent'},
      {sym:'JD',name:'JD.com'},{sym:'PDD',name:'PDD Holdings'},
      {sym:'BIDU',name:'Baidu'},{sym:'NIO',name:'NIO'},
      {sym:'XPEV',name:'Xpeng'},{sym:'LI',name:'Li Auto'},
      {sym:'NTES',name:'NetEase'},{sym:'BILI',name:'Bilibili'},
      {sym:'IQ',name:'iQIYI'},{sym:'TME',name:'Tencent Music'},
      {sym:'VIPS',name:'Vipshop'},{sym:'GRAB',name:'Grab Holdings'},
    ]},
    { id: 'etfs', label: 'ETFs', tickers: [
      {sym:'MCHI',name:'iShares China'},{sym:'FXI',name:'Large-Cap China'},
      {sym:'KWEB',name:'China Internet'},{sym:'CQQQ',name:'China Technology'},
      {sym:'GXC',name:'SPDR China'},
    ]},
    { id: 'indices', label: 'Indices', tickers: [
      {sym:'^HSI',name:'Hang Seng'},{sym:'000001.SS',name:'Shanghai Comp.'},
    ]},
  ]},

  { id: 'japan', label: '🇯🇵 Japan', sub: [
    { id: 'stocks', label: 'Stocks', tickers: [
      {sym:'7203.T',name:'Toyota'},{sym:'6758.T',name:'Sony'},
      {sym:'9984.T',name:'SoftBank'},{sym:'7974.T',name:'Nintendo'},
      {sym:'6861.T',name:'Keyence'},{sym:'8306.T',name:'MUFG'},
      {sym:'9432.T',name:'NTT'},{sym:'6902.T',name:'DENSO'},
      {sym:'4502.T',name:'Takeda Pharma'},{sym:'7267.T',name:'Honda'},
      {sym:'9983.T',name:'Fast Retailing'},{sym:'6501.T',name:'Hitachi'},
      {sym:'8035.T',name:'Tokyo Electron'},{sym:'4063.T',name:'Shin-Etsu'},
    ]},
    { id: 'etfs', label: 'ETFs', tickers: [
      {sym:'EWJ',name:'iShares Japan'},{sym:'DXJ',name:'WisdomTree Japan'},
      {sym:'SCJ',name:'iShares SC Japan'},
    ]},
    { id: 'indices', label: 'Indices', tickers: [
      {sym:'^N225',name:'Nikkei 225'},{sym:'^TOPX',name:'TOPIX'},
    ]},
  ]},

  { id: 'europe', label: '🇪🇺 Europe', sub: [
    { id: 'stocks', label: 'Stocks', tickers: [
      {sym:'ASML',name:'ASML (NL)'},{sym:'SAP',name:'SAP (DE)'},
      {sym:'NVO',name:'Novo Nordisk (DK)'},{sym:'SHEL',name:'Shell (UK)'},
      {sym:'TTE',name:'TotalEnergies (FR)'},{sym:'AZN',name:'AstraZeneca (UK)'},
      {sym:'UL',name:'Unilever (UK)'},{sym:'LVMUY',name:'LVMH (FR)'},
      {sym:'NSRGY',name:'Nestlé (CH)'},{sym:'BP',name:'BP (UK)'},
      {sym:'RIO',name:'Rio Tinto (UK)'},{sym:'HSBC',name:'HSBC (UK)'},
      {sym:'GSK',name:'GSK (UK)'},{sym:'VOD',name:'Vodafone (UK)'},
      {sym:'ORAN',name:'Orange (FR)'},{sym:'SIE.DE',name:'Siemens (DE)'},
    ]},
    { id: 'etfs', label: 'ETFs', tickers: [
      {sym:'EZU',name:'Eurozone'},{sym:'EWG',name:'Germany'},
      {sym:'EWQ',name:'France'},{sym:'EWU',name:'United Kingdom'},
      {sym:'EWI',name:'Italy'},{sym:'EWP',name:'Spain'},
      {sym:'FEZ',name:'EURO STOXX 50'},{sym:'VGK',name:'Europe Vanguard'},
      {sym:'HEDJ',name:'Hedged Europe'},
    ]},
    { id: 'indices', label: 'Indices', tickers: [
      {sym:'^FTSE',name:'FTSE 100 (UK)'},{sym:'^GDAXI',name:'DAX (DE)'},
      {sym:'^FCHI',name:'CAC 40 (FR)'},{sym:'^STOXX50E',name:'Euro Stoxx 50'},
      {sym:'^AEX',name:'AEX (NL)'},
    ]},
  ]},

  { id: 'global', label: '🌍 Global', sub: [
    { id: 'metals', label: 'Metals & Energy', tickers: [
      {sym:'GC=F',name:'Gold Futures'},{sym:'SI=F',name:'Silver Futures'},
      {sym:'PL=F',name:'Platinum'},{sym:'PA=F',name:'Palladium'},
      {sym:'HG=F',name:'Copper'},{sym:'CL=F',name:'Crude Oil (WTI)'},
      {sym:'BZ=F',name:'Brent Crude'},{sym:'NG=F',name:'Natural Gas'},
      {sym:'ZW=F',name:'Wheat'},{sym:'ZC=F',name:'Corn'},
      {sym:'GLD',name:'Gold ETF'},{sym:'SLV',name:'Silver ETF'},
      {sym:'USO',name:'Oil ETF'},{sym:'UNG',name:'Nat Gas ETF'},
      {sym:'CPER',name:'Copper ETF'},{sym:'PDBC',name:'Commodities ETF'},
    ]},
    { id: 'crypto', label: 'Crypto', tickers: [
      {sym:'BTC-USD',name:'Bitcoin'},{sym:'ETH-USD',name:'Ethereum'},
      {sym:'BNB-USD',name:'BNB'},{sym:'SOL-USD',name:'Solana'},
      {sym:'XRP-USD',name:'XRP'},{sym:'ADA-USD',name:'Cardano'},
      {sym:'AVAX-USD',name:'Avalanche'},{sym:'DOGE-USD',name:'Dogecoin'},
      {sym:'DOT-USD',name:'Polkadot'},{sym:'MATIC-USD',name:'Polygon'},
      {sym:'LINK-USD',name:'Chainlink'},{sym:'UNI-USD',name:'Uniswap'},
      {sym:'SHIB-USD',name:'Shiba Inu'},{sym:'LTC-USD',name:'Litecoin'},
      {sym:'ATOM-USD',name:'Cosmos'},{sym:'BCH-USD',name:'Bitcoin Cash'},
    ]},
    { id: 'forex', label: 'Forex', tickers: [
      {sym:'EURUSD=X',name:'EUR/USD'},{sym:'GBPUSD=X',name:'GBP/USD'},
      {sym:'USDJPY=X',name:'USD/JPY'},{sym:'GBPJPY=X',name:'GBP/JPY'},
      {sym:'USDCNY=X',name:'USD/CNY'},{sym:'USDINR=X',name:'USD/INR'},
      {sym:'EURJPY=X',name:'EUR/JPY'},{sym:'AUDUSD=X',name:'AUD/USD'},
      {sym:'USDCHF=X',name:'USD/CHF'},{sym:'NZDUSD=X',name:'NZD/USD'},
      {sym:'USDCAD=X',name:'USD/CAD'},{sym:'EURGBP=X',name:'EUR/GBP'},
      {sym:'EURINR=X',name:'EUR/INR'},{sym:'GBPINR=X',name:'GBP/INR'},
      {sym:'USDBRL=X',name:'USD/BRL'},{sym:'USDMXN=X',name:'USD/MXN'},
    ]},
    { id: 'etfs', label: 'Global ETFs', tickers: [
      {sym:'ACWI',name:'All World'},{sym:'VEU',name:'All World ex-US'},
      {sym:'VXUS',name:'Total Intl'},{sym:'EEM',name:'Emerging Mkts'},
      {sym:'VWO',name:'EM Vanguard'},{sym:'INDA',name:'India'},
      {sym:'EWH',name:'Hong Kong'},{sym:'EWY',name:'South Korea'},
      {sym:'EWT',name:'Taiwan'},{sym:'EWZ',name:'Brazil'},
      {sym:'EWA',name:'Australia'},{sym:'EWC',name:'Canada'},
    ]},
  ]},
];
