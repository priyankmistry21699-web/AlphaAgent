# AlphaAgent — Complete Factor Map

> Every factor that feeds into the final probability: **P(stock goes up) = 73%**

> **Updated May 2026 — Tier 1/2/3/4 factors added. See `QUANT_THEORIES_DEEP_DIVE.md` for full implementation log.**

---

## The Pipeline at a Glance

```
RAW DATA (12+ sources)
    ↓
226+ INDIVIDUAL FACTORS (computed by 9 agents) + scoring models
    ↓
9 AGENT PROBABILITIES (each agent → one P(up))
    ↓
BAYESIAN FUSION (correlation-adjusted log-odds combination)
    ↓
DYNAMIC HORIZON REWEIGHTING (1D/1W/1M/3M/6M/1Y)
    ↓
RISK CIRCUIT BREAKER (6-tier override cascade)
    ↓
FINAL OUTPUT: P(up) = 73%, Conviction = HIGH, Kelly Position Size
```

---

## Layer 1: Raw Data Sources

| Source | API | What We Get | Cost |
|---|---|---|---|
| **Price/Volume** | yfinance | OHLCV daily/intraday, 1-5 years history | Free |
| **Options Chain** | yfinance | IV, put/call ratio, open interest, sweeps | Free |
| **Company Financials** | yfinance | Income statement, balance sheet, cash flow | Free |
| **Company Info** | yfinance | Market cap, sector, P/E, dividend, shares | Free |
| **News Articles** | NewsAPI + RSS | Headlines, full text, source, date | Free tier |
| **Macro Data** | FRED API | Fed rate, yield curve, CPI, VIX, DXY, M2 | Free |
| **Market Benchmarks** | yfinance | SPY, QQQ, sector ETFs, global indices | Free |
| **Currency Data** | yfinance / FRED | USD pairs, DXY, EM currencies | Free |
| **Cross-Asset** | yfinance | Bonds, BTC, copper, gold, commodities, SOX | Free |
| **Behavioral/Crowd** | CNN, AAII, Reddit | Fear & Greed, sentiment surveys, social buzz | Free |
| **Alternative Data** | BDI, SEC 8-K | Shipping rates, buybacks, index changes | Free |
| **Geopolitical Data** | NewsAPI + GPR Index | Conflict news, sanctions, trade policy | Free |

### Institutional / Whale Tracking Data Sources (Detailed)

| Source | API / URL | What It Reveals | Cost | Delay |
|---|---|---|---|---|
| **SEC EDGAR 13F** | efts.sec.gov | Exact holdings of Buffett, Citadel, Bridgewater, etc. | Free | 45 days |
| **SEC EDGAR Form 4** | efts.sec.gov | CEO/CFO/Director insider buys and sells | Free | **2 days** |
| **SEC EDGAR 13D** | efts.sec.gov | Activist investors (Icahn, Elliott) taking >5% stake | Free | 10 days |
| **SEC EDGAR 8-K** | efts.sec.gov | Buybacks, M&A, CEO departures, major events | Free | Immediate |
| **FINRA ATS** | otctransparency.finra.org | Dark pool volume per stock | Free | 2 weeks |
| **Quiver Quantitative** | api.quiverquant.com | Congressional trades, lobbying, govt contracts | Free | 45 days |
| **Capitol Trades** | capitoltrades.com | Politician buy/sell with amounts | Free | 45 days |
| **House Stock Watcher** | housestockwatcher.com | House member trades | Free | 45 days |
| **Senate Stock Watcher** | senatestockwatcher.com | Senate member trades | Free | 45 days |
| **OpenInsider** | openinsider.com | Insider transaction aggregator | Free | 2 days |
| **ETF.com** | etf.com | Sector ETF flows | Free | Daily |

> [!TIP]
> **Key institutions trackable via 13F:** Berkshire Hathaway (Buffett), Citadel (Griffin), Bridgewater (Dalio), Renaissance Tech (Simons), BlackRock, Vanguard, Soros Fund, ARK Invest (Wood), Pershing Square (Ackman), Elliott Management, Goldman Sachs, JP Morgan.

### Whale Proxy Detection (from yfinance — FREE, real-time)

When direct filings are delayed, detect whale activity from price/volume footprints:

| Signal | Detection Method | Threshold |
|---|---|---|
| **Volume spike** | Volume / SMA(volume, 20) | > 3× average |
| **Block trades** | Single trade > 10,000 shares or > $200K | Flag as institutional |
| **VWAP absorption** | Price persistently above VWAP all day | Large buyer present |
| **OBV divergence** | Price flat but OBV rising | Quiet accumulation |
| **Options vol/OI ratio** | Options volume / open interest | > 5× = whale bet |
| **Bid-ask imbalance** | Consistent buying pressure on bid | Hidden accumulation |
| **After-hours block prints** | Large prints at VWAP or close price | Institutional crossing |

---

## Layer 2: Factors by Agent

### Agent 1: Technical Agent (33 factors)

**Trend Factors:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 1 | **RSI (14-day)** | 100 - 100/(1 + avg_gain/avg_loss) | Overbought (>70) or oversold (<30) |
| 2 | **MACD Signal** | EMA(12) - EMA(26) vs Signal(9) | Momentum direction + crossovers |
| 3 | **MACD Histogram** | MACD - Signal line | Momentum acceleration |
| 4 | **SMA Crossover (50/200)** | SMA(50) vs SMA(200) | Golden cross / death cross |
| 5 | **EMA Crossover (9/21)** | EMA(9) vs EMA(21) | Short-term trend direction |
| 6 | **Price vs SMA(200)** | Close / SMA(200) | Above = bullish, below = bearish |
| 7 | **ADX (14-day)** | Directional movement index | Trend strength (>25 = trending) |

**Volatility Factors:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 8 | **Bollinger Band %B** | (Close - Lower) / (Upper - Lower) | Position within volatility range |
| 9 | **Bollinger Bandwidth** | (Upper - Lower) / Middle | Volatility expansion/contraction |
| 10 | **ATR (14-day)** | Average True Range | Current volatility magnitude |
| 11 | **GARCH Forecast** | σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1) | Tomorrow's predicted volatility |
| 12 | **Vol Regime** | GARCH σ vs historical percentiles | Low / Normal / High / Extreme |

**Volume Factors:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 13 | **Volume vs 20-day avg** | Today's volume / SMA(volume, 20) | Unusual activity detection |
| 14 | **OBV Trend** | On-Balance Volume direction | Smart money accumulation/distribution |
| 15 | **VWAP Position** | Price vs VWAP | Institutional buying/selling pressure |

**Regime Factors:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 16 | **HMM State** | Hidden Markov Model (Baum-Welch) | Bull / Bear / Sideways regime |
| 17 | **P(Bull)** | HMM forward algorithm | Current regime probability |
| 18 | **P(Regime Change)** | HMM transition matrix | Risk of regime shift |

**Seasonality / Calendar Factors:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 19 | **Day-of-Week Effect** | Historical return by weekday | Monday bearish, Friday bullish |
| 20 | **Month-of-Year Effect** | Historical return by month | January effect, Sell in May |
| 21 | **OpEx Week Flag** | Days to monthly options expiration | Increased vol + gamma pinning |
| 22 | **Quarter-End Window Dressing** | Days to quarter end | Funds buy winners, dump losers |
| 23 | **Earnings Proximity** | Days to next earnings date | Vol spikes 3-5 days before |
| 24 | **Ex-Dividend Proximity** | Days to ex-dividend date | Stock drops by dividend amount |
| 25 | **Turn-of-Month Effect** | Last 2 days + first 3 days of month | Historically positive (payroll flows) |

**Options-Derived Intelligence:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 26 | **IV Skew (25Δ Put - 25Δ Call)** | Difference in put vs call implied vol | Steep put skew = market pricing crash risk |
| 27 | **Gamma Exposure (GEX)** | Net dealer gamma across strikes | Positive GEX = price pinned, Negative = volatile |
| 28 | **Max Pain** | Strike with max open interest (OI-weighted) | Price gravitates here during OpEx week |
| 29 | **Variance Risk Premium** | Implied Vol - Realized Vol (30-day) | Persistently positive = options overpriced |
| 30 | **Implied Correlation** | CBOE Implied Correlation Index | High = systemic risk, low = stock-picking works |

**Multi-Horizon Momentum:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 31 | **12M minus 1M Return** | Classic Jegadeesh-Titman momentum | Long-term winners keep winning (skip recent month) |
| 32 | **Momentum Acceleration** | Change in 3-month momentum vs prior 3-month | Trend strengthening or fading |
| 33 | **Hurst Exponent** | DFA / R-S analysis on rolling window | H>0.5 = trending (ride it), H<0.5 = mean-reverting (fade it) |

**How Technical Agent converts to probability:**
```
Each factor → scored 0 to 100 (0 = very bearish, 100 = very bullish)
  RSI < 30 → score = 80 (oversold = bullish)
  MACD crossover bullish → score = 75
  Price above SMA(200) → score = 65
  Seasonality favorable → score = 58
  IV skew normalizing → score = 62 (fear receding)
  Hurst > 0.55 → score = 70 (trending, ride it)
  ...etc

Weighted average of all 33 scores:
  Tech_Score = Σ(wᵢ × scoreᵢ) / Σ(wᵢ)

Convert to probability:
  P(up | technicals) = sigmoid(Tech_Score - 50) 
  Example: Tech_Score = 68 → P(up) = 0.72
```

---

### Agent 2: Sentiment Agent (17 factors)

**News Sentiment:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 19 | **News Sentiment Score** | LLM rates each article Bullish/Bearish/Neutral | Overall news tone |
| 20 | **Sentiment Momentum** | Change in sentiment over 3 days | Sentiment improving or worsening |
| 21 | **Article Count** | Number of articles in last 24h | Media attention (high = event) |
| 22 | **Source Credibility Weight** | Reuters/Bloomberg > random blog | Quality-weighted sentiment |
| 23 | **Headline vs Body Alignment** | Compare headline sentiment to full text | Clickbait detection |

**Earnings & Analyst:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 24 | **Earnings Surprise** | Actual EPS - Consensus EPS | Beat or miss expectations |
| 25 | **Earnings Call Tone** | LLM analyzes transcript sentiment | Management confidence |
| 26 | **Analyst Revision Direction** | Net upgrades - downgrades (30 days) | Wall Street shifting view |
| 27 | **Price Target vs Current** | Avg analyst target / current price | Upside/downside potential |

**Market Positioning:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 28 | **Short Interest %** | Shares short / float | Bearish positioning (high = squeeze risk) |
| 29 | **Short Interest Change** | Change in short % over 2 weeks | Shorts increasing or covering |
| 30 | **Put/Call Ratio** | Put volume / Call volume | Options market sentiment |

**Behavioral / Crowd Psychology:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 31 | **CNN Fear & Greed Index** | CNN composite (7 indicators) | Extreme fear = contrarian buy, Greed = sell |
| 32 | **AAII Sentiment Survey** | Weekly retail investor bull/bear % | Contrarian indicator (crowd is usually wrong) |
| 33 | **Social Media Buzz Score** | Reddit/X mention volume + sentiment | Viral stocks, meme momentum |
| 34 | **Retail Trading Surge** | Unusual retail order flow | Robinhood herding (often contrarian signal) |
| 35 | **Margin Debt Change** | NYSE margin debt MoM change | Rising = euphoria, falling = forced selling |

**How Sentiment Agent converts to probability:**
```
For each news article:
  LLM → P(Bullish), P(Bearish), P(Neutral)
  
Aggregate across N articles:
  Sentiment_Score = Σ(credibility_wᵢ × sentiment_scoreᵢ) / N

Combine with analyst/positioning/behavioral factors:
  P(up | sentiment) = Bayesian update starting from 0.50
    × news likelihood ratio
    × analyst revision likelihood ratio  
    × short interest likelihood ratio
    × contrarian adjustment (extreme fear/greed)
    × social media momentum factor
    
  Example: positive news + analyst upgrades + extreme fear (contrarian buy)
  → P(up | sentiment) = 0.70
```

---

### Agent 3: Fundamental Agent (30 factors)

**Valuation Factors (already includes P/E):**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 36 | **P/E Ratio** | Price / EPS | Expensive or cheap vs earnings |
| 37 | **P/E vs Sector Median** | Stock P/E / Sector median P/E | Relative value |
| 38 | **P/E vs 5yr Average** | Current P/E / Historical avg P/E | Expensive vs own history |
| 39 | **Forward P/E** | Price / Forward EPS estimate | Market pricing in future growth |
| 40 | **P/B Ratio** | Price / Book value per share | Asset-based valuation |
| 41 | **PEG Ratio** | P/E / Earnings growth rate | Growth-adjusted value |
| 42 | **EV/EBITDA** | Enterprise value / EBITDA | Cash flow valuation |
| 43 | **P/S Ratio** | Price / Revenue per share | Useful for unprofitable growth stocks |
| 44 | **DCF Fair Value** | Discounted cash flow estimate | Intrinsic value estimate |

**Quality/Growth Factors:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 45 | **Revenue Growth (YoY)** | This year revenue / Last year - 1 | Top-line growth |
| 46 | **Earnings Growth (YoY)** | This year EPS / Last year EPS - 1 | Bottom-line growth |
| 47 | **Gross Margin** | Gross profit / Revenue | Pricing power |
| 48 | **Net Margin Trend** | Current vs 4-quarter avg net margin | Profitability improving? |
| 49 | **ROE** | Net income / Shareholder equity | Capital efficiency |
| 50 | **Debt/Equity** | Total debt / Equity | Financial leverage risk |
| 51 | **Free Cash Flow Yield** | FCF / Market cap | Cash generation vs price |
| 52 | **Earnings Consistency** | Std dev of quarterly EPS growth | Predictability |
| 53 | **Piotroski F-Score** | 9-point score (profitability, leverage, efficiency) | F≥7 = strong, F≤2 = weak (proven alpha) |
| 54 | **Altman Z-Score** | 5-ratio model: Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E | Z<1.8 = bankruptcy risk, Z>3 = safe |

**Corporate Events / Alternative Data:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 53 | **Buyback Announcement** | SEC filings for share repurchase programs | Company buying own stock = very bullish |
| 54 | **M&A Activity** | News + SEC filings for merger/acquisition activity | Target gets premium, acquirer may dip |
| 55 | **Index Inclusion/Exclusion** | S&P/Russell reconstitution announcements | Addition = forced buying by index funds |
| 56 | **Lockup Expiration** | Days since IPO (typically 180 days) | Insider selling flood after lockup |
| 57 | **Job Postings Growth** | Indeed/LinkedIn scrape, company careers page | Hiring = expanding = bullish |
| 58 | **Web Traffic Trend** | SimilarWeb / company metrics | Product demand proxy |

**Earnings Quality / Manipulation Detection:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 59 | **Beneish M-Score** | 8-variable model (DSRI, GMI, AQI, SGI, etc.) | M > -1.78 = probable earnings manipulation |
| 60 | **Accruals Ratio** | (Net Income - Operating Cash Flow) / Total Assets | High accruals = low earnings quality |
| 61 | **CEO/CFO Departure** | SEC 8-K filings for executive changes | Sudden departure = red flag |

**Dividend Signals:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 62 | **Dividend Cut Probability** | Payout ratio > 90% + declining FCF + rising debt | Dividend at risk = price will crash |
| 63 | **Dividend Growth Rate (5yr)** | CAGR of dividends over 5 years | Consistent growers outperform (Dividend Aristocrats) |

**How Fundamental Agent converts to probability:**
```
Valuation Z-Score:
  For each metric (P/E, P/B, PEG, EV/EBITDA, P/S, DCF):
    z = (stock_value - sector_median) / sector_stdev
  Combined valuation z-score = weighted avg of all z-scores
  
  z > 0 → undervalued → bullish
  z < 0 → overvalued → bearish

Quality Score: 0-100 based on margins, growth, ROE, FCF

Earnings Quality Check:
  Beneish M-Score > -1.78 → RED FLAG: possible manipulation
  → Override fundamental score to bearish, increase entropy

Corporate Events Adjustment:
  Buyback active + index addition → boost bullish by 5-10%
  Lockup expiring + no buyback → boost bearish by 5-8%
  CEO departure → boost bearish by 3-5%

P(up | fundamentals) = f(valuation_z, quality_score, event_adjustment)
  Undervalued + high quality + buyback + clean M-Score → P(up) = 0.74
  Overvalued + low quality + lockup + M-Score flag → P(up) = 0.25
  Fair valued + medium quality → P(up) = 0.52
```

---

### Agent 4: Macro Agent (31 factors)

**Interest Rate / Fed:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 59 | **Fed Funds Rate** | FRED: current rate | Cost of borrowing |
| 60 | **Rate Change Direction** | Current vs 3 months ago | Tightening or easing |
| 61 | **Yield Curve (10Y-2Y)** | 10yr Treasury - 2yr Treasury | Positive = healthy, Inverted = recession risk |
| 62 | **Real Rate** | Nominal rate - CPI inflation | True borrowing cost |

**Volatility / Fear:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 63 | **VIX Level** | CBOE Volatility Index | Market fear gauge |
| 64 | **VIX Term Structure** | VIX vs VIX3M | Contango = calm, Backwardation = panic |
| 65 | **VIX Change (5-day)** | Current VIX - VIX 5 days ago | Fear increasing or subsiding |

**Economy:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 66 | **CPI (Inflation)** | FRED: latest CPI YoY | Inflation pressure |
| 67 | **Unemployment Rate** | FRED: latest | Labor market strength |
| 68 | **DXY (Dollar Index)** | Dollar strength | Strong dollar = headwind for stocks |
| 69 | **Credit Spreads** | High yield spread vs treasuries | Credit market stress |
| 70 | **SPY Trend** | S&P 500 above/below SMA(200) | Overall market health |

**Liquidity Factors:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 71 | **Fed Balance Sheet Change** | FRED: Fed total assets MoM change | QE = liquidity flood (bullish), QT = drain (bearish) |
| 72 | **M2 Money Supply Change** | FRED: M2 MoM change | More money → bullish, shrinking → bearish |
| 73 | **Repo Market Stress** | SOFR - Fed Funds spread | Spike = funding stress (2019 repo crisis) |
| 74 | **Bid-Ask Spread** | Average bid-ask for the stock | Widening = liquidity drying up = danger |
| 75 | **Amihud Illiquidity** | avg(abs(return) / dollar_volume) | How much price moves per $ traded |

**Cross-Asset / Intermarket Signals:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 76 | **Bond-Equity Correlation** | Rolling 60-day corr(TLT, SPY) | Both falling = systemic stress |
| 77 | **Copper/Gold Ratio** | HG=F / GC=F | Rising = economic optimism, falling = recession fear |
| 78 | **Semiconductor Index (SOX)** | SOX vs SPY relative performance | Leading indicator for tech/economy |
| 79 | **Transport Index (IYT)** | IYT vs DJI divergence | Dow Theory: transports confirm/deny trend |
| 80 | **Bitcoin Correlation** | Rolling 30-day corr(BTC, SPY) | Crypto as risk appetite proxy |
| 81 | **Commodity Index (CRB)** | CRB trend direction | Inflation pressure + demand signal |

**Sector Rotation:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 82 | **Sector Relative Strength** | Stock's sector ETF vs SPY (20-day) | Money rotating into or out of this sector |
| 83 | **Growth vs Value Rotation** | IWF/IWD ratio trend | Risk-on (growth) vs risk-off (value) |
| 84 | **Large vs Small Cap** | SPY/IWM ratio trend | Flight to quality or risk appetite |
| 85 | **Baltic Dry Index (BDI)** | Global shipping rates | Global trade health, supply chain |

**Global Market Contagion:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 86 | **Asian Market Overnight** | Nikkei + Shanghai composite return (before US open) | Overnight risk: Asia crash → US gap down |
| 87 | **European Pre-Market** | STOXX 600 performance (before US session) | Europe selling → US likely follows |
| 88 | **Global Breadth** | % of world indices above their 200-day MA | High = healthy global market, Low = systemic weakness |
| 89 | **Contagion Correlation Spike** | Rolling cross-market correlation vs 6-month avg | Spike = crisis spreading across borders |

**How Macro Agent converts to probability:**
```
Risk-On / Risk-Off Score:
  Each macro factor → categorized as risk-on or risk-off signal
  
  Risk-on signals: Low VIX, easing Fed, positive yield curve, 
                   SPY above SMA(200), low credit spreads,
                   QE active, M2 growing, copper/gold rising,
                   SOX leading, BTC correlated, sector inflow,
                   Asia/Europe green, global breadth > 70%
  Risk-off signals: High VIX, tightening Fed, inverted curve,
                    SPY below SMA(200), widening spreads,
                    QT active, M2 shrinking, repo stress,
                    bond-equity both falling, BDI collapsing,
                    Asia/Europe crashed overnight, contagion spike

  Macro_Score = Σ(weighted_risk_on - weighted_risk_off) / total
  
P(up | macro) = 0.50 + 0.25 × Macro_Score
  Strong risk-on → P(up) = 0.72
  Strong risk-off → P(up) = 0.30
  Neutral → P(up) = 0.50
```

---

### Agent 5: Volatility/Risk Agent (6 factors)

| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 58 | **GARCH 1-day Vol Forecast** | GARCH(1,1) model | Tomorrow's expected volatility |
| 59 | **GARCH 5-day Vol Forecast** | Multi-step GARCH | Week-ahead volatility |
| 60 | **Implied Vol vs Realized** | IV from options / GARCH forecast | Options cheap or expensive |
| 61 | **Vol Regime** | GARCH percentile rank | Low/Normal/High/Extreme |
| 62 | **Correlation with SPY** | Rolling 60-day correlation | Systematic vs idiosyncratic risk |
| 63 | **Beta** | Kalman Filter dynamic beta | Market sensitivity |

---

### Agent 6: Institutional/Whale Agent (17 factors)

> [!WARNING]
> Markets are manipulated by large institutions, hedge funds, and banks. This agent detects their footprints.

**Data sources used:** SEC EDGAR (13F, Form 4, 13D, 8-K), FINRA ATS, Quiver Quant (Congress), yfinance (volume/options proxy detection), OpenInsider

**Dark Pool / Hidden Activity:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 64 | **Dark Pool Volume %** | FINRA dark pool data / total volume | High % = institutions hiding large trades |
| 65 | **Dark Pool Sentiment** | Net dark pool buys vs sells | Smart money direction |
| 66 | **Block Trade Frequency** | Count of trades > 10,000 shares | Large institutional activity |
| 67 | **Unusual Options Activity** | Options volume vs open interest spike | Someone knows something (whale bets) |
| 68 | **Large Options Sweep** | Multi-exchange aggressive fills | Urgency = strong conviction bet |

**Institutional Positioning:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 69 | **13F Ownership Change** | SEC 13F filings (quarterly) | What Buffett/Soros/Citadel are buying/selling |
| 70 | **Institutional Ownership %** | Total institutional shares / float | >80% = tightly held, susceptible to forced selling |
| 71 | **Insider Buy/Sell Ratio** | SEC Form 4 insider transactions | Insiders buying own stock = very bullish signal |
| 72 | **ETF Flow Impact** | Net flows into sector ETFs | Passive money pushing prices up/down |
| 73 | **Short Squeeze Probability** | Short interest × cost to borrow × days to cover | High = squeeze risk (GME-style) |

**Supply / Demand Mechanics:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 74 | **Float Reduction (Buybacks)** | Shares outstanding trend (quarterly) | Shrinking float = bullish supply squeeze |
| 75 | **Share Dilution** | Secondary offerings + stock comp / total shares | Dilution = bearish supply flood |
| 76 | **Activist Filing (13D)** | SEC 13D filings (>5% ownership with intent) | Activist pressure = catalyst incoming |
| 77 | **Top 10 Holder Concentration** | Top 10 institutional holders / float | >60% = thin float, violent moves possible |

**Market Manipulation Detection:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 78 | **Spoofing Detection** | Order book cancel rate > normal | Fake orders to move price |
| 79 | **Wash Trading Score** | Same-price same-volume pattern detection | Artificial volume inflation |
| 80 | **Congressional Trading** | STOCK Act disclosures (politician buy/sell) | Politicians trade on inside knowledge — proven edge |

**How Institutional Agent converts to probability:**
```
Smart Money Direction:
  Dark pool net buying + insider buying + 13F accumulation
  + float shrinking (buybacks) + activist pressure
  → "Smart money is accumulating" → Bullish
  
  Dark pool net selling + insider selling + 13F trimming
  + share dilution + no activist
  → "Smart money is exiting" → Bearish

Manipulation Risk Adjustment:
  High spoofing/wash trading detected?
  → Reduce confidence (increase entropy)
  → "Price action may be artificial — distrust technicals"

P(up | institutional) = f(smart_money_direction, manipulation_risk)
  Example: Insiders buying + dark pool accumulation + no manipulation
  → P(up) = 0.71
```

---

### Agent 7: Geopolitical Agent (11 factors) — NEW

> [!WARNING]
> Wars, sanctions, elections, and trade policies move markets more than any earnings report. Ignoring geopolitics is reckless.

**Conflict & War:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 76 | **GPR Index** | Caldara-Iacoviello Geopolitical Risk Index | Global conflict level (published monthly by Fed) |
| 77 | **GPR Change (30-day)** | Current GPR vs 30 days ago | Geopolitical risk escalating or calming |
| 78 | **Active Conflict Score** | LLM monitors war/conflict news → severity score | Direct impact on defense, oil, supply chains |
| 79 | **Sanctions Risk** | LLM scans for new sanction announcements | Trade disruption for affected companies/countries |

**Trade & Policy:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 80 | **Tariff/Trade War Score** | LLM monitors trade policy news | New tariffs → supply chain disruption |
| 81 | **Election Cycle Phase** | Days to next major election | Policy uncertainty increases near elections |
| 82 | **Regulatory Risk** | LLM scans for regulatory action news | Antitrust, new regulations → sector impact |

**Global Risk Indicators:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 83 | **Oil Price (Brent)** | yfinance: BZ=F | Proxy for geopolitical tension + inflation |
| 84 | **Gold Price** | yfinance: GC=F | Safe haven demand (rising = fear) |
| 85 | **Global PMI** | FRED / Trading Economics | Global manufacturing health |
| 86 | **Emerging Market Stress** | EEM ETF vs SPY relative performance | Capital flight from risk assets |

**How Geopolitical Agent converts to probability:**
```
Geopolitical Risk Score:
  GPR_rising + active_conflicts + new_sanctions + tariff_war
  → High geo risk → Bearish for equities (except defense sector)
  
  GPR_falling + peace_talks + trade_deals
  → Low geo risk → Bullish for equities

Sector-Specific Adjustment:
  War → Bullish for defense (LMT, RTX), Bearish for airlines (DAL)
  Trade war → Bearish for exporters, Bullish for domestic-focused
  Oil spike → Bullish for energy (XOM), Bearish for transport

P(up | geopolitical) = base_risk_score × sector_adjustment
  Example: Low conflict + trade deal progress + stock is domestic-focused
  → P(up) = 0.62
```

---

### Agent 8: Currency/FX Agent (9 factors) — NEW

> [!IMPORTANT]
> Exchange rates directly impact company earnings. A US company earning 40% revenue from Europe gets crushed when EUR/USD drops.

**Dollar Strength:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 87 | **DXY Level** | US Dollar Index | Strong dollar = headwind for US multinationals |
| 88 | **DXY Momentum (20-day)** | DXY change over 20 days | Dollar strengthening or weakening |
| 89 | **DXY vs SMA(50)** | Trend direction | Dollar trend |

**Relevant Currency Pairs:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 90 | **Revenue-Weighted FX Impact** | Company's geographic revenue mix × currency moves | Direct earnings impact from FX |
| 91 | **EUR/USD Change** | Euro vs Dollar (30-day) | European revenue impact |
| 92 | **USD/JPY Change** | Dollar vs Yen (30-day) | Japanese market / carry trade |
| 93 | **USD/CNY Change** | Dollar vs Yuan (30-day) | China trade + supply chain cost |

**Systemic FX Risk:**
| # | Factor | How Computed | What It Tells You |
|---|---|---|---|
| 94 | **EM Currency Stress** | Index of emerging market currency weakness | Capital flight / risk-off signal |
| 95 | **Carry Trade Unwind Risk** | JPY volatility + rate differential change | Yen carry unwind → global sell-off trigger |

**How Currency Agent converts to probability:**
```
FX Impact Score:
  1. Determine stock's geographic revenue breakdown
     AAPL: 40% Americas, 25% Europe, 20% China, 15% rest
     
  2. Compute revenue-weighted FX impact:
     FX_Impact = Σ(revenue_share_i × currency_change_i)
     
     EUR weakening 3% → hurts 25% of AAPL revenue
     CNY weakening 2% → hurts 20% of AAPL revenue
     → Net FX headwind = -1.15% earnings impact
  
  3. Systemic risk check:
     EM currency collapse + carry trade unwind
     → Severe risk-off → bearish for ALL equities

P(up | currency) = f(fx_impact, systemic_risk)
  Example: Dollar weakening + stable EM + stock has 50% intl revenue
  → Weak dollar benefits multinationals → P(up) = 0.64
```

---

## Layer 3: Agent Probabilities

Each agent produces ONE probability:

```
┌───────────────────────────────────────────────────────┐
│   Agent                    │ Factors │ Output         │
│────────────────────────────│─────────│────────────────│
│ 📉 Technical Agent         │ 18      │ P(up) = 0.72   │
│ 📰 Sentiment Agent         │ 12      │ P(up) = 0.65   │
│ 🏦 Fundamental Agent       │ 15      │ P(up) = 0.58   │
│ 🌍 Macro Agent             │ 12      │ P(up) = 0.63   │
│ 📊 Volatility Agent        │  6      │ Vol = 1.8%     │
│ 🐋 Institutional Agent     │ 12      │ P(up) = 0.71   │
│ ⚔️ Geopolitical Agent      │ 11      │ P(up) = 0.62   │
│ 💱 Currency Agent          │  9      │ P(up) = 0.64   │
│────────────────────────────│─────────│────────────────│
│ TOTAL                      │ 95      │ → Bayesian     │
└───────────────────────────────────────────────────────┘
```

---

## Layer 4: Bayesian Fusion → Final Probability

```
Step 1: Start with prior
  P(up) = 0.52 (historical base rate: stocks go up ~52% of days)

Step 2: Update with Technical Agent
  Likelihood ratio: LR_tech = P(tech_signal | up) / P(tech_signal | down)
  If P(up|tech) = 0.72:
    LR_tech = 0.72 / 0.28 = 2.57
  
  Posterior: P(up) = 0.52 × 2.57 / (0.52 × 2.57 + 0.48 × 1) = 0.736

Step 3: Update with Sentiment Agent
  LR_sent = 0.65 / 0.35 = 1.86
  
  BUT: sentiment partially overlaps with technicals (correlation ρ = 0.3)
  Adjusted LR = LR^(1-ρ) = 1.86^0.7 = 1.56
  
  Posterior: P(up) = 0.736 × 1.56 / (0.736 × 1.56 + 0.264 × 1) = 0.813

Step 4: Update with Fundamental Agent
  LR_fund = 0.58 / 0.42 = 1.38
  Correlation adjustment: LR_adj = 1.38^0.85 = 1.32
  
  Posterior: P(up) = 0.849 → ...

Step 5: Update with Macro Agent
  LR_macro = 0.63 / 0.37 = 1.70
  Correlation adjustment: LR_adj = 1.70^0.80 = 1.53

Step 6: Update with Institutional Agent (NEW)
  LR_inst = 0.71 / 0.29 = 2.45
  Correlation with technicals: ρ = 0.4 (volume overlaps)
  Adjusted LR = 2.45^0.6 = 1.65
  
  MANIPULATION CHECK: If spoofing/wash trading detected:
    → Discount Technical Agent weight by 50%
    → "Price action may be artificial"

Step 7: Update with Geopolitical Agent (NEW)
  LR_geo = 0.62 / 0.38 = 1.63
  Low correlation with other agents (ρ = 0.1) → nearly full weight
  Adjusted LR = 1.63^0.9 = 1.56
  
  CRISIS OVERRIDE: If GPR spikes >2σ above mean:
    → Cap max conviction at 60%
    → Increase VaR estimates by 50%
    → "Geopolitical shock — reduce all positions"

Step 8: Update with Currency Agent (NEW)
  LR_fx = 0.64 / 0.36 = 1.78
  Correlation with macro: ρ = 0.5 (FX and rates move together)
  Adjusted LR = 1.78^0.5 = 1.33
  
  CARRY TRADE UNWIND CHECK: If JPY vol spikes + rate diff collapses:
    → Override to BEARISH for all equities
    → "Carry unwind in progress — systemic risk"

Step 9: Regime adjustment
  If HMM says P(Bull) = 0.78:
    Weight posterior toward bullish slightly
  If HMM says P(Crisis) > 0.15:
    Apply skepticism multiplier (cap conviction at 60%)

Step 10: Final calibration
  Apply isotonic regression calibration from backtested accuracy
  Ensure: when we say 73%, stocks actually go up 73% of the time

FINAL: P(up) = 0.732 → "73% conviction LONG"
```

> [!IMPORTANT]
> **Override triggers** — certain events bypass normal Bayesian fusion:
> - **War breaks out** → Force BEARISH, cap conviction at 35%
> - **Carry trade unwind** → Force BEARISH for all equities
> - **Market manipulation detected** → Discount technical signals, increase entropy
> - **Flash crash pattern** → Halt all signals for 30 minutes
> - **Black swan (>5σ move)** → Emergency mode, close all positions

---

## Layer 5: Portfolio Agent (Post-Signal)

> The Portfolio Agent operates AFTER individual stock signals. It manages the **entire book**, not just one ticker.

```
Signal 1 (NVDA: LONG 73%)  ─┐
Signal 2 (AAPL: HOLD 51%)  ─┤
Signal 3 (TSLA: SHORT 68%) ─┼→ 💼 Portfolio Agent → Optimized Execution
Signal 4 (MSFT: LONG 71%)  ─┤
Signal 5 (META: LONG 65%)  ─┘
```

### Portfolio-Level Checks

| Check | Rule | Action |
|---|---|---|
| **Sector Concentration** | Max 30% in any sector | "NVDA LONG approved, but reduce to 4% (tech already at 28%)" |
| **Position Correlation** | Flag holdings with ρ > 0.7 | "NVDA + MSFT corr = 0.82 — trim MSFT before adding NVDA" |
| **Single Position Limit** | Max 10% in any stock | "Cap NVDA at $20K on $200K portfolio" |
| **Portfolio Beta** | Target 0.8-1.2 | "Adding TSLA SHORT reduces beta from 1.15 to 1.02 ✅" |
| **Cash Reserve** | Min 10-15% in cash | "Cash at 12% — sufficient for new position" |
| **Drawdown Check** | Halt if portfolio -10% from peak | "Drawdown -3.2% — within limits ✅" |
| **Sharpe Impact** | Only add if improves risk-adjusted return | "Portfolio Sharpe: 1.45 → 1.52 after adding NVDA ✅" |

### Portfolio Optimization Methods

| Method | When To Use |
|---|---|
| **Mean-Variance (Markowitz)** | Default — maximize Sharpe ratio |
| **Black-Litterman** | Combine market equilibrium with our agent views |
| **Risk Parity** | Equal risk contribution — good for uncertain times |
| **Minimum Variance** | During crisis regime — minimize overall portfolio vol |

### Performance Tracking

| Metric | What It Measures |
|---|---|
| **Sharpe Ratio** | Return per unit of total risk |
| **Sortino Ratio** | Return per unit of downside risk |
| **Max Drawdown** | Worst peak-to-trough decline |
| **Alpha** | Excess return vs SPY benchmark |
| **Win Rate** | % of profitable trades |
| **Profit Factor** | Gross profit / Gross loss |
| **Calmar Ratio** | Annual return / Max drawdown |

### Stress Testing

| Scenario | What It Simulates |
|---|---|
| **2008 GFC Replay** | Apply 2008 correlations + drawdowns to current portfolio |
| **2020 COVID Crash** | Apply March 2020 speed + magnitude to current holdings |
| **Rate Shock (+200bps)** | What if Fed hikes rates 2% suddenly |
| **Sector Collapse** | What if your heaviest sector drops 30% |
| **Black Monday** | Single-day -20% move on all equities |
| **Custom Scenario** | User-defined stress parameters |

### Factor Exposure Tracking

| Factor | What It Tracks |
|---|---|
| **Value Exposure** | Portfolio tilt toward cheap vs expensive stocks |
| **Momentum Exposure** | Portfolio tilt toward winners vs losers |
| **Quality Exposure** | Portfolio tilt toward high vs low quality |
| **Size Exposure** | Portfolio tilt toward large vs small cap |
| **Volatility Exposure** | Portfolio tilt toward high vs low vol |
| **Sector Exposure** | Herfindahl index for concentration risk |

---

## Factor Importance (Approximate Weights)

### Normal Market Conditions
| Agent | Weight | Why |
|---|---|---|
| Technical | **25%** | Most responsive to short-term price action |
| Sentiment | **15%** | News moves prices, but decays fast |
| Fundamental | **15%** | Anchors valuation, but slow-moving |
| Macro | **12%** | Sets the environment, affects all stocks |
| Volatility | **10%** | Doesn't predict direction, but sizes the move |
| Institutional | **10%** | Whale/smart money footprints |
| Geopolitical | **8%** | Baseline geopolitical risk |
| Currency | **5%** | FX impact on earnings |

### Crisis / High-Geopolitical-Risk Conditions
| Agent | Weight | Why |
|---|---|---|
| Technical | **10%** | Technicals break down in crises |
| Sentiment | **10%** | Panic sentiment dominates |
| Fundamental | **10%** | Fundamentals don't matter short-term in a crash |
| Macro | **20%** | Macro environment drives everything |
| Volatility | **15%** | Vol prediction critical for survival |
| Institutional | **15%** | Follow the whales — they know first |
| Geopolitical | **15%** | War/conflict IS the market mover |
| Currency | **5%** | FX dislocations amplify everything |

> [!NOTE]
> These weights are NOT fixed. They dynamically adjust based on:
> - **Regime**: In a crisis, Macro + Geo + Institutional weights spike
> - **Time horizon**: 1-day holds → Technical 40%. 30-day holds → Fundamental 30%
> - **Stock type**: Exporters → Currency weight up. Defense stocks → Geo weight up
> - **Backtested IC**: Agents with higher historical accuracy get more weight
> - **Manipulation detected**: Technical weight drops, Institutional weight spikes

---

## Summary: 151 Factors → 12 Agents → 1 Probability → Optimized Portfolio

```
151 raw factors across 10 data sources
    ↓ (each agent processes its factors)

    📉 Technical Agent         (33 factors) → P(up) = 0.72
       Indicators, regime, seasonality, options intelligence, momentum
    📰 Sentiment Agent         (17 factors) → P(up) = 0.65
       News, earnings, analysts, crowd psychology, social media
    🏦 Fundamental Agent       (28 factors) → P(up) = 0.58
       Valuation (P/E, DCF), quality, events, earnings quality, dividends
    🌍 Macro Agent             (31 factors) → P(up) = 0.63
       Rates, economy, liquidity, cross-asset, sector rotation, global contagion
    📊 Volatility Agent        ( 6 factors) → Vol = 1.8%
       GARCH forecast, IV vs RV, vol regime, beta
    🐋 Institutional Agent     (16 factors) → P(up) = 0.71
       Dark pools, whales, supply/demand, manipulation detection
    ⚔️ Geopolitical Agent      (11 factors) → P(up) = 0.62
       Wars, sanctions, trade policy, oil, gold, EM stress
    💱 Currency Agent          ( 9 factors) → P(up) = 0.64
       DXY, FX impact on earnings, carry trade, EM currencies

    ↓ (Bayesian fusion with correlation adjustment + override checks)
    ⚖️ Debate Agent → P(up) = 73%
    ↓ (risk guardrails + sizing)
    🛡️ Risk Agent → VaR, CVaR, Kelly, stop/target
    ↓ (portfolio-level optimization)
    💼 Portfolio Agent → sector limits, correlation, drawdown, Sharpe impact
    ↓
✅ FINAL: "LONG NVDA, 4% allocation, stop -3%, target +4.5%"
```

### The Complete 12-Agent Architecture

```
🎯 Orchestrator (1)            — Routes queries, manages flow
    ↓
📉📰🏦🌍📊🐋⚔️💱 (8)        — 151 factors → 8 probabilities (parallel)
    ↓
⚖️ Debate Agent (1)            — Bayesian fusion → conviction %
    ↓
🛡️ Risk Agent (1)              — VaR, Kelly, guardrails, stop/target
    ↓
💼 Portfolio Agent (1)          — Optimization, limits, rebalancing
    ↓
✅ Execution Order              — Final sized, approved, risk-managed trade
```

### What Makes This Better Than Any Existing Trading Agent

```
❌ Claude Agent:  "Buy NVDA" (ignores everything below)

✅ AlphaAgent:   "LONG NVDA @ 73% conviction"
                 + P/E undervalued vs sector (z = +0.8) + buyback active
                 + Beneish M-Score clean (-2.4) → no earnings manipulation
                 + IV skew normalizing → options market calming
                 + GEX positive → dealer hedging pins price above $130
                 + Dark pools show institutional accumulation
                 + Float shrinking (buyback reducing supply)
                 + 12M-1M momentum positive, Hurst = 0.58 (trending)
                 + Fed QE active, M2 growing → liquidity supportive
                 + Copper/Gold rising → economic optimism
                 + Asia green overnight, Europe +0.4% pre-market
                 + No active geopolitical threat to semiconductor supply
                 + Weak dollar benefits NVDA's 45% international revenue
                 + CNN Fear & Greed at 28 (extreme fear → contrarian buy)
                 + Dividend growth 15% CAGR, no cut risk
                 + OpEx week → expect elevated volatility
                 + No manipulation detected (spoofing/wash clean)
                 + No activist 13D filing
                 + Portfolio: Tech 28%→32% (under 35% limit ✅)
                 + Portfolio: Sharpe 1.45→1.52 after adding ✅
                 + Portfolio: Drawdown -3.2% (under -10% limit ✅)
                 + BUT: Taiwan Strait tension at elevated (monitor closely)
```


