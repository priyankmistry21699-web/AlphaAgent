const BASE = '';

async function req(path, opts = {}) {
  const res = await fetch(BASE + path, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  signal: (ticker, horizon = '1m', abortSignal) =>
    req(`/api/v1/signal/${ticker}?horizon=${horizon}`, abortSignal ? { signal: abortSignal } : {}),
  chat: (question, ticker, signal_context) =>
    req('/api/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, ticker, signal_context }),
    }),
  marketChat: (message, history, marketContext) =>
    req('/api/v1/market-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history, market_context: marketContext || null }),
    }),
  portfolio: () => req('/api/v1/portfolio/summary'),
  positions: () => req('/api/v1/portfolio/positions'),
  trades: () => req('/api/v1/portfolio/trades?limit=50'),
  paperSummary: () => req('/api/v1/paper/summary'),
  paperPositions: () => req('/api/v1/paper/positions'),
  paperTrades: () => req('/api/v1/paper/trades?limit=20'),
  paperSignal: (ticker, horizon) =>
    req('/api/v1/paper/signal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, horizon: horizon || '1m' }),
    }),
  backtest: (body) =>
    req('/api/v1/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  walkforward: (body) =>
    req('/api/v1/backtest/walk-forward', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  stress: (body) =>
    req('/api/v1/stress-test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  scenarios: () => req('/api/v1/scenarios'),
  leaderboard: (days = 90) => req(`/api/v1/leaderboard?days=${days}`),
  leaderboardSignalRuns: (limit = 100) => req(`/api/v1/leaderboard/signal-runs?limit=${limit}`),
  leaderboardTuneWeights: (days = 90) =>
    req('/api/v1/leaderboard/tune-weights', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ days }),
    }),
  optimize: (body) =>
    req('/api/v1/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  settings: () => req('/api/v1/settings'),
  saveSettings: (body) =>
    req('/api/v1/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  quantMC: (body) =>
    req('/api/v1/quant/monte-carlo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  quantGARCH: (body) =>
    req('/api/v1/quant/garch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  quantHMM: (body) =>
    req('/api/v1/quant/hmm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  quantExplain: (body) =>
    req('/api/v1/quant/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  quantHeston: (ticker) => req(`/api/v1/quant/heston/${ticker}`),
  quantSABR: (ticker) => req(`/api/v1/quant/sabr/${ticker}`),
  quantRoughVol: (ticker) => req(`/api/v1/quant/rough-vol/${ticker}`),
  quantCopula: (tickerA, tickerB) =>
    req('/api/v1/quant/copula', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tickers: [tickerA, tickerB] }),
    }),
  quantMultifractal: (ticker) => req(`/api/v1/quant/multifractal/${ticker}`),
  quantMarkovRegime: (ticker, window = 20, bullThresh = 0.05) =>
    req(`/api/v1/quant/markov-regime/${ticker}?window=${window}&bull_thresh=${bullThresh}&bear_thresh=${bullThresh}`),
  quantGranger: (ticker) => req(`/api/v1/quant/granger/${ticker}`),
  quantCausal: (ticker) => req(`/api/v1/quant/causal/${ticker}`),
  quantLOB: (ticker) => req(`/api/v1/quant/lob/${ticker}`),
  quantQuantum: (ticker) => req(`/api/v1/quant/quantum/${ticker}`),
  holders: (ticker) => req(`/api/v1/ticker/holders/${ticker}`),
  tickerInfo: (ticker) => req(`/api/v1/ticker/info/${ticker}`),
  tickerChart: (ticker, period = '3mo', interval = '1d') =>
    req(`/api/v1/ticker/chart/${ticker}?period=${period}&interval=${interval}`),
  status: () => req('/api/status'),
  marketSummary: () => req('/api/v1/market/summary'),
  marketQuotes: (tickers) =>
    req('/api/v1/market/quotes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tickers }),
    }),
  marketNews: (limit = 40) => req(`/api/v1/market/news?limit=${limit}`),
  marketEarnings: () => req('/api/v1/market/earnings'),
  portfolioChat: (message, history, portfolio, budget, entryDate, exitDatePref) =>
    req('/api/v1/portfolio-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history, portfolio, budget, entry_date: entryDate, exit_date_pref: exitDatePref }),
    }),
  setRegion: (region) => req('/api/v1/market/set-region', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ region }),
  }),
  aiPortfolioCommit: (positions, capital) =>
    req('/api/v1/ai-portfolio/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ positions, capital }),
    }),
  aiPortfolioLive: () => req('/api/v1/ai-portfolio/live'),
  aiPortfolioReset: () => req('/api/v1/ai-portfolio', { method: 'DELETE' }),
  aiPortfolioSell: (ticker) => req(`/api/v1/ai-portfolio/position/${ticker}`, { method: 'DELETE' }),
  aiPortfolioBuyMore: (ticker, amount) => req(`/api/v1/ai-portfolio/position/${ticker}/buy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amount }),
  }),
  warmupStatus: () => req('/api/v1/market/warmup-status'),
  portfolioScan: (body) =>
    req('/api/v1/portfolio-scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  strategyBuild: (body) =>
    req('/api/v1/portfolio/strategy-build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  marketSearch: (q, region = 'all', assetType = 'all', limit = 10) =>
    req(`/api/v1/market/search?q=${encodeURIComponent(q)}&region=${region}&asset_type=${assetType}&limit=${limit}`),
};

export function priceColor(v) {
  return v >= 0.66 ? 'var(--green)' : v <= 0.33 ? 'var(--red)' : 'var(--yellow)';
}
export function dirColor(dir) {
  return dir === 'LONG' ? 'var(--green)' : dir === 'SHORT' ? 'var(--red)' : 'var(--yellow)';
}
export function fmtNum(v, dec = 2) {
  if (v === null || v === undefined) return '—';
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (Math.abs(v) >= 1e3) return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
  if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(2);
  return Number(v).toFixed(dec);
}
export function fmtPct(v, dec = 1) {
  return v === null || v === undefined ? '—' : (v * 100).toFixed(dec) + '%';
}
