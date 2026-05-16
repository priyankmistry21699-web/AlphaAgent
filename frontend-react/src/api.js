const BASE = '';

async function req(path, opts = {}) {
  const res = await fetch(BASE + path, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  signal: (ticker, horizon = '1m') => req(`/api/v1/signal/${ticker}?horizon=${horizon}`),
  chat: (question, ticker, signal_context) =>
    req('/api/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, ticker, signal_context }),
    }),
  marketChat: (message, history) =>
    req('/api/v1/market-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history }),
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
  leaderboard: () => req('/api/v1/leaderboard?limit=20'),
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
  status: () => req('/api/status'),
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
