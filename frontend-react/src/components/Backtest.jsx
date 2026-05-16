import { useState } from 'react';
import { api } from '../api.js';

const DEFAULT_FORM = {
  ticker: 'AAPL', start_date: '2022-01-01', end_date: '2024-12-31',
  initial_capital: 100000, commission: 0.001, horizon: '1m',
};

export default function BacktestTab({ isActive }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function runBacktest() {
    setLoading(true); setError(''); setResults(null);
    try {
      const d = await api.backtest(form);
      setResults(d);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const r = results?.metrics || results || {};
  const totalReturn = r.total_return ?? r.return ?? 0;
  const returnColor = totalReturn >= 0 ? 'var(--green)' : 'var(--red)';

  return (
    <div style={{ maxWidth: 1000, margin: '0 auto' }}>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><span className="card-title">🔬 Backtest Configuration</span></div>
        <div className="card-body">
          <div className="bt-form">
            {[
              { key: 'ticker', label: 'Ticker', type: 'text', placeholder: 'AAPL' },
              { key: 'start_date', label: 'Start Date', type: 'date' },
              { key: 'end_date', label: 'End Date', type: 'date' },
              { key: 'initial_capital', label: 'Capital ($)', type: 'number' },
              { key: 'commission', label: 'Commission', type: 'number' },
            ].map(f => (
              <div key={f.key} className="bt-field">
                <label className="bt-label">{f.label}</label>
                <input
                  className="bt-input"
                  type={f.type}
                  value={form[f.key]}
                  placeholder={f.placeholder}
                  onChange={e => setForm(prev => ({ ...prev, [f.key]: e.target.value }))}
                  style={{ width: f.key === 'ticker' ? 100 : f.key.includes('date') ? 140 : 120 }}
                />
              </div>
            ))}
            <div className="bt-field">
              <label className="bt-label">Horizon</label>
              <select className="bt-input" value={form.horizon} onChange={e => setForm(prev => ({ ...prev, horizon: e.target.value }))}>
                {['1d','1w','1m','3m','6m','1y'].map(h => <option key={h} value={h}>{h.toUpperCase()}</option>)}
              </select>
            </div>
            <button className="btn btn-primary" onClick={runBacktest} disabled={loading} style={{ alignSelf: 'flex-end' }}>
              {loading ? '⏳ Running…' : '▶ Run Backtest'}
            </button>
          </div>
          {error && <div className="warn-banner" style={{ marginTop: 10 }}>⚠ {error}</div>}
        </div>
      </div>

      {loading && <div className="spinner-wrap"><div className="spinner" /><div className="spinner-text">Running backtest…</div></div>}

      {results && !loading && (
        <div className="fade-in">
          <div className="bt-results">
            {[
              { label: 'Total Return', val: `${totalReturn >= 0 ? '+' : ''}${(totalReturn * 100).toFixed(2)}%`, color: returnColor },
              { label: 'Sharpe Ratio', val: (r.sharpe_ratio ?? r.sharpe ?? 0).toFixed(3), color: 'var(--cyan)' },
              { label: 'Max Drawdown', val: `${((r.max_drawdown ?? 0) * 100).toFixed(2)}%`, color: 'var(--red)' },
              { label: 'Win Rate', val: `${((r.win_rate ?? 0) * 100).toFixed(1)}%`, color: 'var(--green)' },
              { label: 'Total Trades', val: r.total_trades ?? '—', color: 'var(--blue)' },
              { label: 'Profit Factor', val: (r.profit_factor ?? 0).toFixed(2), color: 'var(--purple)' },
              { label: 'Sortino Ratio', val: (r.sortino_ratio ?? r.sortino ?? 0).toFixed(3), color: 'var(--cyan)' },
              { label: 'Calmar Ratio', val: (r.calmar_ratio ?? r.calmar ?? 0).toFixed(3), color: 'var(--orange)' },
            ].map(m => (
              <div key={m.label} className="bt-metric">
                <div className="bt-metric-lbl">{m.label}</div>
                <div className="bt-metric-val" style={{ color: m.color }}>{m.val}</div>
              </div>
            ))}
          </div>

          {results.equity_curve && results.equity_curve.length > 0 && (
            <div className="card">
              <div className="card-header"><span className="card-title">Equity Curve</span></div>
              <div className="card-body">
                <EquityCurve data={results.equity_curve} />
              </div>
            </div>
          )}

          {results.trades && results.trades.length > 0 && (
            <div className="card" style={{ marginTop: 16 }}>
              <div className="card-header"><span className="card-title">Trades ({results.trades.length})</span></div>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr><th>Date</th><th>Action</th><th>Price</th><th>Shares</th><th>P&L</th></tr>
                  </thead>
                  <tbody>
                    {results.trades.slice(0, 30).map((t, i) => {
                      const pnl = t.pnl ?? 0;
                      const color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
                      return (
                        <tr key={i}>
                          <td style={{ fontSize: 11 }}>{t.date}</td>
                          <td><span className={`badge ${t.action === 'BUY' ? 'badge-long' : t.action === 'SELL' ? 'badge-short' : 'badge-hold'}`}>{t.action}</span></td>
                          <td style={{ fontFamily: 'monospace' }}>${(t.price ?? 0).toFixed(2)}</td>
                          <td>{t.shares ?? '—'}</td>
                          <td style={{ color, fontWeight: 700, fontFamily: 'monospace' }}>{pnl !== 0 ? `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(0)}` : '—'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function EquityCurve({ data }) {
  if (!window.Chart) return <div style={{ color: 'var(--dim)', fontSize: 11 }}>Chart.js not loaded</div>;
  return (
    <div ref={el => {
      if (!el || !window.Chart) return;
      const existing = Chart.getChart(el.querySelector('canvas'));
      if (existing) existing.destroy();
      const canvas = el.querySelector('canvas') || el.appendChild(document.createElement('canvas'));
      canvas.style.maxHeight = '280px';
      new window.Chart(canvas, {
        type: 'line',
        data: {
          labels: data.map((_, i) => i),
          datasets: [{
            data: data,
            borderColor: '#3d9eff',
            backgroundColor: 'rgba(61,158,255,.05)',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.3,
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { display: false },
            y: { ticks: { color: '#5d7ba0', font: { size: 10 } }, grid: { color: '#1c2f4a' } },
          }
        }
      });
    }} style={{ width: '100%' }}>
      <canvas />
    </div>
  );
}
