import { useState, useEffect } from 'react';
import { api } from '../api.js';

export default function PortfolioTab({ isActive }) {
  const [summary, setSummary] = useState(null);
  const [positions, setPositions] = useState([]);
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeView, setActiveView] = useState('positions');

  useEffect(() => {
    if (isActive) load();
  }, [isActive]);

  async function load() {
    setLoading(true);
    try {
      const [s, p, t] = await Promise.allSettled([api.portfolio(), api.positions(), api.trades()]);
      if (s.status === 'fulfilled') setSummary(s.value);
      if (p.status === 'fulfilled') setPositions(Array.isArray(p.value) ? p.value : p.value.positions || []);
      if (t.status === 'fulfilled') setTrades(Array.isArray(t.value) ? t.value : t.value.trades || []);
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <div className="spinner-wrap"><div className="spinner" /><div className="spinner-text">Loading portfolio…</div></div>;

  const totalPnl = summary?.total_pnl ?? summary?.total_return ?? 0;
  const pnlColor = totalPnl >= 0 ? 'var(--green)' : 'var(--red)';
  const sign = totalPnl >= 0 ? '+' : '';

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto' }}>
      {/* Summary metrics */}
      <div className="metrics-row" style={{ marginBottom: 20 }}>
        {[
          { label: 'Total P&L', val: `${sign}$${Math.abs(totalPnl).toLocaleString('en-US', { maximumFractionDigits: 0 })}`, color: pnlColor },
          { label: 'Portfolio Value', val: `$${(summary?.portfolio_value || 0).toLocaleString('en-US', { maximumFractionDigits: 0 })}`, color: 'var(--text)' },
          { label: 'Return %', val: `${sign}${(summary?.return_pct ?? 0).toFixed(2)}%`, color: pnlColor },
          { label: 'Open Positions', val: positions.length, color: 'var(--cyan)' },
          { label: 'Win Rate', val: `${((summary?.win_rate ?? 0) * 100).toFixed(1)}%`, color: 'var(--blue)' },
          { label: 'Sharpe Ratio', val: (summary?.sharpe ?? 0).toFixed(2), color: 'var(--purple)' },
        ].map(m => (
          <div key={m.label} className="metric-card">
            <div className="metric-lbl">{m.label}</div>
            <div className="metric-val" style={{ color: m.color, fontSize: 20 }}>{m.val}</div>
          </div>
        ))}
      </div>

      {/* View toggle */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {['positions', 'trades'].map(v => (
          <button key={v} className={`btn ${activeView === v ? 'btn-primary' : ''}`} onClick={() => setActiveView(v)}>
            {v === 'positions' ? '📊 Positions' : '📋 Trade History'}
          </button>
        ))}
        <button className="btn btn-sm" onClick={load}>↻ Refresh</button>
      </div>

      {activeView === 'positions' && (
        positions.length === 0 ? (
          <div className="empty-state"><div className="empty-icon">📊</div><div className="empty-text">No open positions. Run signals and add to portfolio.</div></div>
        ) : (
          <div className="portfolio-grid">
            {positions.map((pos, i) => {
              const pnl = pos.unrealized_pnl ?? pos.pnl ?? 0;
              const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
              const sign = pnl >= 0 ? '+' : '';
              return (
                <div key={i} className="position-card">
                  <div className="pos-header">
                    <div>
                      <div className="pos-sym" style={{ color: pos.direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>
                        {pos.ticker} <span style={{ fontSize: 10, color: 'var(--dim)' }}>{pos.direction}</span>
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--dim)' }}>{pos.shares ?? pos.quantity ?? '—'} shares</div>
                    </div>
                    <div>
                      <div className="pos-pnl" style={{ color: pnlColor }}>{sign}${Math.abs(pnl).toFixed(0)}</div>
                      <div style={{ fontSize: 10, color: pnlColor, textAlign: 'right' }}>{sign}{((pos.pnl_pct ?? 0) * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                  <div className="pos-metrics">
                    {[
                      { label: 'Entry', val: `$${(pos.entry_price ?? 0).toFixed(2)}` },
                      { label: 'Current', val: `$${(pos.current_price ?? 0).toFixed(2)}` },
                      { label: 'Market Val', val: `$${(pos.market_value ?? 0).toFixed(0)}` },
                      { label: 'Conviction', val: `${((pos.conviction ?? 0)).toFixed(1)}%` },
                    ].map(m => (
                      <div key={m.label} className="pos-metric">
                        <div className="pos-metric-lbl">{m.label}</div>
                        <div className="pos-metric-val">{m.val}</div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )
      )}

      {activeView === 'trades' && (
        <div className="card">
          <div className="card-header"><span className="card-title">Trade History (Last 50)</span></div>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th><th>Ticker</th><th>Direction</th><th>Shares</th>
                  <th>Entry $</th><th>Exit $</th><th>P&L</th><th>Return %</th>
                </tr>
              </thead>
              <tbody>
                {trades.length === 0 ? (
                  <tr><td colSpan={8} style={{ textAlign: 'center', color: 'var(--dim)', padding: 24 }}>No trades recorded</td></tr>
                ) : trades.slice(0, 50).map((t, i) => {
                  const pnl = t.pnl ?? 0;
                  const color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
                  return (
                    <tr key={i}>
                      <td style={{ fontSize: 11, color: 'var(--dim)' }}>{t.date || t.timestamp || '—'}</td>
                      <td style={{ fontWeight: 700 }}>{t.ticker}</td>
                      <td><span className={`badge ${t.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>{t.direction}</span></td>
                      <td style={{ color: 'var(--dim2)' }}>{t.shares ?? t.quantity ?? '—'}</td>
                      <td style={{ fontFamily: 'JetBrains Mono, monospace' }}>${(t.entry_price ?? 0).toFixed(2)}</td>
                      <td style={{ fontFamily: 'JetBrains Mono, monospace' }}>${(t.exit_price ?? 0).toFixed(2)}</td>
                      <td style={{ color, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace' }}>{pnl >= 0 ? '+' : ''}${pnl.toFixed(0)}</td>
                      <td style={{ color }}>{((t.return_pct ?? 0) * 100).toFixed(2)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
