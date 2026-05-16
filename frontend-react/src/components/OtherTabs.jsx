import { useState, useEffect } from 'react';
import { api } from '../api.js';

/* ── Paper Trading ──────────────────────────────────────────── */
export function PaperTab({ isActive }) {
  const [summary, setSummary] = useState(null);
  const [positions, setPositions] = useState([]);
  const [ticker, setTicker] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [tradeLoading, setTradeLoading] = useState(false);

  useEffect(() => { if (isActive) load(); }, [isActive]);

  async function load() {
    setLoading(true);
    const [s, p] = await Promise.allSettled([api.paperSummary(), api.paperPositions()]);
    if (s.status === 'fulfilled') setSummary(s.value);
    if (p.status === 'fulfilled') setPositions(Array.isArray(p.value) ? p.value : p.value.positions || []);
    setLoading(false);
  }

  async function paperTrade() {
    if (!ticker.trim()) return;
    setTradeLoading(true);
    try {
      await api.paperSignal(ticker.trim().toUpperCase(), '1m');
      await load();
    } catch (err) {
      alert(err.message);
    } finally {
      setTradeLoading(false);
    }
  }

  const pnl = summary?.total_pnl ?? 0;
  const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <div className="metrics-row" style={{ marginBottom: 20 }}>
        {[
          { label: 'Paper P&L', val: `${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(0)}`, color: pnlColor },
          { label: 'Cash', val: `$${(summary?.cash ?? 0).toLocaleString('en-US', { maximumFractionDigits: 0 })}`, color: 'var(--text)' },
          { label: 'Portfolio Value', val: `$${(summary?.portfolio_value ?? 100000).toLocaleString('en-US', { maximumFractionDigits: 0 })}`, color: 'var(--cyan)' },
          { label: 'Positions', val: positions.length, color: 'var(--blue)' },
        ].map(m => (
          <div key={m.label} className="metric-card">
            <div className="metric-lbl">{m.label}</div>
            <div className="metric-val" style={{ color: m.color, fontSize: 20 }}>{m.val}</div>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><span className="card-title">📄 Paper Trade Signal</span></div>
        <div className="card-body">
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <input
              style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)', padding: '8px 12px', fontSize: 13, outline: 'none', textTransform: 'uppercase', width: 140 }}
              value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              placeholder="AAPL"
            />
            <button className="btn btn-primary" onClick={paperTrade} disabled={tradeLoading}>
              {tradeLoading ? '⏳ Trading…' : '📄 Execute Paper Trade'}
            </button>
            <button className="btn btn-sm" onClick={load}>↻ Refresh</button>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="spinner-wrap"><div className="spinner" /></div>
      ) : positions.length === 0 ? (
        <div className="empty-state"><div className="empty-icon">📄</div><div className="empty-text">No paper positions yet. Enter a ticker and execute a paper trade.</div></div>
      ) : (
        <div className="card">
          <div className="card-header"><span className="card-title">Paper Positions</span></div>
          <table className="data-table">
            <thead><tr><th>Ticker</th><th>Dir</th><th>Shares</th><th>Entry $</th><th>Current $</th><th>P&L</th></tr></thead>
            <tbody>
              {positions.map((p, i) => {
                const pnl = p.unrealized_pnl ?? p.pnl ?? 0;
                const c = pnl >= 0 ? 'var(--green)' : 'var(--red)';
                return (
                  <tr key={i}>
                    <td style={{ fontWeight: 700 }}>{p.ticker}</td>
                    <td><span className={`badge ${p.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>{p.direction}</span></td>
                    <td>{p.shares ?? p.quantity ?? '—'}</td>
                    <td style={{ fontFamily: 'monospace' }}>${(p.entry_price ?? 0).toFixed(2)}</td>
                    <td style={{ fontFamily: 'monospace' }}>${(p.current_price ?? 0).toFixed(2)}</td>
                    <td style={{ color: c, fontWeight: 700, fontFamily: 'monospace' }}>{pnl >= 0 ? '+' : ''}${pnl.toFixed(0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ── Walk-Forward ───────────────────────────────────────────── */
export function WalkFwdTab({ isActive }) {
  const [form, setForm] = useState({ ticker: 'AAPL', start_date: '2021-01-01', end_date: '2024-12-31', train_months: 12, test_months: 3 });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function run() {
    setLoading(true); setError(''); setResults(null);
    try { setResults(await api.walkforward(form)); }
    catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><span className="card-title">🔄 Walk-Forward Optimization</span></div>
        <div className="card-body">
          <div className="bt-form">
            {[
              { key: 'ticker', label: 'Ticker', type: 'text', w: 100 },
              { key: 'start_date', label: 'Start', type: 'date', w: 140 },
              { key: 'end_date', label: 'End', type: 'date', w: 140 },
              { key: 'train_months', label: 'Train Months', type: 'number', w: 100 },
              { key: 'test_months', label: 'Test Months', type: 'number', w: 100 },
            ].map(f => (
              <div key={f.key} className="bt-field">
                <label className="bt-label">{f.label}</label>
                <input className="bt-input" type={f.type} value={form[f.key]} style={{ width: f.w }}
                  onChange={e => setForm(p => ({ ...p, [f.key]: e.target.value }))} />
              </div>
            ))}
            <button className="btn btn-primary" onClick={run} disabled={loading} style={{ alignSelf: 'flex-end' }}>
              {loading ? '⏳' : '▶ Run'}
            </button>
          </div>
          {error && <div className="warn-banner" style={{ marginTop: 10 }}>⚠ {error}</div>}
        </div>
      </div>
      {loading && <div className="spinner-wrap"><div className="spinner" /></div>}
      {results && !loading && (
        <div className="card fade-in">
          <div className="card-header"><span className="card-title">Walk-Forward Results</span></div>
          <div className="card-body">
            <div className="bt-results">
              {Object.entries(results.metrics || results.summary || {}).slice(0, 8).map(([k, v]) => (
                <div key={k} className="bt-metric">
                  <div className="bt-metric-lbl">{k.replace(/_/g, ' ')}</div>
                  <div className="bt-metric-val" style={{ fontSize: 15 }}>{typeof v === 'number' ? v.toFixed(3) : String(v)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Stress Test ────────────────────────────────────────────── */
export function StressTab({ isActive }) {
  const [ticker, setTicker] = useState('AAPL');
  const [results, setResults] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => { if (isActive) api.scenarios().then(s => setScenarios(Array.isArray(s) ? s : s.scenarios || [])).catch(() => {}); }, [isActive]);

  async function run() {
    setLoading(true);
    try { setResults(await api.stress({ ticker: ticker.toUpperCase(), scenarios: scenarios.map(s => s.name || s.scenario_name) })); }
    catch (e) { console.error(e); }
    finally { setLoading(false); }
  }

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><span className="card-title">⚡ Stress Test</span></div>
        <div className="card-body">
          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <div className="bt-field">
              <label className="bt-label">Ticker</label>
              <input className="bt-input" value={ticker} style={{ width: 100 }} onChange={e => setTicker(e.target.value.toUpperCase())} />
            </div>
            <button className="btn btn-primary" onClick={run} disabled={loading}>{loading ? '⏳' : '▶ Run Stress Test'}</button>
          </div>
        </div>
      </div>
      {loading && <div className="spinner-wrap"><div className="spinner" /></div>}
      {results && !loading && (
        <div className="card fade-in">
          <div className="card-header"><span className="card-title">Stress Test Results — {ticker}</span></div>
          <table className="data-table">
            <thead><tr><th>Scenario</th><th>Return Impact</th><th>VaR Impact</th><th>Risk Level</th></tr></thead>
            <tbody>
              {(results.results || results.scenarios || []).map((r, i) => {
                const ret = r.return_impact ?? r.impact ?? 0;
                const color = ret >= 0 ? 'var(--green)' : 'var(--red)';
                return (
                  <tr key={i}>
                    <td style={{ fontWeight: 600 }}>{r.scenario || r.name}</td>
                    <td style={{ color, fontWeight: 700, fontFamily: 'monospace' }}>{ret >= 0 ? '+' : ''}{(ret * 100).toFixed(2)}%</td>
                    <td style={{ fontFamily: 'monospace' }}>{r.var_impact ? `${(r.var_impact * 100).toFixed(2)}%` : '—'}</td>
                    <td><span className={`badge ${r.risk_level === 'HIGH' ? 'badge-short' : r.risk_level === 'LOW' ? 'badge-long' : 'badge-hold'}`}>{r.risk_level || '—'}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ── Quant Lab ──────────────────────────────────────────────── */
export function QuantLabTab({ isActive }) {
  const [ticker, setTicker] = useState('AAPL');
  const [model, setModel] = useState('monte_carlo');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true); setResults(null);
    try {
      const fn = model === 'monte_carlo' ? api.quantMC : model === 'garch' ? api.quantGARCH : api.quantHMM;
      setResults(await fn({ ticker: ticker.toUpperCase(), paths: 1000 }));
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><span className="card-title">🔭 Quant Lab</span></div>
        <div className="card-body">
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <div className="bt-field">
              <label className="bt-label">Ticker</label>
              <input className="bt-input" value={ticker} style={{ width: 100 }} onChange={e => setTicker(e.target.value.toUpperCase())} />
            </div>
            <div className="bt-field">
              <label className="bt-label">Model</label>
              <select className="bt-input" value={model} onChange={e => setModel(e.target.value)}>
                <option value="monte_carlo">Monte Carlo / Quasi-MC</option>
                <option value="garch">GARCH Volatility</option>
                <option value="hmm">HMM Regime Detection</option>
              </select>
            </div>
            <button className="btn btn-primary" onClick={run} disabled={loading}>{loading ? '⏳' : '▶ Run Model'}</button>
          </div>
        </div>
      </div>
      {loading && <div className="spinner-wrap"><div className="spinner" /></div>}
      {results && !loading && (
        <div className="card fade-in">
          <div className="card-header"><span className="card-title">Results — {model.replace('_', ' ').toUpperCase()} — {ticker}</span></div>
          <div className="card-body">
            <div className="bt-results">
              {Object.entries(results).filter(([, v]) => typeof v === 'number').slice(0, 10).map(([k, v]) => (
                <div key={k} className="bt-metric">
                  <div className="bt-metric-lbl">{k.replace(/_/g, ' ')}</div>
                  <div className="bt-metric-val" style={{ fontSize: 15, color: 'var(--cyan)' }}>{v.toFixed(4)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Leaderboard ────────────────────────────────────────────── */
export function LeaderboardTab({ isActive }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isActive) {
      setLoading(true);
      api.leaderboard().then(d => setData(Array.isArray(d) ? d : d.leaderboard || [])).catch(() => {}).finally(() => setLoading(false));
    }
  }, [isActive]);

  if (loading) return <div className="spinner-wrap"><div className="spinner" /></div>;

  return (
    <div style={{ maxWidth: 700, margin: '0 auto' }}>
      <div className="card">
        <div className="card-header"><span className="card-title">🏆 Signal Leaderboard — Top Performers</span></div>
        {data.length === 0 ? (
          <div className="empty-state"><div className="empty-icon">🏆</div><div className="empty-text">No leaderboard data yet. Run signals to build history.</div></div>
        ) : data.map((item, i) => {
          const ret = item.total_return ?? item.return ?? 0;
          const color = ret >= 0 ? 'var(--green)' : 'var(--red)';
          return (
            <div key={i} className="lb-row">
              <div className={`lb-rank ${i === 0 ? 'top1' : i === 1 ? 'top2' : i === 2 ? 'top3' : ''}`}>
                {i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i + 1}`}
              </div>
              <div style={{ flex: 1 }}>
                <div className="lb-sym">{item.ticker}</div>
                <div style={{ fontSize: 10, color: 'var(--dim)' }}>{item.direction} · {item.horizon}</div>
              </div>
              <div className="lb-stat" style={{ color }}>{ret >= 0 ? '+' : ''}{(ret * 100).toFixed(2)}%</div>
              <div className="lb-stat" style={{ color: 'var(--cyan)' }}>{(item.conviction ?? 0).toFixed(1)}%</div>
              <div className="lb-stat" style={{ color: 'var(--dim)' }}>{(item.probability ?? 0.5).toFixed(2)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Optimizer ──────────────────────────────────────────────── */
export function OptimizerTab({ isActive }) {
  const [tickers, setTickers] = useState('AAPL,MSFT,GOOGL,NVDA,AMZN');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true); setResults(null);
    try {
      const syms = tickers.split(',').map(t => t.trim().toUpperCase()).filter(Boolean);
      setResults(await api.optimize({ tickers: syms, method: 'sharpe' }));
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><span className="card-title">⚖️ Portfolio Optimizer</span></div>
        <div className="card-body">
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <div className="bt-field" style={{ flex: 1 }}>
              <label className="bt-label">Tickers (comma-separated)</label>
              <input className="bt-input" value={tickers} style={{ width: '100%' }} onChange={e => setTickers(e.target.value)} />
            </div>
            <button className="btn btn-primary" onClick={run} disabled={loading}>{loading ? '⏳' : '⚖️ Optimize'}</button>
          </div>
        </div>
      </div>
      {loading && <div className="spinner-wrap"><div className="spinner" /></div>}
      {results && !loading && (
        <div className="card fade-in">
          <div className="card-header"><span className="card-title">Optimal Portfolio Weights</span></div>
          <table className="data-table">
            <thead><tr><th>Ticker</th><th>Weight %</th><th>Expected Return</th><th>Contribution</th></tr></thead>
            <tbody>
              {Object.entries(results.weights || {}).map(([sym, w]) => (
                <tr key={sym}>
                  <td style={{ fontWeight: 700 }}>{sym}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 80, background: 'var(--border)', borderRadius: 3, height: 6, overflow: 'hidden' }}>
                        <div style={{ width: `${(w * 100).toFixed(0)}%`, height: 6, background: 'var(--blue)', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontWeight: 700, color: 'var(--blue)', fontFamily: 'monospace' }}>{(w * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td style={{ color: 'var(--green)', fontFamily: 'monospace' }}>{results.expected_returns?.[sym] ? `+${(results.expected_returns[sym] * 100).toFixed(2)}%` : '—'}</td>
                  <td style={{ color: 'var(--dim)' }}>{results.contributions?.[sym] ? `${(results.contributions[sym] * 100).toFixed(2)}%` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="card-body" style={{ borderTop: '1px solid var(--border)' }}>
            <div className="bt-results">
              {[
                ['Expected Return', (results.expected_return ?? 0), '%'],
                ['Expected Volatility', (results.expected_volatility ?? 0), '%'],
                ['Sharpe Ratio', (results.sharpe_ratio ?? 0), ''],
              ].map(([label, val, unit]) => (
                <div key={label} className="bt-metric">
                  <div className="bt-metric-lbl">{label}</div>
                  <div className="bt-metric-val" style={{ color: 'var(--cyan)', fontSize: 15 }}>
                    {unit === '%' ? `${(val * 100).toFixed(2)}%` : val.toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Settings ───────────────────────────────────────────────── */
export function SettingsTab({ isActive }) {
  const [settings, setSettings] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (isActive) api.settings().then(setSettings).catch(() => {});
  }, [isActive]);

  async function save() {
    setSaving(true); setSaved(false);
    try { await api.saveSettings(settings); setSaved(true); setTimeout(() => setSaved(false), 2000); }
    catch (e) { alert(e.message); }
    finally { setSaving(false); }
  }

  if (!settings) return <div className="spinner-wrap"><div className="spinner" /></div>;

  return (
    <div style={{ maxWidth: 700, margin: '0 auto' }}>
      <div className="card">
        <div className="card-header">
          <span className="card-title">⚙️ System Settings</span>
          <button className="btn btn-primary btn-sm" onClick={save} disabled={saving}>
            {saving ? '⏳ Saving…' : saved ? '✓ Saved!' : '💾 Save'}
          </button>
        </div>
        <div className="card-body">
          {Object.entries(settings).map(([section, vals]) => (
            typeof vals === 'object' && vals !== null && !Array.isArray(vals) ? (
              <div key={section} style={{ marginBottom: 24 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--cyan)', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 12, paddingBottom: 6, borderBottom: '1px solid var(--border)' }}>
                  {section.replace(/_/g, ' ')}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
                  {Object.entries(vals).map(([key, val]) => (
                    <div key={key} className="bt-field">
                      <label className="bt-label">{key.replace(/_/g, ' ')}</label>
                      <input
                        className="bt-input"
                        type={typeof val === 'number' ? 'number' : 'text'}
                        value={val}
                        onChange={e => setSettings(prev => ({
                          ...prev,
                          [section]: { ...prev[section], [key]: e.target.type === 'number' ? Number(e.target.value) : e.target.value }
                        }))}
                        style={{ width: '100%' }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            ) : null
          ))}
        </div>
      </div>
    </div>
  );
}
