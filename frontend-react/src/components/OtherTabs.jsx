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

const QUANT_MODELS = [
  { id: 'monte_carlo',  label: 'Monte Carlo / Quasi-MC',     cat: 'Simulation',      icon: '🎲' },
  { id: 'garch',        label: 'GARCH Volatility',           cat: 'Volatility',      icon: '📈' },
  { id: 'hmm',          label: 'HMM Regime Detection',       cat: 'Regime',          icon: '🔄' },
  { id: 'heston',       label: 'Heston Stochastic Vol',      cat: 'Volatility',      icon: '🌊' },
  { id: 'sabr',         label: 'SABR Smile Model',           cat: 'Volatility',      icon: '😁' },
  { id: 'rough_vol',    label: 'Rough Vol (rBergomi/fBM)',   cat: 'Volatility',      icon: '📊' },
  { id: 'multifractal', label: 'Multifractal (MSM)',         cat: 'Volatility',      icon: '🔮' },
  { id: 'granger',      label: 'Granger Causality',          cat: 'Causal',          icon: '🔗' },
  { id: 'causal',       label: 'Do-Calculus / Causal SCM',  cat: 'Causal',          icon: '⚗️' },
  { id: 'lob',          label: 'LOB Microstructure',         cat: 'Microstructure',  icon: '📋' },
  { id: 'copula',       label: 'Copula / Tail Dependence',   cat: 'Portfolio',       icon: '🔀' },
  { id: 'quantum',      label: 'Quantum Finance (QAE+QAOA)', cat: 'Quantum',         icon: '⚛️' },
];

function signalBadge(sig) {
  if (!sig) return null;
  const color =
    sig.includes('LONG') || sig.includes('OVER') || sig.includes('CAUSAL') || sig.includes('CHEAP') || sig.includes('LIQUID') ? 'var(--green)' :
    sig.includes('SHORT') || sig.includes('UNDER') || sig.includes('HIGH_TAIL') || sig.includes('RICH') || sig.includes('TOXIC') ? 'var(--red)' :
    sig.includes('ROUGH') || sig.includes('DRIVEN') || sig.includes('ILLIQUID') ? 'var(--yellow)' : 'var(--cyan)';
  return (
    <span style={{ background: color + '22', color, border: `1px solid ${color}44`, borderRadius: 6, padding: '3px 10px', fontWeight: 700, fontSize: 12, letterSpacing: '.04em' }}>
      {sig.replace(/_/g, ' ')}
    </span>
  );
}

function MetricGrid({ items }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 10, marginBottom: 16 }}>
      {items.map(({ label, val, color }) => (
        <div key={label} style={{ background: 'var(--bg)', borderRadius: 8, padding: '10px 14px', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 4 }}>{label}</div>
          <div style={{ fontSize: 15, fontWeight: 700, fontFamily: 'monospace', color: color || 'var(--cyan)' }}>{val}</div>
        </div>
      ))}
    </div>
  );
}

function Notes({ items }) {
  if (!items?.length) return null;
  return (
    <div style={{ marginTop: 10 }}>
      {items.map((n, i) => (
        <div key={i} style={{ fontSize: 12, color: 'var(--dim)', padding: '4px 0', borderBottom: i < items.length - 1 ? '1px solid var(--border)' : 'none' }}>
          • {n}
        </div>
      ))}
    </div>
  );
}

function fmt(v, decimals = 4) {
  if (v === null || v === undefined) return '—';
  if (typeof v !== 'number') return String(v);
  return Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(3) : v.toFixed(decimals);
}
function fmtPct2(v) { return v === null || v === undefined ? '—' : (v * 100).toFixed(2) + '%'; }

function QuantResults({ modelId, data, ticker }) {
  if (!data) return null;

  /* ── Generic fallback ── */
  function Generic() {
    const nums = Object.entries(data).filter(([, v]) => typeof v === 'number').slice(0, 12);
    const sig = data.signal || data.label || data.regime;
    return (
      <div className="card-body">
        {sig && <div style={{ marginBottom: 12 }}>{signalBadge(sig)}</div>}
        <MetricGrid items={nums.map(([k, v]) => ({ label: k.replace(/_/g, ' '), val: fmt(v) }))} />
        <Notes items={data.notes} />
      </div>
    );
  }

  /* ── Heston ── */
  if (modelId === 'heston') {
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>Heston SV model — {ticker}</span>
        </div>
        <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Calibrated Parameters</div>
        <MetricGrid items={[
          { label: 'v₀ (Initial Var)', val: fmt(data.v0) },
          { label: 'κ (Mean Revert)', val: fmt(data.kappa, 3) },
          { label: 'θ (Long-run Var)', val: fmt(data.theta) },
          { label: 'σ (Vol-of-Vol)', val: fmt(data.sigma, 3) },
          { label: 'ρ (Correlation)', val: fmt(data.rho, 3) },
        ]} />
        <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Pricing</div>
        <MetricGrid items={[
          { label: 'COS Option Price', val: `$${fmt(data.cos_price, 2)}`, color: 'var(--blue)' },
          { label: 'MC Option Price', val: `$${fmt(data.mc_price, 2)}`, color: 'var(--blue)' },
          { label: 'Model IV', val: fmtPct2(data.model_iv), color: 'var(--yellow)' },
          { label: 'Realized Vol', val: fmtPct2(data.realized_vol), color: 'var(--yellow)' },
        ]} />
        <Notes items={data.notes} />
      </div>
    );
  }

  /* ── SABR ── */
  if (modelId === 'sabr') {
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>SABR smile — {ticker}</span>
        </div>
        <MetricGrid items={[
          { label: 'α (Vol Level)', val: fmt(data.alpha, 3) },
          { label: 'β (CEV Exponent)', val: fmt(data.beta, 3) },
          { label: 'ν (Vol-of-Vol)', val: fmt(data.nu, 3) },
          { label: 'ρ (Correlation)', val: fmt(data.rho, 3) },
          { label: 'ATM Vol', val: fmtPct2(data.atm_vol), color: 'var(--cyan)' },
          { label: 'Skew (dσ/dK)', val: fmt(data.skew) },
          { label: 'Convexity', val: fmt(data.convexity) },
        ]} />
        <Notes items={data.notes} />
      </div>
    );
  }

  /* ── Rough Vol ── */
  if (modelId === 'rough_vol') {
    const hColor = data.hurst < 0.3 ? 'var(--red)' : data.hurst < 0.45 ? 'var(--yellow)' : 'var(--green)';
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal || data.label)}
        </div>
        <MetricGrid items={[
          { label: 'Hurst Exponent H', val: fmt(data.hurst, 3), color: hColor },
          { label: 'Vol-of-Vol', val: fmt(data.vol_of_vol, 3) },
          { label: 'Roughness Label', val: data.label || '—', color: hColor },
        ]} />
        {data.forecasted_vol?.length > 0 && (
          <>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>rBergomi Vol Forecast (next {data.forecasted_vol.length} days)</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {data.forecasted_vol.slice(0, 10).map((v, i) => (
                <div key={i} style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '4px 8px', fontSize: 12, fontFamily: 'monospace' }}>
                  T+{i + 1}: {fmtPct2(v)}
                </div>
              ))}
            </div>
          </>
        )}
        <Notes items={data.notes} />
      </div>
    );
  }

  /* ── Multifractal ── */
  if (modelId === 'multifractal') {
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
        </div>
        <MetricGrid items={[
          { label: 'DFA Hurst', val: fmt(data.hurst, 3) },
          { label: 'Spectrum Width α', val: fmt(data.spectrum_width, 3) },
          { label: 'MSM Vol Forecast', val: data.vol_forecast?.[0] ? fmtPct2(data.vol_forecast[0]) : '—' },
          { label: 'Regime', val: data.volatility_regime || data.signal || '—', color: 'var(--yellow)' },
        ]} />
        <Notes items={data.notes} />
      </div>
    );
  }

  /* ── Granger ── */
  if (modelId === 'granger') {
    const tests = data.tests || data.granger_tests || [];
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
          {data.narrative && <span style={{ fontSize: 12, color: 'var(--dim)' }}>{data.narrative}</span>}
        </div>
        {tests.length > 0 && (
          <table className="data-table" style={{ marginBottom: 12 }}>
            <thead>
              <tr><th>Reference</th><th>F-stat</th><th>p-value</th><th>Lag</th><th>Verdict</th></tr>
            </thead>
            <tbody>
              {tests.map((t, i) => {
                const causal = t.is_causal || t.p_value < 0.05;
                return (
                  <tr key={i}>
                    <td style={{ fontWeight: 600 }}>{t.reference || t.ticker}</td>
                    <td style={{ fontFamily: 'monospace' }}>{fmt(t.f_stat, 2)}</td>
                    <td style={{ fontFamily: 'monospace', color: t.p_value < 0.05 ? 'var(--green)' : 'var(--dim)' }}>{fmt(t.p_value, 4)}</td>
                    <td style={{ fontFamily: 'monospace' }}>{t.lag ?? t.optimal_lag ?? '—'}</td>
                    <td>{causal ? <span style={{ color: 'var(--green)', fontWeight: 700 }}>GRANGER-CAUSES</span> : <span style={{ color: 'var(--dim)' }}>NO</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        <Notes items={data.notes} />
      </div>
    );
  }

  /* ── Causal / SCM ── */
  if (modelId === 'causal') {
    const effects = data.causal_effects || [];
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
        </div>
        {data.narrative && (
          <div style={{ fontSize: 12, color: 'var(--text)', background: 'var(--bg)', borderRadius: 8, padding: '8px 12px', marginBottom: 14, border: '1px solid var(--border)' }}>
            {data.narrative}
          </div>
        )}
        {data.dominant_causes?.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.06em' }}>Dominant Causal Drivers</div>
            <div style={{ display: 'flex', gap: 6 }}>
              {data.dominant_causes.map(c => (
                <span key={c} style={{ background: 'var(--green)22', color: 'var(--green)', border: '1px solid var(--green)44', borderRadius: 6, padding: '2px 8px', fontSize: 12 }}>{c.replace(/_/g, ' ')}</span>
              ))}
            </div>
          </div>
        )}
        {effects.length > 0 && (
          <table className="data-table">
            <thead>
              <tr><th>Factor</th><th>ATE</th><th>p-value</th><th>Verdict</th></tr>
            </thead>
            <tbody>
              {effects.map((e, i) => {
                const color = e.causal_verdict === 'CAUSAL' ? 'var(--green)' : e.causal_verdict === 'LIKELY_CAUSAL' ? 'var(--yellow)' : 'var(--dim)';
                return (
                  <tr key={i}>
                    <td style={{ fontWeight: 600 }}>{(e.treatment || '').replace(/_/g, ' ')}</td>
                    <td style={{ fontFamily: 'monospace', color: e.ate > 0 ? 'var(--green)' : 'var(--red)' }}>{fmt(e.ate * 100, 3)}%</td>
                    <td style={{ fontFamily: 'monospace' }}>{fmt(e.significance, 4)}</td>
                    <td><span style={{ color, fontWeight: 700, fontSize: 11 }}>{e.causal_verdict}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        <Notes items={data.notes} />
      </div>
    );
  }

  /* ── LOB ── */
  if (modelId === 'lob') {
    const regColor = data.regime === 'LIQUID' ? 'var(--green)' : data.regime === 'TOXIC' ? 'var(--red)' : 'var(--yellow)';
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal || data.regime)}
        </div>
        <MetricGrid items={[
          { label: 'Kyle λ', val: fmt(data.kyle_lambda, 6) },
          { label: 'Amihud Illiq.', val: fmt(data.amihud, 6) },
          { label: 'Roll Spread', val: fmtPct2(data.roll_spread) },
          { label: 'PIN', val: fmt(data.pin, 3) },
          { label: 'VPIN', val: fmt(data.vpin, 3) },
          { label: 'Regime', val: data.regime || '—', color: regColor },
        ]} />
        {data.lob_simulation && (
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>LOB Simulation (Poisson)</div>
            <MetricGrid items={[
              { label: 'Arrival λ', val: fmt(data.lob_simulation.arrival_rate, 3) },
              { label: 'Mid Price', val: `$${fmt(data.lob_simulation.mid_price, 2)}` },
              { label: 'Bid-Ask Spread', val: fmtPct2(data.lob_simulation.bid_ask_spread) },
              { label: 'Depth (lots)', val: fmt(data.lob_simulation.depth, 0) },
            ]} />
          </div>
        )}
        <Notes items={data.notes} />
      </div>
    );
  }

  /* ── Copula ── */
  if (modelId === 'copula') {
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
        </div>
        <MetricGrid items={[
          { label: 'Lower Tail Dep. (λL)', val: fmt(data.lower_tail_dep, 4), color: 'var(--red)' },
          { label: 'Upper Tail Dep. (λU)', val: fmt(data.upper_tail_dep, 4), color: 'var(--green)' },
          { label: 'Portfolio VaR 95%', val: fmtPct2(data.portfolio_var), color: 'var(--red)' },
          { label: 'Portfolio CVaR 95%', val: fmtPct2(data.portfolio_cvar), color: 'var(--red)' },
        ]} />
        {data.copula_params && (
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Fitted Copula Parameters</div>
            <MetricGrid items={Object.entries(data.copula_params).map(([k, v]) => ({ label: k.replace(/_/g, ' '), val: fmt(v, 4) }))} />
          </div>
        )}
        <Notes items={data.notes} />
      </div>
    );
  }

  /* ── Quantum ── */
  if (modelId === 'quantum') {
    const weights = data.qaoa_weights || [];
    const peerNames = ['Ticker', 'SPY', 'QQQ', 'GLD', 'TLT', 'XLK'];
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
          <span style={{ fontSize: 11, color: 'var(--dim)' }}>Method: {data.method}</span>
        </div>
        <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>QAE Option Pricing</div>
        <MetricGrid items={[
          { label: 'QAE Call Price', val: `$${fmt(data.qae_option_price, 2)}`, color: 'var(--blue)' },
          { label: 'QAE Confidence', val: fmt(data.qae_confidence, 4) },
        ]} />
        <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>QAOA Portfolio Weights</div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
          {weights.map((w, i) => (
            <div key={i} style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8, padding: '8px 12px', minWidth: 80, textAlign: 'center' }}>
              <div style={{ fontSize: 10, color: 'var(--dim)', marginBottom: 3 }}>{peerNames[i] || `Asset ${i}`}</div>
              <div style={{ fontWeight: 700, color: w > 0.25 ? 'var(--green)' : 'var(--cyan)', fontFamily: 'monospace' }}>{(w * 100).toFixed(1)}%</div>
              <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, marginTop: 4 }}>
                <div style={{ height: 3, width: `${Math.min(100, w * 200)}%`, background: 'var(--blue)', borderRadius: 2 }} />
              </div>
            </div>
          ))}
        </div>
        <MetricGrid items={[
          { label: 'QAOA E[Return]', val: fmtPct2(data.qaoa_expected_return), color: 'var(--green)' },
          { label: 'QAOA Risk σ', val: fmtPct2(data.qaoa_risk) },
          { label: 'QAOA Sharpe', val: fmt(data.qaoa_sharpe, 3), color: 'var(--cyan)' },
        ]} />
        {data.quantum_advantage_note && (
          <div style={{ fontSize: 11, color: 'var(--dim)', marginTop: 10, padding: '8px 12px', background: 'var(--bg)', borderRadius: 8, border: '1px solid var(--border)' }}>
            ⚛️ {data.quantum_advantage_note}
          </div>
        )}
        <Notes items={data.notes} />
      </div>
    );
  }

  return <Generic />;
}

export function QuantLabTab({ isActive }) {
  const [ticker, setTicker] = useState('AAPL');
  const [peerTicker, setPeerTicker] = useState('MSFT');
  const [modelId, setModelId] = useState('monte_carlo');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [runLabel, setRunLabel] = useState('AAPL');

  const modelInfo = QUANT_MODELS.find(m => m.id === modelId);
  const isCopula = modelId === 'copula';

  async function run() {
    setLoading(true); setResults(null); setError('');
    const t = ticker.trim().toUpperCase();
    const p = peerTicker.trim().toUpperCase();
    setRunLabel(isCopula ? `${t} vs ${p}` : t);
    try {
      let res;
      switch (modelId) {
        case 'monte_carlo':   res = await api.quantMC({ ticker: t, paths: 1000 }); break;
        case 'garch':         res = await api.quantGARCH({ ticker: t }); break;
        case 'hmm':           res = await api.quantHMM({ ticker: t }); break;
        case 'heston':        res = await api.quantHeston(t); break;
        case 'sabr':          res = await api.quantSABR(t); break;
        case 'rough_vol':     res = await api.quantRoughVol(t); break;
        case 'multifractal':  res = await api.quantMultifractal(t); break;
        case 'granger':       res = await api.quantGranger(t); break;
        case 'causal':        res = await api.quantCausal(t); break;
        case 'lob':           res = await api.quantLOB(t); break;
        case 'copula':        res = await api.quantCopula(t, p); break;
        case 'quantum':       res = await api.quantQuantum(t); break;
        default:              throw new Error('Unknown model');
      }
      setResults(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const categories = [...new Set(QUANT_MODELS.map(m => m.cat))];

  return (
    <div style={{ maxWidth: 960, margin: '0 auto' }}>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><span className="card-title">🔭 Advanced Quant Lab</span></div>
        <div className="card-body">
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <div className="bt-field">
              <label className="bt-label">Model</label>
              <select className="bt-input" value={modelId} onChange={e => { setModelId(e.target.value); setResults(null); setError(''); }} style={{ minWidth: 240 }}>
                {categories.map(cat => (
                  <optgroup key={cat} label={`── ${cat} ──`}>
                    {QUANT_MODELS.filter(m => m.cat === cat).map(m => (
                      <option key={m.id} value={m.id}>{m.icon} {m.label}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
            <div className="bt-field">
              <label className="bt-label">{isCopula ? 'Ticker A' : 'Ticker'}</label>
              <input className="bt-input" value={ticker} style={{ width: 90 }} onChange={e => setTicker(e.target.value.toUpperCase())} placeholder="AAPL" />
            </div>
            {isCopula && (
              <div className="bt-field">
                <label className="bt-label">Ticker B</label>
                <input className="bt-input" value={peerTicker} style={{ width: 90 }} onChange={e => setPeerTicker(e.target.value.toUpperCase())} placeholder="MSFT" />
              </div>
            )}
            <button className="btn btn-primary" onClick={run} disabled={loading} style={{ alignSelf: 'flex-end' }}>
              {loading ? '⏳ Running…' : `${modelInfo?.icon || '▶'} Run`}
            </button>
          </div>
          <div style={{ marginTop: 10, fontSize: 11, color: 'var(--dim)' }}>
            {modelInfo?.cat} · {modelInfo?.label}
            {modelId === 'heston' && ' — COS pricing + Milstein MC + calibration to vol surface'}
            {modelId === 'sabr' && ' — Hagan 2002 closed-form, ATM skew & convexity analytics'}
            {modelId === 'rough_vol' && ' — Hurst estimation via structure functions, rBergomi simulation'}
            {modelId === 'multifractal' && ' — MSM with k=6 components, DFA Hurst, multifractal spectrum'}
            {modelId === 'granger' && ' — AIC lag selection, F-test vs SPY/QQQ/TLT/GLD/^VIX, spectral causality'}
            {modelId === 'causal' && ' — IPW ATE, 2SLS IV, mediation analysis, Pearl do-calculus'}
            {modelId === 'lob' && ' — Kyle λ, Amihud, Roll spread, PIN, VPIN, Poisson LOB simulation'}
            {modelId === 'copula' && ' — Gaussian/t/Clayton/Gumbel/Frank, empirical tail dep., portfolio VaR/CVaR'}
            {modelId === 'quantum' && ' — QAE option pricing O(1/ε), QAOA portfolio (QUBO), Quantum Annealing'}
          </div>
        </div>
      </div>

      {error && <div className="warn-banner" style={{ marginBottom: 16 }}>⚠ {error}</div>}
      {loading && <div className="spinner-wrap"><div className="spinner" /></div>}

      {results && !loading && (
        <div className="card fade-in">
          <div className="card-header">
            <span className="card-title">{modelInfo?.icon} {modelInfo?.label} — {runLabel}</span>
          </div>
          <QuantResults modelId={modelId} data={results} ticker={runLabel} />
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
                  {Object.entries(vals).map(([key, val]) => {
                    const isApiKey = section === 'api_keys' || /api.?key|secret|token|password/i.test(key);
                    const inputType = typeof val === 'number' ? 'number' : isApiKey ? 'password' : 'text';
                    return (
                    <div key={key} className="bt-field">
                      <label className="bt-label">{key.replace(/_/g, ' ')}</label>
                      <input
                        className="bt-input"
                        type={inputType}
                        value={val}
                        placeholder={isApiKey ? 'Paste API key here…' : undefined}
                        autoComplete={isApiKey ? 'off' : undefined}
                        onChange={e => setSettings(prev => ({
                          ...prev,
                          [section]: { ...prev[section], [key]: e.target.type === 'number' ? Number(e.target.value) : e.target.value }
                        }))}
                        style={{ width: '100%' }}
                      />
                    </div>
                    );
                  })}
                </div>
              </div>
            ) : null
          ))}
        </div>
      </div>
    </div>
  );
}
