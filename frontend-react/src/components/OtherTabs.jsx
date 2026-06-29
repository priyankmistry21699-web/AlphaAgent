import { useState, useEffect } from 'react';
import { api } from '../api.js';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  ReferenceLine, Cell, ComposedChart,
} from 'recharts';

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
  { id: 'markov_regime', label: 'Markov Regime Chain',       cat: 'Regime',          icon: '⛓️' },
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

/* ── Chart helpers ──────────────────────────────────────────── */
const C = { blue: '#3b82f6', cyan: '#22d3ee', green: '#22c55e', red: '#ef4444', yellow: '#f59e0b', purple: '#a855f7', dim: '#555', bg: 'var(--bg)', border: 'var(--border)' };
const chartTooltipStyle = { background: 'var(--card2)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 11 };
const axisStyle = { fontSize: 9, fill: '#666' };

function MCFanChart({ bands, currentPrice }) {
  if (!bands?.length) return null;
  const data = bands.map(b => ({
    day: `T+${b.day}`,
    p10: b.p10,
    p25: b.p25,
    p50: b.p50,
    p75: b.p75,
    p90: b.p90,
    // stacked from-zero approach: base invisible, then each band width
    base: b.p10,
    band_outer_lo: b.p25 - b.p10,
    band_inner:    b.p75 - b.p25,
    band_outer_hi: b.p90 - b.p75,
  }));
  const allVals = bands.flatMap(b => [b.p10, b.p90]);
  const yMin = Math.min(...allVals) * 0.998;
  const yMax = Math.max(...allVals) * 1.002;
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Price Fan Chart — P10 / P25 / Median / P75 / P90</div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis dataKey="day" tick={axisStyle} interval={Math.floor(data.length / 6)} />
          <YAxis domain={[yMin, yMax]} tick={axisStyle} tickFormatter={v => `$${v.toFixed(0)}`} width={55} />
          <Tooltip contentStyle={chartTooltipStyle} formatter={(v, n) => [`$${Number(v).toFixed(2)}`, n]} />
          <ReferenceLine y={currentPrice} stroke={C.yellow} strokeDasharray="4 2" label={{ value: 'Now', fill: C.yellow, fontSize: 9 }} />
          {/* Stacked outer band (p10→p25 + p75→p90) */}
          <Area type="monotone" dataKey="base"          stackId="s" fill="transparent" stroke="none" />
          <Area type="monotone" dataKey="band_outer_lo" stackId="s" fill={C.blue} fillOpacity={0.10} stroke="none" />
          <Area type="monotone" dataKey="band_inner"    stackId="s" fill={C.blue} fillOpacity={0.22} stroke="none" />
          <Area type="monotone" dataKey="band_outer_hi" stackId="s" fill={C.blue} fillOpacity={0.10} stroke="none" />
          <Line type="monotone" dataKey="p50" stroke={C.cyan} strokeWidth={2} dot={false} name="Median" />
        </ComposedChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', gap: 14, fontSize: 10, color: 'var(--dim)', marginTop: 6 }}>
        <span style={{ color: C.yellow }}>── Now ${currentPrice?.toFixed(2)}</span>
        <span style={{ color: C.cyan }}>── Median</span>
        <span style={{ color: C.blue, opacity: 0.7 }}>■ P25–P75 band</span>
        <span style={{ color: C.blue, opacity: 0.4 }}>■ P10–P90 band</span>
      </div>
    </div>
  );
}

function GARCHForecastChart({ series, regime, regimeColor }) {
  if (!series?.length) return null;
  const color = regimeColor || C.blue;
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Vol Forecast — next {series.length} days (annualized %)</div>
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={series} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis dataKey="day" tick={axisStyle} tickFormatter={v => `T+${v}`} />
          <YAxis tick={axisStyle} tickFormatter={v => `${v.toFixed(1)}%`} width={42} />
          <Tooltip contentStyle={chartTooltipStyle} formatter={v => [`${Number(v).toFixed(2)}%`, 'Vol']} />
          <defs>
            <linearGradient id="garchGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3} />
              <stop offset="95%" stopColor={color} stopOpacity={0.03} />
            </linearGradient>
          </defs>
          <Area type="monotone" dataKey="vol_pct" stroke={color} fill="url(#garchGrad)" strokeWidth={2} dot={false} name="Vol %" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function HMMRegimeChart({ probabilities, stateSequence }) {
  const probData = Object.entries(probabilities || {}).map(([regime, prob]) => ({
    regime,
    prob: Math.round(prob * 1000) / 10,
    color: regime === 'Bull' ? C.green : regime === 'Bear' ? C.red : C.yellow,
  }));
  const seqData = (stateSequence || []).slice(-40).map((s, i) => ({
    t: i,
    regime: s.regime,
    val: s.regime === 'Bull' ? 2 : s.regime === 'Bear' ? 0 : 1,
  }));
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Current Regime Probs</div>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={probData} layout="vertical" margin={{ top: 0, right: 8, left: 8, bottom: 0 }}>
              <XAxis type="number" domain={[0, 100]} tick={axisStyle} tickFormatter={v => `${v}%`} />
              <YAxis type="category" dataKey="regime" tick={{ fontSize: 10, fill: '#999' }} width={52} />
              <Tooltip contentStyle={chartTooltipStyle} formatter={v => [`${v}%`, 'Probability']} />
              <Bar dataKey="prob" radius={[0, 4, 4, 0]}>
                {probData.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        {seqData.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Regime History (last 40 days)</div>
            <ResponsiveContainer width="100%" height={140}>
              <AreaChart data={seqData} margin={{ top: 0, right: 4, left: 0, bottom: 0 }}>
                <XAxis dataKey="t" hide />
                <YAxis domain={[0, 2]} hide />
                <Tooltip contentStyle={chartTooltipStyle}
                  formatter={(v, n, p) => [p.payload.regime, 'Regime']} />
                <defs>
                  <linearGradient id="hmmGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={C.blue} stopOpacity={0.4} />
                    <stop offset="95%" stopColor={C.blue} stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <Area type="stepAfter" dataKey="val" stroke={C.blue} fill="url(#hmmGrad)" strokeWidth={1.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', gap: 10, fontSize: 9, color: 'var(--dim)', marginTop: 4 }}>
              <span style={{ color: C.red }}>0=Bear</span>
              <span style={{ color: C.yellow }}>1=Neutral</span>
              <span style={{ color: C.green }}>2=Bull</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Markov charts ───────────────────────────────────────────── */
const REGIME_COLOR = { Bull: '#22c55e', Bear: '#ef4444', Sideways: '#f59e0b' };

function TransitionMatrixChart({ matrix, labels }) {
  if (!matrix?.length) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>
        3×3 Transition Matrix — P(next | current)
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ padding: '4px 8px', color: 'var(--dim)', fontSize: 10, textAlign: 'left' }}>From \ To</th>
            {labels.map(l => (
              <th key={l} style={{ padding: '4px 8px', color: REGIME_COLOR[l], fontSize: 11, textAlign: 'center' }}>{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, ri) => (
            <tr key={ri}>
              <td style={{ padding: '4px 8px', fontWeight: 700, color: REGIME_COLOR[labels[ri]], fontSize: 11 }}>{labels[ri]}</td>
              {row.map((val, ci) => {
                const intensity = Math.round(val * 180);
                const isDiag = ri === ci;
                return (
                  <td key={ci} style={{
                    padding: '6px 8px', textAlign: 'center', fontFamily: 'monospace', fontSize: 12, fontWeight: isDiag ? 700 : 400,
                    background: `rgba(${ri === 1 ? '239,68,68' : ri === 0 ? '34,197,94' : '245,158,11'},${val * 0.4})`,
                    color: isDiag ? REGIME_COLOR[labels[ri]] : 'var(--text)',
                    borderRadius: 4, border: isDiag ? `1px solid ${REGIME_COLOR[labels[ri]]}44` : '1px solid transparent',
                  }}>
                    {(val * 100).toFixed(1)}%
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MarkovForecastFan({ forecastSteps }) {
  if (!forecastSteps?.length) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>
        Chapman-Kolmogorov n-Step Forecast
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={forecastSteps} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="step" tick={{ fontSize: 10, fill: 'var(--dim)' }} label={{ value: 'Steps ahead', position: 'insideBottom', offset: -2, fontSize: 10, fill: 'var(--dim)' }} />
          <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: 'var(--dim)' }} domain={[0, 1]} />
          <Tooltip formatter={(v, n) => [`${(v * 100).toFixed(1)}%`, n.charAt(0).toUpperCase() + n.slice(1)]} contentStyle={chartTooltipStyle} />
          <Area type="monotone" dataKey="bull"     stackId="1" stroke="#22c55e" fill="#22c55e" fillOpacity={0.6} strokeWidth={1.5} name="bull" />
          <Area type="monotone" dataKey="sideways" stackId="1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.5} strokeWidth={1.5} name="sideways" />
          <Area type="monotone" dataKey="bear"     stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} strokeWidth={1.5} name="bear" />
        </AreaChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', gap: 14, fontSize: 10, color: 'var(--dim)', marginTop: 4, justifyContent: 'center' }}>
        <span style={{ color: '#22c55e' }}>■ Bull</span>
        <span style={{ color: '#f59e0b' }}>■ Sideways</span>
        <span style={{ color: '#ef4444' }}>■ Bear</span>
      </div>
    </div>
  );
}

function RegimeHistoryChart({ history }) {
  if (!history?.length) return null;
  const data = history.map((d, i) => ({ ...d, idx: i, color: REGIME_COLOR[d.regime] || '#555' }));
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>
        Regime History — Last 60 Days
      </div>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }} barCategoryGap={1}>
          <XAxis dataKey="date" hide />
          <YAxis hide domain={[-15, 15]} />
          <Tooltip formatter={(v, n, p) => [`${v > 0 ? '+' : ''}${v.toFixed(1)}%`, p.payload.regime]} contentStyle={chartTooltipStyle} />
          <Bar dataKey="return_pct" radius={[1, 1, 0, 0]}>
            {data.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.85} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', gap: 14, fontSize: 10, color: 'var(--dim)', marginTop: 4, justifyContent: 'center' }}>
        <span style={{ color: '#22c55e' }}>■ Bull</span>
        <span style={{ color: '#f59e0b' }}>■ Sideways</span>
        <span style={{ color: '#ef4444' }}>■ Bear</span>
      </div>
    </div>
  );
}

function MarkovEquityChart({ equity }) {
  if (!equity?.length) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>
        Walk-Forward Equity (Base 100, No Lookahead)
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={equity} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="idx" hide />
          <YAxis tick={{ fontSize: 10, fill: 'var(--dim)' }} />
          <Tooltip formatter={(v) => [v.toFixed(2), 'Equity']} contentStyle={chartTooltipStyle} />
          <defs>
            <linearGradient id="mkvEqGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <ReferenceLine y={100} stroke="#555" strokeDasharray="4 2" />
          <Area type="monotone" dataKey="value" stroke="#3b82f6" fill="url(#mkvEqGrad)" strokeWidth={1.5} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function HestonVolSurface({ volSurface }) {
  if (!volSurface?.length) return null;
  const mats = [...new Set(volSurface.map(r => r.maturity_months))].sort((a, b) => a - b);
  const strikes = [...new Set(volSurface.map(r => r.strike_pct))].sort((a, b) => a - b);
  const lookup = {};
  volSurface.forEach(r => { lookup[`${r.maturity_months}_${r.strike_pct}`] = r.iv; });
  const ivVals = volSurface.map(r => r.iv).filter(Boolean);
  const ivMin = Math.min(...ivVals), ivMax = Math.max(...ivVals);
  const ivColor = (iv) => {
    if (!iv) return C.bg;
    const t = Math.max(0, Math.min(1, (iv - ivMin) / (ivMax - ivMin + 0.001)));
    const r = Math.round(59 + t * 196), g = Math.round(130 - t * 108), b = Math.round(246 - t * 210);
    return `rgb(${r},${g},${b})`;
  };
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Implied Vol Surface</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 10, width: '100%' }}>
          <thead>
            <tr>
              <th style={{ padding: '4px 8px', color: 'var(--dim)', textAlign: 'left', fontWeight: 400 }}>Mat ↓ / Strike →</th>
              {strikes.map(k => <th key={k} style={{ padding: '4px 8px', color: 'var(--dim)', fontWeight: 400 }}>{(k * 100).toFixed(0)}%</th>)}
            </tr>
          </thead>
          <tbody>
            {mats.map(m => (
              <tr key={m}>
                <td style={{ padding: '4px 8px', color: 'var(--dim)', fontFamily: 'monospace' }}>{m}M</td>
                {strikes.map(k => {
                  const iv = lookup[`${m}_${k}`];
                  return (
                    <td key={k} style={{ padding: '4px 8px', textAlign: 'center', fontFamily: 'monospace', background: ivColor(iv), borderRadius: 3, color: iv > (ivMin + ivMax) / 2 ? '#fff' : '#000' }}>
                      {iv ? `${iv.toFixed(1)}%` : '—'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RoughVolChart({ forecastedVol }) {
  if (!forecastedVol?.length) return null;
  const data = forecastedVol.slice(0, 20).map((v, i) => ({ day: i + 1, vol: parseFloat((v * 100).toFixed(3)) }));
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>rBergomi Vol Forecast (% annualized)</div>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis dataKey="day" tick={axisStyle} tickFormatter={v => `T+${v}`} />
          <YAxis tick={axisStyle} tickFormatter={v => `${v.toFixed(1)}%`} width={42} />
          <Tooltip contentStyle={chartTooltipStyle} formatter={v => [`${v}%`, 'Vol']} />
          <Line type="monotone" dataKey="vol" stroke={C.purple} strokeWidth={2} dot={{ r: 3, fill: C.purple }} name="Vol %" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function CopulaChart({ lowerTailDep, upperTailDep, portfolioVar, portfolioCvar }) {
  const tailData = [
    { name: 'Lower Tail λL', value: Math.round((lowerTailDep || 0) * 1000) / 10, color: C.red },
    { name: 'Upper Tail λU', value: Math.round((upperTailDep || 0) * 1000) / 10, color: C.green },
  ];
  const riskData = [
    { name: 'VaR 95%', value: Math.abs(Math.round((portfolioVar || 0) * 1000) / 10), color: C.yellow },
    { name: 'CVaR 95%', value: Math.abs(Math.round((portfolioCvar || 0) * 1000) / 10), color: C.red },
  ];
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
      <div>
        <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Tail Dependence</div>
        <ResponsiveContainer width="100%" height={130}>
          <BarChart data={tailData} margin={{ top: 0, right: 8, left: 0, bottom: 0 }}>
            <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#999' }} />
            <YAxis tick={axisStyle} tickFormatter={v => `${v}%`} domain={[0, 100]} />
            <Tooltip contentStyle={chartTooltipStyle} formatter={v => [`${v}%`, 'λ']} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {tailData.map((e, i) => <Cell key={i} fill={e.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div>
        <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Portfolio Risk</div>
        <ResponsiveContainer width="100%" height={130}>
          <BarChart data={riskData} margin={{ top: 0, right: 8, left: 0, bottom: 0 }}>
            <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#999' }} />
            <YAxis tick={axisStyle} tickFormatter={v => `${v}%`} />
            <Tooltip contentStyle={chartTooltipStyle} formatter={v => [`${v}%`, 'Loss']} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {riskData.map((e, i) => <Cell key={i} fill={e.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* ── AI Explain panel ────────────────────────────────────────── */
function ExplainPanel({ modelId, ticker, tickerB, data }) {
  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState('');
  const [err, setErr] = useState('');

  async function explain() {
    setLoading(true); setExplanation(''); setErr('');
    try {
      const res = await api.quantExplain({ model_id: modelId, ticker, ticker_b: tickerB || '', data });
      setExplanation(res.explanation || '');
    } catch (e) { setErr(e.message); }
    finally { setLoading(false); }
  }

  return (
    <div style={{ margin: '16px 0 0 0', borderTop: '1px solid var(--border)', paddingTop: 14 }}>
      {!explanation && !loading && (
        <button className="btn btn-sm" onClick={explain} style={{ background: 'var(--purple)22', color: 'var(--purple)', border: '1px solid var(--purple)44' }}>
          🤖 Explain with Gemini AI
        </button>
      )}
      {loading && <div style={{ fontSize: 12, color: 'var(--dim)' }}>⏳ Generating plain-English explanation…</div>}
      {err && <div style={{ fontSize: 11, color: 'var(--red)', marginTop: 6 }}>⚠ {err}</div>}
      {explanation && (
        <div style={{ background: 'var(--card2)', border: '1px solid var(--purple)33', borderRadius: 8, padding: '12px 14px', fontSize: 12, color: 'var(--text)', lineHeight: 1.65 }}>
          <div style={{ fontSize: 10, color: 'var(--purple)', fontWeight: 700, letterSpacing: '.06em', marginBottom: 8 }}>🤖 GEMINI AI EXPLANATION</div>
          {explanation.split('\n').map((line, i) => {
            const parts = line.split(/(\*\*[^*]+\*\*)/g);
            return (
              <p key={i} style={{ margin: '0 0 6px 0' }}>
                {parts.map((p, j) =>
                  p.startsWith('**') ? <strong key={j}>{p.slice(2, -2)}</strong> : p
                )}
              </p>
            );
          })}
          <button className="btn btn-sm" onClick={explain} style={{ marginTop: 6, fontSize: 10 }}>↻ Regenerate</button>
        </div>
      )}
    </div>
  );
}

/* ── QuantResults ────────────────────────────────────────────── */
function QuantResults({ modelId, data, ticker, tickerB }) {
  if (!data) return null;

  function Generic() {
    const nums = Object.entries(data).filter(([, v]) => typeof v === 'number').slice(0, 12);
    const sig = data.signal || data.label || data.regime;
    return (
      <div className="card-body">
        {sig && <div style={{ marginBottom: 12 }}>{signalBadge(sig)}</div>}
        <MetricGrid items={nums.map(([k, v]) => ({ label: k.replace(/_/g, ' '), val: fmt(v) }))} />
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} tickerB={tickerB} data={data} />
      </div>
    );
  }

  if (modelId === 'monte_carlo') {
    const prob = data.prob_above_current ?? 0;
    const probColor = prob >= 60 ? C.green : prob <= 40 ? C.red : C.yellow;
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(prob >= 60 ? 'BULLISH_BIAS' : prob <= 40 ? 'BEARISH_BIAS' : 'NEUTRAL_DRIFT')}
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>{data.num_paths?.toLocaleString()} paths · {data.simulation_days} days</span>
        </div>
        <MCFanChart bands={data.path_bands} currentPrice={data.current_price} />
        <MetricGrid items={[
          { label: 'Current Price',    val: `$${fmt(data.current_price, 2)}` },
          { label: 'Median (T+30)',    val: `$${fmt(data.median_price, 2)}`,      color: C.cyan },
          { label: 'Expected Return',  val: `${data.expected_return_pct >= 0 ? '+' : ''}${fmt(data.expected_return_pct, 2)}%`, color: data.expected_return_pct >= 0 ? C.green : C.red },
          { label: 'P(above current)', val: `${fmt(prob, 1)}%`,                   color: probColor },
          { label: 'P10 Price',        val: `$${fmt(data.p10_price, 2)}`,          color: C.red },
          { label: 'P90 Price',        val: `$${fmt(data.p90_price, 2)}`,          color: C.green },
        ]} />
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

  if (modelId === 'garch') {
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal || data.vol_regime)}
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>GARCH(1,1) — convergence: {data.converged ? '✓' : '✗'}</span>
        </div>
        <GARCHForecastChart series={data.forecast_series} regime={data.vol_regime} regimeColor={data.regime_color} />
        <MetricGrid items={[
          { label: '1-Day Vol',       val: `${fmt(data.vol_1day, 2)}%`,   color: data.regime_color || C.blue },
          { label: '5-Day Vol',       val: `${fmt(data.vol_5day, 2)}%` },
          { label: '10-Day Vol',      val: `${fmt(data.vol_10day, 2)}%` },
          { label: 'Vol Regime',      val: data.vol_regime || '—',        color: data.regime_color || C.blue },
          { label: 'Percentile',      val: `${fmt(data.vol_percentile, 0)}th`,  color: C.dim },
          { label: 'α (News Impact)', val: fmt(data.alpha, 5) },
          { label: 'β (Persistence)', val: fmt(data.beta, 5) },
          { label: 'α+β',             val: fmt(data.persistence, 5),      color: data.persistence >= 0.99 ? C.red : C.green },
          { label: 'Long-run Vol',    val: `${fmt(data.long_run_vol, 2)}%` },
        ]} />
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

  if (modelId === 'hmm') {
    const regColor = (data.current_regime || '').includes('Bull') ? C.green : (data.current_regime || '').includes('Bear') ? C.red : C.yellow;
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal || data.current_regime?.toUpperCase()?.replace(' ', '_'))}
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>Duration: {data.regime_duration_days} days</span>
        </div>
        <HMMRegimeChart probabilities={data.probabilities} stateSequence={data.state_sequence} />
        <MetricGrid items={[
          { label: 'Current Regime', val: data.current_regime || '—', color: regColor },
          { label: 'Duration',       val: `${data.regime_duration_days} days` },
          ...Object.entries(data.probabilities || {}).map(([k, v]) => ({ label: `P(${k})`, val: `${(v * 100).toFixed(1)}%` })),
        ]} />
        {data.transition_risk && Object.keys(data.transition_risk).length > 0 && (
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.06em' }}>Transition Risk</div>
            <MetricGrid items={Object.entries(data.transition_risk).map(([k, v]) => ({ label: k, val: `${(v * 100).toFixed(1)}%`, color: C.yellow }))} />
          </div>
        )}
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

  if (modelId === 'markov_regime') {
    const regColor = REGIME_COLOR[data.current_regime] || C.dim;
    const bt = data.backtest || {};
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14, flexWrap: 'wrap' }}>
          {signalBadge(data.signal)}
          <span style={{ fontSize: 14, fontWeight: 800, color: regColor }}>{data.current_regime}</span>
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>for {data.regime_duration_days} day{data.regime_duration_days !== 1 ? 's' : ''}</span>
          <span style={{ fontSize: 11, color: C.dim, marginLeft: 8 }}>
            score: <b style={{ color: (data.signed_score ?? 0) >= 0 ? C.green : C.red }}>{(data.signed_score >= 0 ? '+' : '')}{(data.signed_score ?? 0).toFixed(3)}</b>
            &nbsp;(bull − bear, 1-step)
          </span>
        </div>

        <TransitionMatrixChart matrix={data.transition_matrix} labels={data.state_labels || ['Bull','Bear','Sideways']} />
        <MarkovForecastFan forecastSteps={data.forecast_steps} />
        <RegimeHistoryChart history={data.regime_history} />

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 14 }}>
          {Object.entries(data.stationary || {}).map(([k, v]) => (
            <div key={k} style={{ background: `${REGIME_COLOR[k]}18`, border: `1px solid ${REGIME_COLOR[k]}44`, borderRadius: 6, padding: '4px 12px', textAlign: 'center' }}>
              <div style={{ fontSize: 10, color: 'var(--dim)' }}>Stationary {k}</div>
              <div style={{ fontSize: 15, fontWeight: 700, fontFamily: 'monospace', color: REGIME_COLOR[k] }}>{(v * 100).toFixed(1)}%</div>
            </div>
          ))}
        </div>

        <MetricGrid items={[
          { label: 'Conviction',     val: `${(data.conviction ?? 0).toFixed(1)}%`,         color: regColor },
          { label: 'Signed Score',   val: (data.signed_score >= 0 ? '+' : '') + (data.signed_score ?? 0).toFixed(3), color: (data.signed_score ?? 0) >= 0 ? C.green : C.red },
          { label: 'WF Sharpe',      val: fmt(bt.sharpe, 3),                                color: (bt.sharpe ?? 0) >= 1 ? C.green : (bt.sharpe ?? 0) >= 0 ? C.yellow : C.red },
          { label: 'WF Return',      val: `${(bt.total_return_pct ?? 0) >= 0 ? '+' : ''}${fmt(bt.total_return_pct, 1)}%`, color: (bt.total_return_pct ?? 0) >= 0 ? C.green : C.red },
          { label: 'Max Drawdown',   val: `${fmt(bt.max_dd_pct, 1)}%`,                      color: C.red },
          { label: 'Win Rate',       val: `${fmt(bt.win_rate_pct, 1)}%`,                    color: (bt.win_rate_pct ?? 0) >= 55 ? C.green : C.yellow },
          { label: 'WF Trades',      val: bt.n_trades ?? 0 },
        ]} />

        <MarkovEquityChart equity={bt.equity} />
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

  if (modelId === 'heston') {
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>Heston SV — {ticker}</span>
        </div>
        <HestonVolSurface volSurface={data.vol_surface} />
        <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Calibrated Parameters</div>
        <MetricGrid items={[
          { label: 'v₀ (Initial Var)', val: fmt(data.v0 ?? data.params?.v0) },
          { label: 'κ (Mean Revert)',  val: fmt(data.kappa ?? data.params?.kappa, 3) },
          { label: 'θ (Long-run Var)', val: fmt(data.theta ?? data.params?.theta) },
          { label: 'σ (Vol-of-Vol)',   val: fmt(data.sigma ?? data.params?.sigma, 3) },
          { label: 'ρ (Correlation)',  val: fmt(data.rho ?? data.params?.rho, 3) },
        ]} />
        <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Pricing</div>
        <MetricGrid items={[
          { label: 'COS Option Price', val: `$${fmt(data.cos_price ?? data.call_price, 2)}`, color: C.blue },
          { label: 'MC Option Price',  val: `$${fmt(data.mc_price ?? data.put_price, 2)}`,   color: C.blue },
          { label: 'Model IV',         val: fmtPct2(data.model_iv ?? data.implied_vol / 100), color: C.yellow },
        ]} />
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

  if (modelId === 'sabr') {
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>SABR smile — {ticker}</span>
        </div>
        <MetricGrid items={[
          { label: 'α (Vol Level)',    val: fmt(data.alpha, 3) },
          { label: 'β (CEV Exponent)', val: fmt(data.beta, 3) },
          { label: 'ν (Vol-of-Vol)',   val: fmt(data.nu, 3) },
          { label: 'ρ (Correlation)',  val: fmt(data.rho, 3) },
          { label: 'ATM Vol',          val: fmtPct2(data.atm_vol), color: C.cyan },
          { label: 'Skew (dσ/dK)',     val: fmt(data.skew) },
          { label: 'Convexity',        val: fmt(data.convexity) },
        ]} />
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

  if (modelId === 'rough_vol') {
    const hColor = (data.hurst || 0) < 0.3 ? C.red : (data.hurst || 0) < 0.45 ? C.yellow : C.green;
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal || data.label)}
        </div>
        <RoughVolChart forecastedVol={data.forecasted_vol} />
        <MetricGrid items={[
          { label: 'Hurst Exponent H', val: fmt(data.hurst, 3), color: hColor },
          { label: 'Vol-of-Vol',       val: fmt(data.vol_of_vol, 3) },
          { label: 'Roughness Label',  val: data.label || '—', color: hColor },
        ]} />
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

  if (modelId === 'multifractal') {
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
        </div>
        <MetricGrid items={[
          { label: 'DFA Hurst',     val: fmt(data.hurst, 3) },
          { label: 'Spectrum Width α', val: fmt(data.spectrum_width, 3) },
          { label: 'MSM Vol Forecast', val: data.vol_forecast?.[0] ? fmtPct2(data.vol_forecast[0]) : '—' },
          { label: 'Regime',        val: data.volatility_regime || data.signal || '—', color: C.yellow },
        ]} />
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

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
            <thead><tr><th>Reference</th><th>F-stat</th><th>p-value</th><th>Lag</th><th>Verdict</th></tr></thead>
            <tbody>
              {tests.map((t, i) => {
                const causal = t.is_causal || t.p_value < 0.05;
                return (
                  <tr key={i}>
                    <td style={{ fontWeight: 600 }}>{t.reference || t.ticker}</td>
                    <td style={{ fontFamily: 'monospace' }}>{fmt(t.f_stat, 2)}</td>
                    <td style={{ fontFamily: 'monospace', color: t.p_value < 0.05 ? C.green : 'var(--dim)' }}>{fmt(t.p_value, 4)}</td>
                    <td style={{ fontFamily: 'monospace' }}>{t.lag ?? t.optimal_lag ?? '—'}</td>
                    <td>{causal ? <span style={{ color: C.green, fontWeight: 700 }}>GRANGER-CAUSES</span> : <span style={{ color: 'var(--dim)' }}>NO</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

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
                <span key={c} style={{ background: C.green + '22', color: C.green, border: `1px solid ${C.green}44`, borderRadius: 6, padding: '2px 8px', fontSize: 12 }}>{c.replace(/_/g, ' ')}</span>
              ))}
            </div>
          </div>
        )}
        {effects.length > 0 && (
          <table className="data-table">
            <thead><tr><th>Factor</th><th>ATE</th><th>p-value</th><th>Verdict</th></tr></thead>
            <tbody>
              {effects.map((e, i) => {
                const color = e.causal_verdict === 'CAUSAL' ? C.green : e.causal_verdict === 'LIKELY_CAUSAL' ? C.yellow : 'var(--dim)';
                return (
                  <tr key={i}>
                    <td style={{ fontWeight: 600 }}>{(e.treatment || '').replace(/_/g, ' ')}</td>
                    <td style={{ fontFamily: 'monospace', color: e.ate > 0 ? C.green : C.red }}>{fmt(e.ate * 100, 3)}%</td>
                    <td style={{ fontFamily: 'monospace' }}>{fmt(e.significance, 4)}</td>
                    <td><span style={{ color, fontWeight: 700, fontSize: 11 }}>{e.causal_verdict}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

  if (modelId === 'lob') {
    const regColor = data.regime === 'LIQUID' ? C.green : data.regime === 'TOXIC' ? C.red : C.yellow;
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal || data.regime)}
        </div>
        <MetricGrid items={[
          { label: 'Kyle λ',       val: fmt(data.kyle_lambda, 6) },
          { label: 'Amihud Illiq.', val: fmt(data.amihud, 6) },
          { label: 'Roll Spread',  val: fmtPct2(data.roll_spread) },
          { label: 'PIN',          val: fmt(data.pin, 3) },
          { label: 'VPIN',         val: fmt(data.vpin, 3) },
          { label: 'Regime',       val: data.regime || '—', color: regColor },
        ]} />
        {data.lob_simulation && (
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>LOB Simulation (Poisson)</div>
            <MetricGrid items={[
              { label: 'Arrival λ',     val: fmt(data.lob_simulation.arrival_rate, 3) },
              { label: 'Mid Price',     val: `$${fmt(data.lob_simulation.mid_price, 2)}` },
              { label: 'Bid-Ask Spread', val: fmtPct2(data.lob_simulation.bid_ask_spread) },
              { label: 'Depth (lots)',  val: fmt(data.lob_simulation.depth, 0) },
            ]} />
          </div>
        )}
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

  if (modelId === 'copula') {
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>{ticker} — tail co-dependence</span>
        </div>
        <CopulaChart lowerTailDep={data.lower_tail_dep} upperTailDep={data.upper_tail_dep} portfolioVar={data.portfolio_var} portfolioCvar={data.portfolio_cvar} />
        <MetricGrid items={[
          { label: 'Lower Tail λL',      val: fmt(data.lower_tail_dep, 4), color: C.red },
          { label: 'Upper Tail λU',      val: fmt(data.upper_tail_dep, 4), color: C.green },
          { label: 'Portfolio VaR 95%',  val: fmtPct2(data.portfolio_var), color: C.red },
          { label: 'Portfolio CVaR 95%', val: fmtPct2(data.portfolio_cvar), color: C.red },
        ]} />
        {data.copula_params && (
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Fitted Copula Parameters</div>
            <MetricGrid items={Object.entries(data.copula_params).map(([k, v]) => ({ label: k.replace(/_/g, ' '), val: fmt(v, 4) }))} />
          </div>
        )}
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} tickerB={tickerB} data={data} />
      </div>
    );
  }

  if (modelId === 'quantum') {
    const weights = data.qaoa_weights || [];
    const peerNames = ['Ticker', 'SPY', 'QQQ', 'GLD', 'TLT', 'XLK'];
    const wData = weights.map((w, i) => ({ name: peerNames[i] || `A${i}`, weight: Math.round(w * 1000) / 10 }));
    return (
      <div className="card-body">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          {signalBadge(data.signal)}
          <span style={{ fontSize: 11, color: 'var(--dim)' }}>Method: {data.method}</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>QAOA Portfolio Weights</div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={wData} margin={{ top: 0, right: 8, left: 0, bottom: 0 }}>
                <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#999' }} />
                <YAxis tick={axisStyle} tickFormatter={v => `${v}%`} domain={[0, 100]} />
                <Tooltip contentStyle={chartTooltipStyle} formatter={v => [`${v}%`, 'Weight']} />
                <Bar dataKey="weight" fill={C.blue} radius={[4, 4, 0, 0]}>
                  {wData.map((e, i) => <Cell key={i} fill={e.weight > 25 ? C.green : C.blue} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>QAE Option Pricing</div>
            <MetricGrid items={[
              { label: 'QAE Call Price',  val: `$${fmt(data.qae_option_price, 2)}`, color: C.blue },
              { label: 'QAE Confidence', val: fmt(data.qae_confidence, 4) },
              { label: 'QAOA E[Return]', val: fmtPct2(data.qaoa_expected_return), color: C.green },
              { label: 'QAOA Risk σ',    val: fmtPct2(data.qaoa_risk) },
              { label: 'QAOA Sharpe',    val: fmt(data.qaoa_sharpe, 3), color: C.cyan },
            ]} />
          </div>
        </div>
        {data.quantum_advantage_note && (
          <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 10, padding: '8px 12px', background: 'var(--bg)', borderRadius: 8, border: '1px solid var(--border)' }}>
            ⚛️ {data.quantum_advantage_note}
          </div>
        )}
        <Notes items={data.notes} />
        <ExplainPanel modelId={modelId} ticker={ticker} data={data} />
      </div>
    );
  }

  return <Generic />;
}

/* ── QuantLabTab ─────────────────────────────────────────────── */
export function QuantLabTab({ isActive }) {
  const [ticker, setTicker]       = useState('AAPL');
  const [peerTicker, setPeerTicker] = useState('MSFT');
  const [modelId, setModelId]     = useState('monte_carlo');
  const [compareMode, setCompareMode] = useState(false);
  const [results, setResults]     = useState(null);
  const [resultsB, setResultsB]   = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState('');
  const [runLabel, setRunLabel]   = useState('AAPL');

  const modelInfo = QUANT_MODELS.find(m => m.id === modelId);
  const isCopula  = modelId === 'copula';
  const needsB    = isCopula || compareMode;

  async function runOne(t, isB = false) {
    switch (modelId) {
      case 'monte_carlo':  return api.quantMC({ ticker: t, paths: 1000 });
      case 'garch':        return api.quantGARCH({ ticker: t });
      case 'hmm':          return api.quantHMM({ ticker: t });
      case 'markov_regime': return api.quantMarkovRegime(t);
      case 'heston':       return api.quantHeston(t);
      case 'sabr':         return api.quantSABR(t);
      case 'rough_vol':    return api.quantRoughVol(t);
      case 'multifractal': return api.quantMultifractal(t);
      case 'granger':      return api.quantGranger(t);
      case 'causal':       return api.quantCausal(t);
      case 'lob':          return api.quantLOB(t);
      case 'copula':       return isB ? null : api.quantCopula(ticker.trim().toUpperCase(), peerTicker.trim().toUpperCase());
      case 'quantum':      return api.quantQuantum(t);
      default:             throw new Error('Unknown model');
    }
  }

  async function run() {
    setLoading(true); setResults(null); setResultsB(null); setError('');
    const t = ticker.trim().toUpperCase();
    const p = peerTicker.trim().toUpperCase();
    setRunLabel(isCopula ? `${t} vs ${p}` : compareMode ? t : t);
    try {
      if (isCopula) {
        setResults(await runOne(t));
      } else if (compareMode) {
        const [rA, rB] = await Promise.all([runOne(t), runOne(p, true)]);
        setResults(rA); setResultsB(rB);
      } else {
        setResults(await runOne(t));
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const categories = [...new Set(QUANT_MODELS.map(m => m.cat))];

  return (
    <div style={{ maxWidth: compareMode ? 1200 : 960, margin: '0 auto' }}>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <span className="card-title">🔭 Advanced Quant Lab</span>
          <span style={{ fontSize: 10, color: 'var(--dim)' }}>Charts · AI Explain · Compare</span>
        </div>
        <div className="card-body">
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <div className="bt-field">
              <label className="bt-label">Model</label>
              <select className="bt-input" value={modelId} onChange={e => { setModelId(e.target.value); setResults(null); setResultsB(null); setError(''); }} style={{ minWidth: 240 }}>
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
              <label className="bt-label">{isCopula ? 'Ticker A' : compareMode ? 'Ticker A' : 'Ticker'}</label>
              <input className="bt-input" value={ticker} style={{ width: 90 }} onChange={e => setTicker(e.target.value.toUpperCase())} placeholder="AAPL" />
            </div>
            {needsB && (
              <div className="bt-field">
                <label className="bt-label">{isCopula ? 'Ticker B' : 'Ticker B (compare)'}</label>
                <input className="bt-input" value={peerTicker} style={{ width: 90 }} onChange={e => setPeerTicker(e.target.value.toUpperCase())} placeholder="MSFT" />
              </div>
            )}
            <button className="btn btn-primary" onClick={run} disabled={loading} style={{ alignSelf: 'flex-end' }}>
              {loading ? '⏳ Running…' : `${modelInfo?.icon || '▶'} Run`}
            </button>
            {!isCopula && (
              <button
                className="btn btn-sm"
                onClick={() => { setCompareMode(c => !c); setResults(null); setResultsB(null); }}
                style={{ alignSelf: 'flex-end', background: compareMode ? 'var(--blue)22' : 'transparent', color: compareMode ? 'var(--blue)' : 'var(--dim)', border: `1px solid ${compareMode ? 'var(--blue)' : 'var(--border)'}` }}
              >
                ⇄ {compareMode ? 'Comparing' : 'Compare'}
              </button>
            )}
          </div>
          <div style={{ marginTop: 10, fontSize: 11, color: 'var(--dim)' }}>
            {modelInfo?.cat} · {modelInfo?.label}
            {modelId === 'monte_carlo'  && ' — GBM + GARCH vol, 1,000 paths, P10/P25/P50/P75/P90 fan chart'}
            {modelId === 'garch'        && ' — GARCH(1,1) fit, multi-step vol forecast series'}
            {modelId === 'hmm'          && ' — 3-state Gaussian HMM, Viterbi decoding, regime probabilities'}
            {modelId === 'markov_regime' && ' — rolling-return labels (±5%, 20d) → 3×3 ML transition matrix → Chapman-Kolmogorov forecast → stationary π → walk-forward backtest'}
            {modelId === 'heston'       && ' — COS pricing + Milstein MC + calibration to vol surface'}
            {modelId === 'sabr'         && ' — Hagan 2002 closed-form, ATM skew & convexity analytics'}
            {modelId === 'rough_vol'    && ' — Hurst estimation via structure functions, rBergomi simulation'}
            {modelId === 'multifractal' && ' — MSM with k=6 components, DFA Hurst, multifractal spectrum'}
            {modelId === 'granger'      && ' — AIC lag selection, F-test vs SPY/QQQ/TLT/GLD/^VIX'}
            {modelId === 'causal'       && ' — IPW ATE, 2SLS IV, mediation analysis, Pearl do-calculus'}
            {modelId === 'lob'          && ' — Kyle λ, Amihud, Roll spread, PIN, VPIN, Poisson LOB simulation'}
            {modelId === 'copula'       && ' — Gaussian/t/Clayton/Gumbel/Frank, empirical tail dep., portfolio VaR/CVaR'}
            {modelId === 'quantum'      && ' — QAE option pricing O(1/ε), QAOA portfolio (QUBO)'}
          </div>
        </div>
      </div>

      {error && <div className="warn-banner" style={{ marginBottom: 16 }}>⚠ {error}</div>}
      {loading && <div className="spinner-wrap"><div className="spinner" /></div>}

      {results && !loading && (
        <div style={{ display: compareMode && resultsB ? 'grid' : 'block', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="card fade-in">
            <div className="card-header">
              <span className="card-title">{modelInfo?.icon} {modelInfo?.label} — {ticker.toUpperCase()}</span>
            </div>
            <QuantResults modelId={modelId} data={results} ticker={ticker.toUpperCase()} tickerB={isCopula ? peerTicker.toUpperCase() : undefined} />
          </div>
          {compareMode && resultsB && (
            <div className="card fade-in">
              <div className="card-header">
                <span className="card-title">{modelInfo?.icon} {modelInfo?.label} — {peerTicker.toUpperCase()}</span>
              </div>
              <QuantResults modelId={modelId} data={resultsB} ticker={peerTicker.toUpperCase()} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Leaderboard ────────────────────────────────────────────── */
const LB_AGENT_COLORS = {
  fundamental: '#3b82f6', macro: '#22d3ee', technical: '#22c55e',
  insider: '#f59e0b', geopolitical: '#a855f7', sentiment: '#ec4899',
  volatility: '#ef4444', currency: '#6366f1', risk: '#555',
};
const LB_DAYS_OPTIONS = [30, 60, 90, 180];

function AccuracyBar({ pct, color = '#3b82f6' }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ flex: 1, background: 'var(--border)', borderRadius: 3, height: 6, overflow: 'hidden' }}>
        <div style={{ width: `${Math.min(pct, 100)}%`, height: 6, background: color, borderRadius: 3, transition: 'width 0.5s ease' }} />
      </div>
      <span style={{ fontFamily: 'monospace', fontSize: 11, color, minWidth: 36 }}>{pct.toFixed(1)}%</span>
    </div>
  );
}

function OutcomeBadge({ outcome }) {
  const styles = {
    CORRECT: { background: 'rgba(34,197,94,0.15)', color: 'var(--green)', border: '1px solid rgba(34,197,94,0.3)' },
    WRONG:   { background: 'rgba(239,68,68,0.15)',  color: 'var(--red)',   border: '1px solid rgba(239,68,68,0.3)' },
    PENDING: { background: 'rgba(148,163,184,0.1)',  color: 'var(--dim)',   border: '1px solid var(--border)' },
  };
  return (
    <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 4, letterSpacing: '0.05em', ...styles[outcome] }}>
      {outcome}
    </span>
  );
}

export function LeaderboardTab({ isActive }) {
  const [lb, setLb] = useState(null);
  const [runs, setRuns] = useState(null);
  const [activeTab, setActiveTab] = useState('rankings');
  const [days, setDays] = useState(90);
  const [loading, setLoading] = useState(false);
  const [tuning, setTuning] = useState(false);
  const [tuneResult, setTuneResult] = useState(null);

  useEffect(() => {
    if (isActive) load();
  }, [isActive, days]);

  async function load() {
    setLoading(true);
    setLb(null); setRuns(null); setTuneResult(null);
    try {
      const [lbRes, runsRes] = await Promise.allSettled([
        api.leaderboard(days),
        api.leaderboardSignalRuns(100),
      ]);
      if (lbRes.status === 'fulfilled') setLb(lbRes.value);
      if (runsRes.status === 'fulfilled') setRuns(runsRes.value);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }

  async function handleTuneWeights() {
    setTuning(true); setTuneResult(null);
    try {
      const r = await api.leaderboardTuneWeights(days);
      setTuneResult(r);
    } catch (e) { setTuneResult({ status: 'error', reason: String(e) }); }
    finally { setTuning(false); }
  }

  const rankings = lb?.rankings || [];
  const isDemo = lb?.is_demo ?? true;

  const tabStyle = (t) => ({
    padding: '6px 16px', cursor: 'pointer', fontSize: 12, fontWeight: 600,
    color: activeTab === t ? 'var(--blue)' : 'var(--dim)',
    background: 'none', border: 'none',
    borderBottom: activeTab === t ? '2px solid var(--blue)' : '2px solid transparent',
  });

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>

      {/* Header */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-body" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>Agent Leaderboard</span>
          {isDemo && (
            <span style={{ fontSize: 10, fontWeight: 700, background: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.3)', padding: '2px 8px', borderRadius: 4 }}>
              DEMO DATA
            </span>
          )}
          {lb && !isDemo && (
            <span style={{ fontSize: 11, color: 'var(--dim)' }}>
              {lb.total_signals_evaluated.toLocaleString()} predictions evaluated
            </span>
          )}
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: 'var(--dim)' }}>Window:</span>
            {LB_DAYS_OPTIONS.map(d => (
              <button key={d} onClick={() => setDays(d)}
                style={{ padding: '3px 10px', fontSize: 11, borderRadius: 4, cursor: 'pointer', fontWeight: 600,
                  background: days === d ? 'var(--blue)' : 'var(--card)', color: days === d ? '#fff' : 'var(--dim)',
                  border: '1px solid var(--border)' }}>
                {d}d
              </button>
            ))}
            <button onClick={load} disabled={loading} style={{ padding: '3px 10px', fontSize: 11, borderRadius: 4, cursor: 'pointer', background: 'var(--card)', color: 'var(--dim)', border: '1px solid var(--border)', fontWeight: 600 }}>
              {loading ? '⏳' : '↺'}
            </button>
          </div>
        </div>
        <div style={{ display: 'flex', borderTop: '1px solid var(--border)', padding: '0 16px' }}>
          <button style={tabStyle('rankings')} onClick={() => setActiveTab('rankings')}>Agent Rankings</button>
          <button style={tabStyle('runs')} onClick={() => setActiveTab('runs')}>Signal Runs</button>
        </div>
      </div>

      {loading && <div className="spinner-wrap"><div className="spinner" /></div>}

      {/* ── Agent Rankings Tab ── */}
      {!loading && activeTab === 'rankings' && (
        <>
          {rankings.length === 0 ? (
            <div className="empty-state"><div className="empty-icon">🏆</div><div className="empty-text">No rankings yet. Run signals to build history.</div></div>
          ) : (
            <>
              {/* Rankings table */}
              <div className="card" style={{ marginBottom: 16 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Agent</th>
                      <th>Predictions</th>
                      <th style={{ minWidth: 160 }}>Accuracy</th>
                      <th>Brier ↓</th>
                      <th>IC</th>
                      <th>IC_IR</th>
                      <th>Ann. IR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rankings.map((r, i) => {
                      const color = LB_AGENT_COLORS[r.agent] || '#3b82f6';
                      return (
                        <tr key={r.agent}>
                          <td style={{ fontWeight: 700 }}>
                            {i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${r.rank}`}
                          </td>
                          <td>
                            <span style={{ fontWeight: 700, color, textTransform: 'capitalize' }}>{r.agent}</span>
                          </td>
                          <td style={{ fontFamily: 'monospace', color: 'var(--dim)' }}>{r.n_predictions}</td>
                          <td><AccuracyBar pct={r.directional_accuracy_pct} color={color} /></td>
                          <td style={{ fontFamily: 'monospace', color: (r.brier_score ?? 0.25) < 0.25 ? 'var(--green)' : 'var(--red)' }}>
                            {(r.brier_score ?? 0).toFixed(4)}
                          </td>
                          <td style={{ fontFamily: 'monospace', color: (r.ic ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                            {(r.ic ?? 0).toFixed(4)}
                          </td>
                          <td style={{ fontFamily: 'monospace', color: (r.ic_ir ?? 0) >= 0.5 ? 'var(--green)' : (r.ic_ir ?? 0) >= 0 ? 'var(--yellow)' : 'var(--red)' }}>
                            {(r.ic_ir ?? 0).toFixed(3)}
                          </td>
                          <td style={{ fontFamily: 'monospace', color: (r.information_ratio ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                            {(r.information_ratio ?? 0).toFixed(3)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Accuracy bar chart */}
              <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-header"><span className="card-title">Directional Accuracy</span></div>
                <div className="card-body">
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={rankings} layout="vertical" margin={{ left: 80, right: 40, top: 4, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                      <XAxis type="number" domain={[40, 70]} tick={{ fontSize: 10, fill: 'var(--dim)' }} tickFormatter={v => `${v}%`} />
                      <YAxis type="category" dataKey="agent" tick={{ fontSize: 11, fill: 'var(--dim)', textTransform: 'capitalize' }} />
                      <Tooltip formatter={(v) => [`${v.toFixed(1)}%`, 'Accuracy']} contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', fontSize: 11 }} />
                      <ReferenceLine x={50} stroke="#555" strokeDasharray="4 2" label={{ value: '50%', fill: 'var(--dim)', fontSize: 10 }} />
                      <Bar dataKey="directional_accuracy_pct" radius={[0, 3, 3, 0]}>
                        {rankings.map(r => <Cell key={r.agent} fill={LB_AGENT_COLORS[r.agent] || '#3b82f6'} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* IC_IR chart */}
              <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-header"><span className="card-title">IC Information Ratio (higher = more consistent signal quality)</span></div>
                <div className="card-body">
                  <ResponsiveContainer width="100%" height={160}>
                    <BarChart data={rankings} layout="vertical" margin={{ left: 80, right: 40, top: 4, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                      <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--dim)' }} />
                      <YAxis type="category" dataKey="agent" tick={{ fontSize: 11, fill: 'var(--dim)', textTransform: 'capitalize' }} />
                      <Tooltip formatter={(v) => [v.toFixed(3), 'IC_IR']} contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', fontSize: 11 }} />
                      <ReferenceLine x={0} stroke="#555" />
                      <Bar dataKey="ic_ir" radius={[0, 3, 3, 0]}>
                        {rankings.map(r => <Cell key={r.agent} fill={(r.ic_ir ?? 0) >= 0 ? (LB_AGENT_COLORS[r.agent] || '#3b82f6') : '#ef4444'} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Weight suggestions + tune button */}
              {lb?.weight_suggestions && Object.keys(lb.weight_suggestions).length > 0 && (
                <div className="card" style={{ marginBottom: 16 }}>
                  <div className="card-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span className="card-title">Suggested Agent Weights (from IC_IR)</span>
                    <button className="btn btn-primary" onClick={handleTuneWeights} disabled={tuning || isDemo}
                      style={{ fontSize: 11, padding: '4px 12px' }}>
                      {tuning ? '⏳ Writing...' : '⚡ Apply Weights'}
                    </button>
                  </div>
                  {tuneResult && (
                    <div className="card-body" style={{ padding: '8px 16px' }}>
                      <span style={{ fontSize: 11, color: tuneResult.status === 'updated' ? 'var(--green)' : 'var(--red)' }}>
                        {tuneResult.status === 'updated' ? '✓ agents.yaml updated' : tuneResult.reason || 'Failed'}
                      </span>
                    </div>
                  )}
                  <div className="card-body" style={{ display: 'flex', flexWrap: 'wrap', gap: 10, paddingTop: 0 }}>
                    {Object.entries(lb.weight_suggestions).sort(([,a],[,b]) => b - a).map(([name, w]) => (
                      <div key={name} style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '6px 12px', minWidth: 110 }}>
                        <div style={{ fontSize: 11, fontWeight: 700, color: LB_AGENT_COLORS[name] || 'var(--text)', textTransform: 'capitalize' }}>{name}</div>
                        <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'monospace', color: 'var(--text)' }}>{(w * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ── Signal Runs Tab ── */}
      {!loading && activeTab === 'runs' && (
        <>
          {/* Summary stats */}
          {runs && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-body" style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                {[
                  { label: 'Total Runs', value: runs.total, color: 'var(--text)' },
                  { label: 'Correct', value: runs.correct, color: 'var(--green)' },
                  { label: 'Wrong', value: runs.wrong, color: 'var(--red)' },
                  { label: 'Pending', value: runs.pending, color: 'var(--dim)' },
                  { label: 'Live Accuracy', value: runs.accuracy_pct != null ? `${runs.accuracy_pct}%` : '—', color: runs.accuracy_pct >= 55 ? 'var(--green)' : runs.accuracy_pct >= 50 ? 'var(--yellow)' : 'var(--red)' },
                ].map(s => (
                  <div key={s.label} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 20, fontWeight: 800, fontFamily: 'monospace', color: s.color }}>{s.value}</div>
                    <div style={{ fontSize: 10, color: 'var(--dim)' }}>{s.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {/* Runs table */}
          {!runs || runs.runs?.length === 0 ? (
            <div className="empty-state"><div className="empty-icon">📋</div><div className="empty-text">No signal runs recorded yet.</div></div>
          ) : (
            <div className="card">
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Ticker</th>
                      <th>Direction</th>
                      <th>Horizon</th>
                      <th>Probability</th>
                      <th>Conviction</th>
                      <th>Actual Ret</th>
                      <th>Outcome</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.runs.map((r, i) => (
                      <tr key={i}>
                        <td style={{ fontSize: 10, color: 'var(--dim)', whiteSpace: 'nowrap' }}>
                          {r.timestamp ? new Date(r.timestamp).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—'}
                        </td>
                        <td style={{ fontWeight: 700, color: 'var(--blue)' }}>{r.ticker}</td>
                        <td style={{ color: r.direction === 'LONG' ? 'var(--green)' : r.direction === 'SHORT' ? 'var(--red)' : 'var(--dim)', fontWeight: 700, fontSize: 11 }}>
                          {r.direction}
                        </td>
                        <td style={{ color: 'var(--dim)', fontSize: 11 }}>{r.horizon || '1m'}</td>
                        <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{r.probability != null ? (r.probability * 100).toFixed(1) + '%' : '—'}</td>
                        <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{r.conviction != null ? r.conviction.toFixed(1) + '%' : '—'}</td>
                        <td style={{ fontFamily: 'monospace', fontSize: 11, color: r.actual_return_pct == null ? 'var(--dim)' : r.actual_return_pct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                          {r.actual_return_pct != null ? `${r.actual_return_pct >= 0 ? '+' : ''}${r.actual_return_pct.toFixed(2)}%` : '—'}
                        </td>
                        <td><OutcomeBadge outcome={r.outcome || 'PENDING'} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
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
