import { useState, useEffect, useRef, useCallback } from 'react';
import { api, dirColor, priceColor, fmtNum } from '../api.js';
import { HZ_LABELS, TOP_TICKERS, AGENT_THEORIES, QUICK_CHIPS } from '../constants.js';
import AgentRow from './AgentRow.jsx';
import AIAssistant from './AIAssistant.jsx';

function ProbBar({ prob, color }) {
  return (
    <div className="prob-bar-wrap">
      <div className="prob-bar">
        <div className="prob-bar-fill" style={{ width: `${(prob * 100).toFixed(0)}%`, background: color }} />
      </div>
      <span className="prob-val" style={{ color }}>{(prob * 100).toFixed(1)}%</span>
    </div>
  );
}

function AgentTheoryTags({ agentName }) {
  const info = AGENT_THEORIES[agentName];
  if (!info) return null;
  return (
    <div style={{ marginTop: 4, lineHeight: 2 }}>
      {info.tags.slice(0, 6).map(tag => (
        <span key={tag} className={`tag ${info.color}`}>{tag}</span>
      ))}
    </div>
  );
}

export default function SignalTab({ isActive }) {
  const [ticker, setTicker] = useState('');
  const [horizon, setHorizon] = useState('1m');
  const [loading, setLoading] = useState(false);
  const [signal, setSignal] = useState(null);
  const [error, setError] = useState('');
  const [autocomplete, setAutocomplete] = useState([]);
  const [showAuto, setShowAuto] = useState(false);
  const inputRef = useRef(null);

  // Handle pending ticker from Market tab "Analyze" button
  useEffect(() => {
    if (isActive && window._pendingTicker) {
      const t = window._pendingTicker;
      delete window._pendingTicker;
      setTicker(t);
      runSignal(t, horizon);
    }
  }, [isActive]);

  function filterAutocomplete(val) {
    if (!val || val.length < 1) { setAutocomplete([]); setShowAuto(false); return; }
    const q = val.toUpperCase();
    const matches = TOP_TICKERS.filter(t =>
      t.sym.startsWith(q) || t.name.toUpperCase().includes(q)
    ).slice(0, 8);
    setAutocomplete(matches);
    setShowAuto(matches.length > 0);
  }

  function handleInput(e) {
    const val = e.target.value.toUpperCase();
    setTicker(val);
    filterAutocomplete(val);
  }

  function selectAuto(sym) {
    setTicker(sym);
    setShowAuto(false);
    runSignal(sym, horizon);
  }

  async function runSignal(t, h) {
    const sym = (t || ticker).trim().toUpperCase();
    if (!sym) return;
    setLoading(true);
    setError('');
    setSignal(null);
    setShowAuto(false);
    try {
      const d = await api.signal(sym, h || horizon);
      setSignal({ ...d, ticker: sym });
      // Store globally for deep dive
      window._lastSignal = d;
      window._agentMap = {};
      (d.agents || []).forEach(a => {
        window._agentMap[a.agent_name || a.name || 'Unknown'] = a;
      });
    } catch (err) {
      setError(`Failed to fetch signal for ${sym}: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') { runSignal(ticker, horizon); }
    if (e.key === 'Escape') setShowAuto(false);
  }

  function changeHorizon(h) {
    setHorizon(h);
    if (signal) runSignal(signal.ticker, h);
  }

  const dir = signal?.direction ? signal.direction.toUpperCase() : '';
  const prob = signal?.probability || 0.5;
  const baseProb = signal?.base_probability || prob;
  const conv = signal?.conviction || 0;
  const warns = signal?.warnings || [];
  const agents = signal?.agents || [];
  const holding = signal?.holding_period || {};
  const hzLabel = HZ_LABELS[horizon] || horizon.toUpperCase();
  const dColor = dirColor(dir);
  const pColor = priceColor(prob);

  const hasOverride = warns.some(w => /(HALT|CRITICAL|OVERRIDE|SWAN|CRASH)/i.test(w));

  return (
    <div>
      {/* Top chips */}
      <div className="chips-row">
        {TOP_TICKERS.slice(0, 20).map(t => (
          <button key={t.sym} className="chip" onClick={() => { setTicker(t.sym); runSignal(t.sym, horizon); }}>
            {t.sym}
          </button>
        ))}
      </div>

      {/* Form */}
      <div className="signal-form">
        <div className="signal-input-wrap">
          <input
            ref={inputRef}
            className="signal-input"
            placeholder="Enter ticker (e.g. AAPL, NVDA, BTC-USD)"
            value={ticker}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            onFocus={() => filterAutocomplete(ticker)}
            onBlur={() => setTimeout(() => setShowAuto(false), 160)}
            autoComplete="off"
            spellCheck={false}
          />
          {showAuto && (
            <div className="signal-autocomplete">
              {autocomplete.map(item => (
                <div key={item.sym} className="sig-auto-item" onMouseDown={() => selectAuto(item.sym)}>
                  <span className="sig-auto-sym">{item.sym}</span>
                  <span className="sig-auto-name">{item.name}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="hz-buttons">
          {Object.entries(HZ_LABELS).map(([h, label]) => (
            <button key={h} className={`hz-btn ${horizon === h ? 'active' : ''}`} onClick={() => changeHorizon(h)}>
              {h.toUpperCase()}
            </button>
          ))}
        </div>

        <button className="btn btn-primary" onClick={() => runSignal(ticker, horizon)} disabled={loading}>
          {loading ? '⏳ Analyzing…' : '⚡ Run Signal'}
        </button>
      </div>

      {/* Content */}
      <div className="signal-layout">
        <div className="signal-main">
          {loading && (
            <div className="spinner-wrap fade-in">
              <div className="spinner" />
              <div className="spinner-text">Running 9 agents in parallel…</div>
            </div>
          )}

          {error && (
            <div className="warn-banner fade-in">⚠ {error}</div>
          )}

          {!loading && !signal && !error && (
            <div className="empty-state">
              <div className="empty-icon">📡</div>
              <div className="empty-text">Click a ticker above or enter a symbol to run deep research across 9 agents and 163+ factors</div>
            </div>
          )}

          {signal && !loading && (
            <div className="fade-in">
              {/* Override banner */}
              {hasOverride && (
                <div className="override-banner">
                  🚨 Risk Override Active — {warns[0]}
                </div>
              )}
              {warns.filter(w => !/(HALT|CRITICAL|OVERRIDE|SWAN|CRASH)/i.test(w)).map((w, i) => (
                <div key={i} className="warn-banner">⚠ {w}</div>
              ))}

              {/* Action summary */}
              <div className={`action-card ${dir.toLowerCase() || 'hold'}`}>
                <div className="action-badge" style={{ color: dColor }}>{dir || 'HOLD'}</div>
                <div className="action-text">
                  <strong style={{ color: dColor }}>{signal.summary || `${dir} signal for ${signal.ticker}`}</strong>
                  {holding.horizon && (
                    <span> · Hold {holding.horizon} · Half-life {holding.half_life_days?.toFixed(1)} days</span>
                  )}
                </div>
              </div>

              {/* Metrics */}
              <div className="metrics-row">
                <div className="metric-card">
                  <div className="metric-lbl">Direction</div>
                  <div className="metric-val" style={{ color: dColor }}>{dir || '—'}</div>
                  <div className="metric-sub" style={{ color: 'var(--purple)' }}>Bayesian Fusion</div>
                </div>
                <div className="metric-card">
                  <div className="metric-lbl">P(Up) · <span style={{ color: 'var(--cyan)' }}>{hzLabel}</span></div>
                  <div className="metric-val" style={{ color: pColor }}>{(prob * 100).toFixed(1)}%</div>
                  <div className="metric-sub">
                    Horizon-Weighted
                    {horizon !== '1m' && (
                      <span style={{ color: 'var(--dim)', marginLeft: 6 }}>base {(baseProb * 100).toFixed(1)}%</span>
                    )}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-lbl">Conviction</div>
                  <div className="metric-val" style={{ color: 'var(--cyan)' }}>{conv.toFixed(1)}%</div>
                  <div className="metric-sub" style={{ color: 'var(--cyan)' }}>Ensemble</div>
                </div>
                <div className="metric-card">
                  <div className="metric-lbl">Agents</div>
                  <div className="metric-val" style={{ color: 'var(--blue)' }}>{agents.length}</div>
                  <div className="metric-sub">
                    {agents.filter(a => (a.vote || '').toUpperCase() === 'LONG').length}L /&nbsp;
                    {agents.filter(a => (a.vote || '').toUpperCase() === 'SHORT').length}S /&nbsp;
                    {agents.filter(a => (a.vote || '').toUpperCase() === 'HOLD').length}H
                  </div>
                </div>
              </div>

              {/* Agent Breakdown */}
              <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-header">
                  <span className="card-title">
                    Agent Breakdown — {signal.ticker}&nbsp;·&nbsp;
                    <span style={{ color: 'var(--cyan)' }}>{hzLabel} Horizon</span>
                    &nbsp;({agents.length} agents)
                  </span>
                  <span style={{ fontSize: 10, color: 'var(--dim)' }}>Click row to expand factors · 🔬 for deep analysis</span>
                </div>
                <div className="agent-table-wrap">
                  <table className="data-table" style={{ minWidth: 800 }}>
                    <thead>
                      <tr>
                        <th>Agent</th>
                        <th>Vote</th>
                        <th style={{ minWidth: 130 }}>P(Up)</th>
                        <th>Confidence</th>
                        <th>Compute</th>
                        <th style={{ maxWidth: 260 }}>Reasoning</th>
                      </tr>
                    </thead>
                    <tbody>
                      {agents.map(agent => (
                        <AgentRow key={agent.agent_name || agent.name} agent={agent} ticker={signal.ticker} hzLabel={hzLabel} />
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* AI Chatbot — shown after signal */}
        {signal && !loading && (
          <div className="signal-sidebar">
            <AIAssistant ticker={signal.ticker} signalData={signal} />
          </div>
        )}
      </div>
    </div>
  );
}
