import { useState, useEffect, useRef, useCallback } from 'react';
import { api, dirColor, priceColor, fmtNum } from '../api.js';
import { HZ_LABELS, TICKER_CATEGORIES, AGENT_THEORIES, QUICK_CHIPS, TOP_TICKERS } from '../constants.js';

import AgentRow from './AgentRow.jsx';
import AIAssistant from './AIAssistant.jsx';
import PriceChart from './PriceChart.jsx';
import QuantPanel from './QuantPanel.jsx';
import VolumeCard from './VolumeCard.jsx';

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
  const [holders, setHolders] = useState(null);
  const [holdersLoading, setHoldersLoading] = useState(false);
  const [holdersTab, setHoldersTab] = useState('institutional');
  const [tickerInfo, setTickerInfo] = useState(null);
  const [activeCat, setActiveCat] = useState('us');
  const [activeSub, setActiveSub] = useState('stocks');
  const [mktSummary, setMktSummary] = useState(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const genRef = useRef(0);

  // Load market summary once on mount
  useEffect(() => {
    api.marketSummary().then(s => setMktSummary(s)).catch(() => {});
  }, []);

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

    // Cancel any previous in-flight request
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const gen = ++genRef.current;

    setLoading(true);
    setError('');
    setSignal(null);
    setShowAuto(false);
    try {
      const d = await api.signal(sym, h || horizon, controller.signal);
      if (gen !== genRef.current) return;
      setSignal({ ...d, ticker: sym });
      setHolders(null);
      window._lastSignal = d;
      window._agentMap = {};
      (d.agents || []).forEach(a => {
        window._agentMap[a.agent_name || a.name || 'Unknown'] = a;
      });
      // Load smart money + ticker info in background after signal is shown
      setHoldersLoading(true);
      setTickerInfo(null);
      api.holders(sym).then(h => setHolders(h)).catch(() => {}).finally(() => setHoldersLoading(false));
      api.tickerInfo(sym).then(info => setTickerInfo(info)).catch(() => {});
    } catch (err) {
      if (gen !== genRef.current) return;
      // AbortError means a new analysis was started — let that one show its own spinner
      if (err.name !== 'AbortError') {
        console.error('[Signal] fetch error:', err);
        setError(`Failed to fetch signal for ${sym}: ${err.message || 'Unknown error'}`);
      }
    } finally {
      if (gen === genRef.current) setLoading(false);
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
  const council = signal?.council || [];
  const agreementScore = signal?.agreement_score ?? 0;
  const marketRegime = signal?.market_regime || null;
  const holding = signal?.holding_period || {};
  const hzLabel = HZ_LABELS[horizon] || horizon.toUpperCase();
  const dColor = dirColor(dir);
  const pColor = priceColor(prob);

  const hasOverride = warns.some(w => /(HALT|CRITICAL|OVERRIDE|SWAN|CRASH)/i.test(w));

  return (
    <div>
      {/* Tabbed ticker browser */}
      <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, marginBottom: 12, overflow: 'hidden' }}>
        {/* Row 1: Country tabs */}
        <div style={{ display: 'flex', overflowX: 'auto', borderBottom: '1px solid var(--border)', scrollbarWidth: 'none' }}>
          {TICKER_CATEGORIES.map(cat => (
            <button key={cat.id} onClick={() => {
              setActiveCat(cat.id);
              setActiveSub(cat.sub?.[0]?.id || 'stocks');
            }} style={{
              flexShrink: 0, padding: '7px 12px', fontSize: 11, border: 'none', cursor: 'pointer',
              background: 'transparent', whiteSpace: 'nowrap',
              color: activeCat === cat.id ? 'var(--text)' : 'var(--dim)',
              fontWeight: activeCat === cat.id ? 700 : 400,
              borderBottom: activeCat === cat.id ? '2px solid var(--blue)' : '2px solid transparent',
              transition: 'all .15s',
            }}>{cat.label}</button>
          ))}
        </div>
        {/* Row 2: Sub-category tabs (Stocks / ETFs / Sectors / Indices / Crypto …) */}
        {(() => {
          const subs = TICKER_CATEGORIES.find(c => c.id === activeCat)?.sub || [];
          return subs.length > 1 ? (
            <div style={{ display: 'flex', overflowX: 'auto', borderBottom: '1px solid var(--border)', scrollbarWidth: 'none', background: 'var(--card2)' }}>
              {subs.map(s => (
                <button key={s.id} onClick={() => setActiveSub(s.id)} style={{
                  flexShrink: 0, padding: '5px 11px', fontSize: 10, border: 'none', cursor: 'pointer',
                  background: 'transparent', whiteSpace: 'nowrap',
                  color: activeSub === s.id ? 'var(--cyan)' : 'var(--dim)',
                  fontWeight: activeSub === s.id ? 700 : 400,
                  borderBottom: activeSub === s.id ? '2px solid var(--cyan)' : '2px solid transparent',
                  transition: 'all .15s',
                }}>{s.label}</button>
              ))}
            </div>
          ) : null;
        })()}
        {/* Row 3: Ticker chips */}
        <div style={{ display: 'flex', gap: 5, overflowX: 'auto', padding: '7px 10px', flexWrap: 'nowrap', scrollbarWidth: 'none' }}>
          {(() => {
            const cat = TICKER_CATEGORIES.find(c => c.id === activeCat);
            const tickers = cat?.sub?.find(s => s.id === activeSub)?.tickers
              || cat?.sub?.[0]?.tickers
              || [];
            return tickers.map(t => (
              <button key={t.sym} className="chip" title={t.name}
                onClick={() => { setTicker(t.sym); runSignal(t.sym, horizon); }}
                style={{ whiteSpace: 'nowrap', flexShrink: 0 }}
              >{t.sym}</button>
            ));
          })()}
        </div>
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

        {/* ── LEFT PANEL: Live chart + stock data + prediction ── */}
        <div className="signal-left">

          {/* Market Summary strip — always visible */}
          {mktSummary && (() => {
            const indices = [
              ...(mktSummary.us     || []).slice(0, 3),
              ...(mktSummary.assets || []).filter(a => /BTC|Gold|Oil/i.test(a.label)).slice(0, 2),
            ];
            return (
              <div className="card" style={{ padding: '8px 10px', marginBottom: 8 }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 6 }}>Market Summary</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {indices.map(m => {
                    const pos = (m.change_pct ?? 0) >= 0;
                    const c   = pos ? 'var(--green)' : 'var(--red)';
                    const fp  = v => v == null ? '—' : v >= 10000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : v >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 1 }) : v.toFixed(2);
                    return (
                      <div key={m.symbol} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: 10, color: 'var(--dim2)', fontWeight: 600, minWidth: 90 }}>{m.label}</span>
                        <span style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--text)', fontWeight: 700 }}>{fp(m.price)}</span>
                        <span style={{ fontSize: 10, color: c, fontFamily: 'monospace', fontWeight: 700, minWidth: 54, textAlign: 'right' }}>
                          {pos ? '▲' : '▼'} {Math.abs(m.change_pct ?? 0).toFixed(2)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })()}

          {signal && (
            <>
              <PriceChart
                ticker={signal.ticker}
                sector={tickerInfo?.sector}
                currentPrice={tickerInfo?.price ?? signal?.current_price}
              />

              {/* Quote stats — renders from signal.current_price immediately, upgrades with full tickerInfo */}
              {(() => {
                const p      = tickerInfo?.price ?? signal?.current_price ?? null;
                const chg    = tickerInfo?.change ?? 0;
                const chgPct = tickerInfo?.change_pct ?? signal?.change_pct ?? 0;
                const isPos  = chgPct >= 0;
                const pc     = isPos ? 'var(--green)' : 'var(--red)';
                const lo52   = tickerInfo?.week52_low;
                const hi52   = tickerInfo?.week52_high;
                const rangePct = (p && lo52 && hi52 && hi52 > lo52)
                  ? ((p - lo52) / (hi52 - lo52) * 100).toFixed(0) : null;
                const fmtV = (v, dec = 2) => {
                  if (v == null) return '—';
                  if (Math.abs(v) >= 1e12) return (v / 1e12).toFixed(2) + 'T';
                  if (Math.abs(v) >= 1e9)  return (v / 1e9).toFixed(2) + 'B';
                  if (Math.abs(v) >= 1e6)  return (v / 1e6).toFixed(1) + 'M';
                  if (Math.abs(v) >= 1e3)  return (v / 1e3).toFixed(1) + 'K';
                  return Number(v).toFixed(dec);
                };
                if (p == null) return null;
                return (
                  <>
                    {/* Price header */}
                    <div className="card" style={{ padding: '10px 12px' }}>
                      <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 4 }}>
                        {tickerInfo?.short_name || signal.ticker}
                        {tickerInfo?.exchange && <span style={{ marginLeft: 6, color: 'var(--dim)' }}>· {tickerInfo.exchange}</span>}
                        {!tickerInfo && <span style={{ marginLeft: 6, color: 'var(--dim)', fontStyle: 'italic' }}>loading details…</span>}
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 8 }}>
                        <span style={{ fontSize: 22, fontWeight: 900, fontFamily: 'monospace', color: 'var(--text)' }}>
                          {`$${p.toLocaleString('en-US', { maximumFractionDigits: 2 })}`}
                        </span>
                        <div style={{ textAlign: 'right' }}>
                          {chg !== 0 && (
                            <div style={{ fontSize: 12, fontWeight: 700, color: pc, fontFamily: 'monospace' }}>
                              {isPos ? '+' : ''}{chg.toFixed(2)}
                            </div>
                          )}
                          {chgPct !== 0 && (
                            <div style={{ fontSize: 10, fontWeight: 700, color: pc }}>
                              {isPos ? '▲' : '▼'} {Math.abs(chgPct).toFixed(2)}%
                            </div>
                          )}
                        </div>
                      </div>

                      {/* OHLCV grid — only when tickerInfo available */}
                      {tickerInfo && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 8px', fontSize: 10 }}>
                          {[
                            ['Open',  tickerInfo.open],
                            ['High',  tickerInfo.high],
                            ['Close', tickerInfo.close],
                            ['Low',   tickerInfo.low],
                          ].map(([label, val]) => (
                            <div key={label} style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span style={{ color: 'var(--dim)' }}>{label}</span>
                              <span style={{ color: 'var(--text)', fontFamily: 'monospace', fontWeight: 600 }}>
                                {val != null ? `$${val.toLocaleString('en-US', { maximumFractionDigits: 2 })}` : '—'}
                              </span>
                            </div>
                          ))}
                          <div style={{ display: 'flex', justifyContent: 'space-between', gridColumn: '1/-1' }}>
                            <span style={{ color: 'var(--dim)' }}>Volume</span>
                            <span style={{ color: 'var(--text)', fontFamily: 'monospace', fontWeight: 600 }}>{fmtV(tickerInfo.volume, 0)}</span>
                          </div>
                        </div>
                      )}

                      {/* 52-week range bar */}
                      {rangePct != null && (
                        <div style={{ marginTop: 10 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--dim)', marginBottom: 3 }}>
                            <span>52W Low ${fmtV(lo52)}</span>
                            <span>52W High ${fmtV(hi52)}</span>
                          </div>
                          <div style={{ height: 5, background: 'var(--border)', borderRadius: 3, position: 'relative' }}>
                            <div style={{ position: 'absolute', left: `${rangePct}%`, top: -2, width: 9, height: 9, borderRadius: '50%', background: pc, border: '2px solid var(--card)', transform: 'translateX(-50%)' }} />
                            <div style={{ height: 5, width: `${rangePct}%`, borderRadius: 3, background: `linear-gradient(90deg, var(--red), var(--green))` }} />
                          </div>
                          <div style={{ fontSize: 9, color: 'var(--dim2)', textAlign: 'center', marginTop: 4 }}>{rangePct}% of range</div>
                        </div>
                      )}
                    </div>

                    {/* Key metrics — only when tickerInfo available */}
                    {tickerInfo && (
                      <div className="card" style={{ padding: '10px 12px' }}>
                        <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--cyan)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 8 }}>Key Metrics</div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px 8px', fontSize: 10 }}>
                          {[
                            ['Mkt Cap',   fmtV(tickerInfo.market_cap)],
                            ['P/E (TTM)', tickerInfo.pe_trailing?.toFixed(1) ?? '—'],
                            ['Beta',      tickerInfo.beta?.toFixed(2) ?? '—'],
                            ['Fwd P/E',   tickerInfo.pe_forward?.toFixed(1) ?? '—'],
                            ['EPS',       tickerInfo.eps != null ? `$${tickerInfo.eps.toFixed(2)}` : '—'],
                            ['Div Yld',   tickerInfo.dividend_yield != null ? `${(tickerInfo.dividend_yield * 100).toFixed(2)}%` : '—'],
                          ].map(([label, val]) => (
                            <div key={label} style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span style={{ color: 'var(--dim)' }}>{label}</span>
                              <span style={{ color: 'var(--text)', fontFamily: 'monospace', fontWeight: 600 }}>{val}</span>
                            </div>
                          ))}
                          {tickerInfo.industry && (
                            <div style={{ gridColumn: '1/-1', color: 'var(--dim)', fontSize: 9, borderTop: '1px solid var(--border)', paddingTop: 4, marginTop: 2 }}>
                              {tickerInfo.industry}
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* About / description */}
                    {tickerInfo?.description && (
                      <div className="card" style={{ padding: '10px 12px' }}>
                        <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--cyan)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 5 }}>About</div>
                        <div style={{ fontSize: 10, color: 'var(--dim2)', lineHeight: 1.65 }}>
                          {tickerInfo.description.length > 240
                            ? tickerInfo.description.slice(0, 240) + '…'
                            : tickerInfo.description}
                        </div>
                      </div>
                    )}
                  </>
                );
              })()}

              {/* Signal Prediction Gauge */}
              <div className="card" style={{ padding: '10px 12px' }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--cyan)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 10 }}>Market Prediction</div>
                {(() => {
                  const p100 = Math.round(prob * 100);
                  const angle = 180 - prob * 180;  // 180° (bear/left) → 90° (neut/top) → 0° (bull/right)
                  const rad = angle * Math.PI / 180;
                  const cx = 80, cy = 72, r = 56;
                  const nx = cx + r * Math.cos(rad);
                  const ny = cy - r * Math.sin(rad);  // Negate y: SVG y-axis points down
                  return (
                    <div style={{ textAlign: 'center' }}>
                      <svg viewBox="0 0 160 90" style={{ width: '100%', maxWidth: 200, height: 'auto', margin: '0 auto', display: 'block' }}>
                        {/* Arc segments */}
                        {[
                          ['rgba(245,54,92,.7)',  0,     0.33],
                          ['rgba(255,183,3,.7)',  0.33,  0.66],
                          ['rgba(0,214,143,.7)',  0.66,  1.0],
                        ].map(([color, from, to]) => {
                          const a1 = (180 - from * 180) * Math.PI / 180;
                          const a2 = (180 - to   * 180) * Math.PI / 180;
                          const x1 = cx + r * Math.cos(a1), y1 = cy - r * Math.sin(a1);
                          const x2 = cx + r * Math.cos(a2), y2 = cy - r * Math.sin(a2);
                          return (
                            <path key={color} d={`M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 0 0 ${x2} ${y2} Z`}
                              fill={color} opacity={.85} />
                          );
                        })}
                        {/* Arc border track */}
                        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 0 ${cx + r} ${cy}`}
                          fill="none" stroke="var(--border)" strokeWidth="2" />
                        {/* Needle */}
                        <line x1={cx} y1={cy} x2={nx} y2={ny}
                          stroke="white" strokeWidth="2.5" strokeLinecap="round"
                          style={{ filter: 'drop-shadow(0 0 3px rgba(255,255,255,.6))' }} />
                        <circle cx={cx} cy={cy} r={4} fill="var(--text)" />
                        {/* Labels */}
                        <text x="8"  y="86" fill="var(--red)"   fontSize="8" fontWeight="700">BEAR</text>
                        <text x="68" y="22" fill="var(--yellow)" fontSize="8" fontWeight="700">NEUT</text>
                        <text x="128" y="86" fill="var(--green)" fontSize="8" fontWeight="700">BULL</text>
                      </svg>
                      <div style={{ marginTop: 4 }}>
                        <span style={{ fontSize: 20, fontWeight: 900, fontFamily: 'monospace', color: dColor }}>{p100}%</span>
                        <span style={{ fontSize: 10, color: 'var(--dim)', marginLeft: 6 }}>P(Up)</span>
                      </div>
                      <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 6, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 11, fontWeight: 800, padding: '3px 10px', borderRadius: 8, color: dColor, background: `${dColor}18`, border: `1px solid ${dColor}40` }}>{dir || 'HOLD'}</span>
                        <span style={{ fontSize: 10, color: 'var(--dim2)' }}>Conv {conv.toFixed(0)}%</span>
                      </div>
                      {signal.ticker && (() => {
                        const longs  = agents.filter(a => (a.vote||'').toUpperCase() === 'LONG').length;
                        const shorts = agents.filter(a => (a.vote||'').toUpperCase() === 'SHORT').length;
                        const holds  = agents.filter(a => (a.vote||'').toUpperCase() === 'HOLD').length;
                        const aligned = dir === 'LONG' ? longs : dir === 'SHORT' ? shorts : holds;
                        const strength = prob >= 0.72 ? 'Strong bullish signal' : prob <= 0.28 ? 'Strong bearish signal'
                          : prob >= 0.60 ? 'Mild bullish bias' : prob <= 0.40 ? 'Mild bearish bias' : 'Neutral — wait for catalyst';
                        return (
                          <>
                            <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 4 }}>
                              {[
                                { label: 'Long',  val: longs,  color: 'var(--green)' },
                                { label: 'Hold',  val: holds,  color: 'var(--yellow)' },
                                { label: 'Short', val: shorts, color: 'var(--red)' },
                              ].map(({ label, val, color }) => (
                                <div key={label} style={{ background: 'var(--card2)', borderRadius: 6, padding: '4px 0', textAlign: 'center' }}>
                                  <div style={{ fontSize: 14, fontWeight: 800, color }}>{val}</div>
                                  <div style={{ fontSize: 8, color: 'var(--dim)', textTransform: 'uppercase' }}>{label}</div>
                                </div>
                              ))}
                            </div>
                            {/* Prediction summary */}
                            <div style={{ marginTop: 8, padding: '8px 10px', background: 'var(--card2)', borderRadius: 7, textAlign: 'left', borderLeft: `3px solid ${dColor}` }}>
                              <div style={{ fontSize: 10, color: dColor, fontWeight: 700, marginBottom: 3 }}>{strength}</div>
                              <div style={{ fontSize: 10, color: 'var(--dim2)', lineHeight: 1.6 }}>
                                {aligned} of {agents.length} agents aligned
                                {holding.half_life_days != null && ` · ~${Math.round(holding.half_life_days)}d signal half-life`}
                              </div>
                            </div>

                            {/* Agent vs Ensemble comparison */}
                            <div style={{ marginTop: 10, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                              <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 5 }}>Agent vs Ensemble</div>
                              {agents.map(a => {
                                const aName = (a.agent_name || a.name || '?').slice(0, 11);
                                const aVote = (a.vote || 'HOLD').toUpperCase();
                                const aProb = (a.probability_up ?? 0.5) * 100;
                                const aColor = aVote === 'LONG' ? 'var(--green)' : aVote === 'SHORT' ? 'var(--red)' : 'var(--yellow)';
                                return (
                                  <div key={aName} style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 3 }}>
                                    <div style={{ width: 58, fontSize: 8.5, color: 'var(--dim2)', flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{aName}</div>
                                    <div style={{ width: 10, fontSize: 9, color: aColor, flexShrink: 0 }}>{aVote === 'LONG' ? '▲' : aVote === 'SHORT' ? '▼' : '—'}</div>
                                    <div style={{ flex: 1, height: 4, background: 'var(--border)', borderRadius: 3, position: 'relative', overflow: 'hidden' }}>
                                      <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--dim)', opacity: .35 }} />
                                      <div style={{ width: `${aProb}%`, height: '100%', background: aColor, borderRadius: 3, opacity: .85 }} />
                                    </div>
                                    <div style={{ width: 28, fontSize: 8.5, fontFamily: 'monospace', color: aColor, flexShrink: 0, textAlign: 'right' }}>{aProb.toFixed(0)}%</div>
                                  </div>
                                );
                              })}
                              {/* Ensemble row */}
                              <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 5, paddingTop: 5, borderTop: '1px solid var(--border)' }}>
                                <div style={{ width: 58, fontSize: 8.5, fontWeight: 700, color: 'var(--text)', flexShrink: 0 }}>Ensemble</div>
                                <div style={{ width: 10, fontSize: 9, color: dColor, flexShrink: 0 }}>{dir === 'LONG' ? '▲' : dir === 'SHORT' ? '▼' : '—'}</div>
                                <div style={{ flex: 1, height: 4, background: 'var(--border)', borderRadius: 3, position: 'relative', overflow: 'hidden' }}>
                                  <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--dim)', opacity: .35 }} />
                                  <div style={{ width: `${prob * 100}%`, height: '100%', background: dColor, borderRadius: 3 }} />
                                </div>
                                <div style={{ width: 28, fontSize: 8.5, fontFamily: 'monospace', color: dColor, fontWeight: 700, flexShrink: 0, textAlign: 'right' }}>{(prob * 100).toFixed(0)}%</div>
                              </div>
                            </div>

                            {/* Conclusion */}
                            {(() => {
                              const byConv = [...agents].sort((a, b) => Math.abs((b.probability_up||.5)-.5) - Math.abs((a.probability_up||.5)-.5));
                              const strongest = byConv[0];
                              const weakest   = byConv[byConv.length - 1];
                              const dissent   = agents.filter(a => {
                                const v = (a.vote||'HOLD').toUpperCase();
                                return dir === 'LONG' ? v === 'SHORT' : dir === 'SHORT' ? v === 'LONG' : false;
                              }).length;
                              const agreeN = agents.filter(a => (a.vote||'').toUpperCase() === dir).length;
                              return (
                                <div style={{ marginTop: 8, padding: '7px 9px', background: 'rgba(61,158,255,.07)', borderRadius: 6, borderLeft: '3px solid var(--blue)', textAlign: 'left' }}>
                                  <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--blue)', marginBottom: 3 }}>Conclusion</div>
                                  <div style={{ fontSize: 9.5, color: 'var(--dim2)', lineHeight: 1.65 }}>
                                    {agreeN}/{agents.length} agree on <strong style={{ color: dColor }}>{dir}</strong>.{' '}
                                    Highest: <strong>{(strongest?.agent_name||strongest?.name||'?').slice(0,9)}</strong> ({Math.round((strongest?.probability_up||.5)*100)}%).{' '}
                                    Lowest: <strong>{(weakest?.agent_name||weakest?.name||'?').slice(0,9)}</strong> ({Math.round((weakest?.probability_up||.5)*100)}%).{' '}
                                    {dissent > 0
                                      ? <span style={{ color: 'var(--yellow)' }}>{dissent} dissenting.</span>
                                      : <span style={{ color: 'var(--green)' }}>No dissent.</span>}
                                  </div>
                                </div>
                              );
                            })()}
                          </>
                        );
                      })()}
                    </div>
                  );
                })()}
              </div>
            </>
          )}

          {!signal && (
            <div className="card" style={{ padding: 20, textAlign: 'center', color: 'var(--dim)', fontSize: 12, lineHeight: 1.7 }}>
              <div style={{ fontSize: 28, marginBottom: 8 }}>📊</div>
              Run a signal analysis to see live chart, OHLCV data, 52-week range, and market prediction gauge.
            </div>
          )}

        </div>

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
              <div className="empty-text">Click a ticker above or enter a symbol to run deep research across 9 agents and 226+ factors</div>
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
              {(() => {
                const sm = signal.summary && typeof signal.summary === 'object' ? signal.summary : null;
                const headline = sm?.headline ?? (typeof signal.summary === 'string' ? signal.summary : `${dir} signal for ${signal.ticker}`);
                const smColor = sm?.action_color === 'green' ? 'var(--green)' : sm?.action_color === 'red' ? 'var(--red)' : 'var(--yellow)';
                return (
                  <>
                    <div className={`action-card ${dir.toLowerCase() || 'hold'}`}>
                      <div className="action-badge" style={{ color: dColor }}>{dir || 'HOLD'}</div>
                      <div className="action-text">
                        <strong style={{ color: dColor }}>{headline}</strong>
                        {holding.half_life_days != null && (
                          <span style={{ color: 'var(--dim)', fontSize: 11 }}> · Half-life {holding.half_life_days?.toFixed(1)}d</span>
                        )}
                      </div>
                    </div>

                    {sm && (
                      <div className="card" style={{ marginBottom: 14 }}>
                        <div className="card-header">
                          <span className="card-title">📋 Action Plan</span>
                          <span style={{ fontSize: 10, color: 'var(--dim)' }}>{sm.consensus}</span>
                        </div>
                        <div className="card-body" style={{ padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
                          <div style={{ fontSize: 12, color: smColor, fontWeight: 600 }}>{sm.action}</div>

                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                            {sm.bull_case?.length > 0 && (
                              <div>
                                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--green)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '.06em' }}>Bull Case</div>
                                {sm.bull_case.map((pt, i) => (
                                  <div key={i} style={{ fontSize: 11, color: 'var(--dim2)', marginBottom: 3 }}>▲ {pt.replace(/\*\*/g, '')}</div>
                                ))}
                              </div>
                            )}
                            {sm.bear_case?.length > 0 && (
                              <div>
                                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--red)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '.06em' }}>Bear Case</div>
                                {sm.bear_case.map((pt, i) => (
                                  <div key={i} style={{ fontSize: 11, color: 'var(--dim2)', marginBottom: 3 }}>▼ {pt.replace(/\*\*/g, '')}</div>
                                ))}
                              </div>
                            )}
                          </div>

                          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                            {sm.entry_notes?.length > 0 && (
                              <div style={{ flex: 1, minWidth: 180 }}>
                                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--cyan)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '.06em' }}>Entry</div>
                                {sm.entry_notes.map((n, i) => <div key={i} style={{ fontSize: 11, color: 'var(--dim2)' }}>· {n}</div>)}
                              </div>
                            )}
                            {sm.risk_notes?.length > 0 && (
                              <div style={{ flex: 1, minWidth: 180 }}>
                                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--yellow)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '.06em' }}>Risk</div>
                                {sm.risk_notes.map((n, i) => <div key={i} style={{ fontSize: 11, color: 'var(--dim2)' }}>· {n}</div>)}
                              </div>
                            )}
                            {sm.hold_guidance && (
                              <div style={{ flex: 1, minWidth: 180 }}>
                                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--purple)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '.06em' }}>Hold</div>
                                <div style={{ fontSize: 11, color: 'var(--dim2)' }}>· {sm.hold_guidance}</div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                );
              })()}

              {/* Price Range Forecast */}
              {(tickerInfo?.price || signal?.current_price || (tickerInfo?.closes?.length > 0)) && (() => {
                const price    = tickerInfo?.price || signal?.current_price || tickerInfo?.closes?.[tickerInfo.closes.length - 1];
                const hi52     = tickerInfo?.week52_high;
                const lo52     = tickerInfo?.week52_low;
                const annualVol = (hi52 && lo52 && hi52 > lo52)
                  ? Math.log(hi52 / lo52) / 2.7 : 0.28;
                const holdDays = holding.half_life_days ||
                  ({ '1d':1,'1w':5,'1m':22,'3m':65,'6m':130,'1y':252 }[horizon] || 22);
                const periodVol  = annualVol * Math.sqrt(holdDays / 252);
                const bias       = (prob - 0.5) * 2 * periodVol;
                const baseTarget = price * (1 + bias);
                const bullTarget = price * (1 + periodVol + Math.max(bias * 0.5, 0));
                const bearTarget = price * (1 - periodVol + Math.min(bias * 0.5, 0));
                const bullPct    = (bullTarget - price) / price;
                const bearPct    = (bearTarget - price) / price;
                const basePct    = (baseTarget - price) / price;

                const fp  = v => v == null ? '—' : v >= 1000 ? '$' + v.toLocaleString('en-US',{maximumFractionDigits:0}) : v < 1 ? '$' + v.toFixed(4) : '$' + v.toFixed(2);
                const fpc = v => `${v >= 0 ? '+' : ''}${(v*100).toFixed(1)}%`;

                const agentTargets = agents.map(a => {
                  const ap = a.probability_up ?? 0.5;
                  const ab = (ap - 0.5) * 2 * periodVol;
                  return {
                    name: (a.agent_name || a.name || '?').slice(0, 12),
                    vote: (a.vote || 'HOLD').toUpperCase(),
                    target: price * (1 + ab),
                    pct: ab,
                  };
                }).sort((a, b) => b.target - a.target);

                const minT   = Math.min(...agentTargets.map(a => a.target));
                const maxT   = Math.max(...agentTargets.map(a => a.target));
                const spread = maxT - minT;

                const rangeMin   = Math.min(bearTarget, price) * 0.995;
                const rangeMax   = Math.max(bullTarget, price) * 1.005;
                const currentPos = Math.max(0, Math.min(100, (price      - rangeMin) / (rangeMax - rangeMin) * 100));
                const basePos    = Math.max(0, Math.min(100, (baseTarget - rangeMin) / (rangeMax - rangeMin) * 100));

                return (
                  <div className="card" style={{ marginBottom: 14 }}>
                    <div className="card-header">
                      <span className="card-title">🎯 Price Range Forecast</span>
                      <span style={{ fontSize: 10, color: 'var(--dim)' }}>
                        {signal.ticker} · {hzLabel} · ±{(periodVol*100).toFixed(1)}% vol · ~{Math.round(holdDays)}d
                      </span>
                    </div>
                    <div className="card-body" style={{ padding: '12px 14px' }}>

                      {/* Three targets */}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 12 }}>
                        {[
                          { label: 'Bear Target', tp: bearTarget, pct: bearPct, color: 'var(--red)',   icon: '▼' },
                          { label: 'Base (Ensemble)', tp: baseTarget, pct: basePct, color: dColor,    icon: dir === 'LONG' ? '▲' : dir === 'SHORT' ? '▼' : '—' },
                          { label: 'Bull Target', tp: bullTarget, pct: bullPct, color: 'var(--green)', icon: '▲' },
                        ].map(({ label, tp, pct, color, icon }) => (
                          <div key={label} style={{ textAlign: 'center', background: 'var(--card2)', borderRadius: 8, padding: '8px 6px', border: `1px solid ${color}22` }}>
                            <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '.04em' }}>{label}</div>
                            <div style={{ fontSize: 15, fontWeight: 800, fontFamily: 'monospace', color }}>{fp(tp)}</div>
                            <div style={{ fontSize: 10, color, fontWeight: 600, marginTop: 2 }}>{icon} {fpc(pct)}</div>
                          </div>
                        ))}
                      </div>

                      {/* Gradient range bar */}
                      <div style={{ marginBottom: 12 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--dim)', marginBottom: 4 }}>
                          <span>Bear {fp(bearTarget)}</span>
                          <span style={{ color: 'var(--text)', fontWeight: 700 }}>Now {fp(price)}</span>
                          <span>Bull {fp(bullTarget)}</span>
                        </div>
                        <div style={{ height: 10, background: 'linear-gradient(90deg,var(--red),var(--yellow),var(--green))', borderRadius: 5, position: 'relative', opacity: .65 }}>
                          <div style={{ position: 'absolute', left: `${currentPos}%`, top: '50%', transform: 'translate(-50%,-50%)', width: 14, height: 14, borderRadius: '50%', background: 'var(--text)', border: '2px solid var(--card)', boxShadow: '0 1px 5px rgba(0,0,0,.5)' }} />
                          <div style={{ position: 'absolute', left: `${basePos}%`, top: 0, bottom: 0, width: 2, background: dColor, borderRadius: 1 }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: 'var(--dim)', marginTop: 3 }}>
                          <span>Downside</span>
                          <span style={{ color: dColor }}>Base {fp(baseTarget)} ({fpc(basePct)})</span>
                          <span>Upside</span>
                        </div>
                      </div>

                      {/* Per-agent targets */}
                      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 10 }}>
                        <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>
                          Agent Price Targets &nbsp;·&nbsp; Spread: <span style={{ color: 'var(--text)' }}>{fp(spread)}</span>
                        </div>
                        {agentTargets.map(a => {
                          const aColor  = a.vote === 'LONG' ? 'var(--green)' : a.vote === 'SHORT' ? 'var(--red)' : 'var(--yellow)';
                          const barPos  = spread > 0 ? (a.target - minT) / spread * 100 : 50;
                          return (
                            <div key={a.name} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                              <div style={{ width: 76, fontSize: 9, color: 'var(--dim2)', flexShrink: 0 }}>{a.name}</div>
                              <div style={{ flex: 1, height: 5, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                                <div style={{ width: `${Math.max(barPos, 3)}%`, height: '100%', background: aColor, borderRadius: 3, opacity: .8 }} />
                              </div>
                              <div style={{ width: 54, fontSize: 9, fontFamily: 'monospace', color: aColor, flexShrink: 0, textAlign: 'right' }}>{fp(a.target)}</div>
                              <div style={{ width: 40, fontSize: 9, color: aColor, flexShrink: 0, textAlign: 'right' }}>{fpc(a.pct)}</div>
                            </div>
                          );
                        })}
                      </div>

                      <div style={{ marginTop: 10, fontSize: 9, color: 'var(--dim)', lineHeight: 1.6, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                        ⚠ <strong>Probability-weighted ranges</strong>, not price targets. Derived from ±{(periodVol*100).toFixed(1)}% estimated period volatility. Not financial advice.
                      </div>
                    </div>
                  </div>
                );
              })()}

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
                {marketRegime && marketRegime !== 'UNKNOWN' && (() => {
                  const regimeMap = {
                    BULL_TREND:  { label: 'Bull Trend',  color: 'var(--green)',  icon: '📈' },
                    BULL_CHOPPY: { label: 'Bull Choppy', color: 'var(--yellow)', icon: '〰️' },
                    BEAR:        { label: 'Bear',         color: 'var(--red)',    icon: '📉' },
                    CRISIS:      { label: 'Crisis',       color: '#ff4444',       icon: '⚠️' },
                  };
                  const r = regimeMap[marketRegime] || { label: marketRegime, color: 'var(--dim)', icon: '🔍' };
                  return (
                    <div className="metric-card">
                      <div className="metric-lbl">Market Regime</div>
                      <div className="metric-val" style={{ color: r.color, fontSize: 15 }}>{r.icon} {r.label}</div>
                      <div className="metric-sub" style={{ color: 'var(--dim)' }}>Regime-Weighted Fusion</div>
                    </div>
                  );
                })()}
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

              {/* Bayesian Council — convergence path */}
              {council.length > 0 && (
                <div className="card" style={{ marginBottom: 16 }}>
                  <div className="card-header">
                    <span className="card-title">⚖️ Bayesian Council</span>
                    <span style={{ fontSize: 10, color: agreementScore < 0.2 ? 'var(--green)' : agreementScore < 0.4 ? 'var(--yellow)' : 'var(--red)' }}>
                      {agreementScore < 0.2 ? 'Strong consensus' : agreementScore < 0.4 ? 'Moderate agreement' : 'Agents disagree'}
                      &nbsp;· disagreement {(agreementScore * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="card-body" style={{ padding: '12px 14px' }}>
                    <div style={{ fontSize: 10, color: 'var(--dim)', marginBottom: 10 }}>
                      Bayesian prior → posterior path as each agent votes. Wider bar = more influence. Prior: 50.0%
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                      {council.map((step, i) => {
                        const delta = step.delta;
                        const isBull = step.agent_prob > 0.5;
                        const barColor = delta > 0 ? 'var(--green)' : delta < 0 ? 'var(--red)' : 'var(--dim)';
                        const barW = Math.min(Math.abs(delta) * 400, 60); // cap at 60px
                        const agentProb = (step.agent_prob * 100).toFixed(1);
                        const after = (step.prob_after * 100).toFixed(1);
                        return (
                          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <div style={{ width: 90, fontSize: 10, fontWeight: 700, color: 'var(--dim2)', textTransform: 'capitalize', flexShrink: 0 }}>{step.agent}</div>
                            <div style={{ width: 50, fontSize: 10, fontFamily: 'monospace', color: isBull ? 'var(--green)' : 'var(--red)', flexShrink: 0 }}>
                              {agentProb}%
                            </div>
                            <div style={{ flex: 1, height: 8, background: 'var(--border)', borderRadius: 4, position: 'relative', overflow: 'hidden' }}>
                              {/* Posterior fill */}
                              <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${step.prob_after * 100}%`, background: step.prob_after > 0.55 ? 'rgba(0,214,143,.25)' : step.prob_after < 0.45 ? 'rgba(245,54,92,.25)' : 'rgba(255,183,3,.2)', transition: 'width .4s' }} />
                              {/* Delta marker */}
                              <div style={{ position: 'absolute', top: 1, height: 6, width: `${Math.max(barW, 3)}px`, background: barColor, borderRadius: 3, left: delta >= 0 ? `${step.prob_before * 100}%` : `${step.prob_after * 100}%`, opacity: .85 }} />
                              {/* 50% midline */}
                              <div style={{ position: 'absolute', left: '50%', top: 0, width: 1, height: '100%', background: 'var(--dim)', opacity: .4 }} />
                            </div>
                            <div style={{ width: 48, fontSize: 10, fontFamily: 'monospace', color: 'var(--text)', flexShrink: 0, textAlign: 'right' }}>
                              → {after}%
                            </div>
                            <div style={{ width: 36, fontSize: 9, color: barColor, fontFamily: 'monospace', flexShrink: 0, textAlign: 'right' }}>
                              {delta >= 0 ? '+' : ''}{(delta * 100).toFixed(1)}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    {/* Final */}
                    <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 10, paddingTop: 10, borderTop: '1px solid var(--border)' }}>
                      <div style={{ width: 90, fontSize: 10, fontWeight: 800, color: 'var(--text)', flexShrink: 0 }}>FINAL</div>
                      <div style={{ flex: 1, height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${prob * 100}%`, background: prob > 0.55 ? 'var(--green)' : prob < 0.45 ? 'var(--red)' : 'var(--yellow)', transition: 'width .4s', borderRadius: 4 }} />
                      </div>
                      <div style={{ width: 84, fontSize: 11, fontWeight: 800, fontFamily: 'monospace', color: dColor, flexShrink: 0, textAlign: 'right' }}>
                        {(prob * 100).toFixed(1)}% · {dir}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Smart Money Panel ── */}
          {signal && !loading && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-header">
                <span className="card-title">🐋 Smart Money — {signal.ticker}</span>
                <span style={{ fontSize: 10, color: 'var(--dim)' }}>
                  {holdersLoading ? '⏳ Loading…' : holders?.rate_limited ? '⚠️ Rate limited' : holders ? 'Institutional · Mutual Funds · Congressional · Insider' : 'No data'}
                </span>
              </div>

              {holders?.rate_limited && !holdersLoading && (
                <div style={{ padding: '12px 16px', color: 'var(--yellow)', fontSize: 12, background: 'rgba(255,200,0,.06)', borderBottom: '1px solid var(--border)' }}>
                  Yahoo Finance is rate-limiting requests for {signal.ticker}. Smart Money data will load automatically once the limit clears — try again in 30–60 seconds.
                </div>
              )}

              {/* Tab buttons */}
              {holders && !holders.rate_limited && (
                <>
                  <div style={{ display: 'flex', gap: 4, padding: '6px 12px', borderBottom: '1px solid var(--border)' }}>
                    {[
                      { id: 'institutional', label: '🏦 Institutions' },
                      { id: 'funds',         label: '📊 Mutual Funds' },
                      { id: 'congressional', label: '🏛️ Congress' },
                      { id: 'insider',       label: '👤 Insider' },
                    ].map(tab => (
                      <button
                        key={tab.id}
                        onClick={() => setHoldersTab(tab.id)}
                        style={{
                          padding: '4px 10px', fontSize: 11, borderRadius: 6, border: 'none', cursor: 'pointer',
                          background: holdersTab === tab.id ? 'var(--accent)' : 'var(--card2)',
                          color: holdersTab === tab.id ? '#fff' : 'var(--dim2)', fontWeight: holdersTab === tab.id ? 700 : 400,
                        }}
                      >{tab.label}</button>
                    ))}

                    {/* Major holders summary pills */}
                    {Object.keys(holders.major_holders || {}).length > 0 && (
                      <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
                        {Object.entries(holders.major_holders).slice(0, 3).map(([label, val]) => (
                          <span key={label} style={{ fontSize: 10, color: 'var(--dim2)', background: 'var(--card2)', padding: '2px 8px', borderRadius: 10 }}>
                            {label.replace('% held by', '').replace('insiders', 'Insiders').replace('institutions', 'Instit.').trim()}: <strong>{typeof val === 'number' ? val.toFixed(2) + '%' : val}</strong>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="card-body" style={{ padding: '8px 12px', maxHeight: 280, overflowY: 'auto' }}>

                    {/* Institutional holdings */}
                    {holdersTab === 'institutional' && (
                      holders.institutional?.length > 0 ? (
                        <table className="data-table" style={{ width: '100%' }}>
                          <thead><tr>
                            <th style={{ textAlign: 'left' }}>Institution</th>
                            <th style={{ textAlign: 'right' }}>% Held</th>
                            <th style={{ textAlign: 'right' }}>Shares</th>
                            <th style={{ textAlign: 'right' }}>Value ($M)</th>
                          </tr></thead>
                          <tbody>
                            {holders.institutional.map((h, i) => (
                              <tr key={i}>
                                <td style={{ fontSize: 11, color: 'var(--text)' }}>{h.name}</td>
                                <td style={{ textAlign: 'right', fontSize: 11, color: 'var(--cyan)', fontFamily: 'monospace' }}>{h.pct_out != null ? h.pct_out.toFixed(2) + '%' : '—'}</td>
                                <td style={{ textAlign: 'right', fontSize: 11, color: 'var(--dim2)', fontFamily: 'monospace' }}>{h.shares != null ? (h.shares / 1e6).toFixed(2) + 'M' : '—'}</td>
                                <td style={{ textAlign: 'right', fontSize: 11, color: 'var(--dim2)', fontFamily: 'monospace' }}>{h.value_m != null ? '$' + h.value_m.toLocaleString() : '—'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : <div style={{ color: 'var(--dim)', fontSize: 12, padding: 12 }}>No institutional holder data available.</div>
                    )}

                    {/* Mutual fund holdings */}
                    {holdersTab === 'funds' && (
                      holders.mutual_funds?.length > 0 ? (
                        <table className="data-table" style={{ width: '100%' }}>
                          <thead><tr>
                            <th style={{ textAlign: 'left' }}>Fund</th>
                            <th style={{ textAlign: 'right' }}>% Held</th>
                            <th style={{ textAlign: 'right' }}>Shares</th>
                            <th style={{ textAlign: 'right' }}>Value ($M)</th>
                          </tr></thead>
                          <tbody>
                            {holders.mutual_funds.map((h, i) => (
                              <tr key={i}>
                                <td style={{ fontSize: 11, color: 'var(--text)' }}>{h.name}</td>
                                <td style={{ textAlign: 'right', fontSize: 11, color: 'var(--cyan)', fontFamily: 'monospace' }}>{h.pct_out != null ? h.pct_out.toFixed(2) + '%' : '—'}</td>
                                <td style={{ textAlign: 'right', fontSize: 11, color: 'var(--dim2)', fontFamily: 'monospace' }}>{h.shares != null ? (h.shares / 1e6).toFixed(2) + 'M' : '—'}</td>
                                <td style={{ textAlign: 'right', fontSize: 11, color: 'var(--dim2)', fontFamily: 'monospace' }}>{h.value_m != null ? '$' + h.value_m.toLocaleString() : '—'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : <div style={{ color: 'var(--dim)', fontSize: 12, padding: 12 }}>No mutual fund data available.</div>
                    )}

                    {/* Congressional trades */}
                    {holdersTab === 'congressional' && (
                      holders.congressional?.length > 0 ? (
                        <>
                          <div style={{ fontSize: 10, color: 'var(--dim)', marginBottom: 8 }}>{holders.congressional_note}</div>
                          <table className="data-table" style={{ width: '100%' }}>
                            <thead><tr>
                              <th style={{ textAlign: 'left' }}>Politician</th>
                              <th>Chamber</th>
                              <th>Party</th>
                              <th>Transaction</th>
                              <th>Range</th>
                              <th>Date</th>
                            </tr></thead>
                            <tbody>
                              {holders.congressional.map((c, i) => {
                                const isBuy = /purchase|buy/i.test(c.transaction);
                                const isSell = /sale|sell/i.test(c.transaction);
                                const txnColor = isBuy ? 'var(--green)' : isSell ? 'var(--red)' : 'var(--dim2)';
                                const partyColor = c.party === 'R' ? '#f87171' : c.party === 'D' ? '#60a5fa' : 'var(--dim)';
                                return (
                                  <tr key={i}>
                                    <td style={{ fontSize: 11, color: 'var(--text)', fontWeight: 600 }}>{c.politician}</td>
                                    <td style={{ fontSize: 10, color: 'var(--dim2)', textAlign: 'center' }}>{c.chamber}</td>
                                    <td style={{ fontSize: 11, color: partyColor, textAlign: 'center', fontWeight: 700 }}>{c.party}</td>
                                    <td style={{ fontSize: 11, color: txnColor, fontWeight: 600 }}>{c.transaction}</td>
                                    <td style={{ fontSize: 11, color: 'var(--dim2)', fontFamily: 'monospace' }}>{c.range}</td>
                                    <td style={{ fontSize: 10, color: 'var(--dim)' }}>{c.date}</td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </>
                      ) : (holders.congressional_note === 'setup_required' || holders.congressional_note === 'no_data') ? (
                        <div style={{ padding: '16px 12px', color: 'var(--dim)', fontSize: 12 }}>
                          No STOCK Act disclosures found for <strong style={{ color: 'var(--text)' }}>{holders?.ticker}</strong> in Senate records.
                          Congressional members may not have traded this symbol recently, or the disclosure hasn't been filed yet (30-day window).
                        </div>
                      ) : (
                        <div style={{ color: 'var(--dim)', fontSize: 12, padding: 12 }}>
                          {holders.congressional_note || 'No congressional trades on record.'}
                        </div>
                      )
                    )}

                    {/* Insider transactions */}
                    {holdersTab === 'insider' && (
                      holders.insider_transactions?.length > 0 ? (
                        <table className="data-table" style={{ width: '100%' }}>
                          <thead><tr>
                            <th style={{ textAlign: 'left' }}>Name</th>
                            <th style={{ textAlign: 'left' }}>Title</th>
                            <th>Transaction</th>
                            <th style={{ textAlign: 'right' }}>Shares</th>
                            <th style={{ textAlign: 'right' }}>Value ($M)</th>
                            <th>Date</th>
                          </tr></thead>
                          <tbody>
                            {holders.insider_transactions.map((it, i) => {
                              const isBuy = /purchase|buy|acquisition/i.test(it.txn);
                              const isSell = /sale|sell/i.test(it.txn);
                              const txnColor = isBuy ? 'var(--green)' : isSell ? 'var(--red)' : 'var(--dim2)';
                              return (
                                <tr key={i}>
                                  <td style={{ fontSize: 11, color: 'var(--text)', fontWeight: 600 }}>{it.name}</td>
                                  <td style={{ fontSize: 10, color: 'var(--dim)' }}>{it.title}</td>
                                  <td style={{ fontSize: 11, color: txnColor, fontWeight: 600, textAlign: 'center' }}>{it.txn}</td>
                                  <td style={{ textAlign: 'right', fontSize: 11, color: 'var(--dim2)', fontFamily: 'monospace' }}>{it.shares != null ? it.shares.toLocaleString() : '—'}</td>
                                  <td style={{ textAlign: 'right', fontSize: 11, color: 'var(--dim2)', fontFamily: 'monospace' }}>{it.value_m != null ? '$' + it.value_m : '—'}</td>
                                  <td style={{ fontSize: 10, color: 'var(--dim)' }}>{it.date}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      ) : <div style={{ color: 'var(--dim)', fontSize: 12, padding: 12 }}>No recent insider transactions found.</div>
                    )}

                  </div>
                </>
              )}

              {holdersLoading && !holders && (
                <div style={{ padding: 16, color: 'var(--dim)', fontSize: 12, textAlign: 'center' }}>
                  ⏳ Fetching institutional, congressional & insider data…
                </div>
              )}
            </div>
          )}
          {/* Quant Theory Index */}
          <QuantPanel />
        </div>

        {/* AI Chatbot + Volume card */}
        <div className="signal-sidebar">
          <AIAssistant
            ticker={signal?.ticker}
            signalData={signal}
            onAnalyze={(t) => { setTicker(t); runSignal(t, horizon); }}
          />
          <VolumeCard ticker={signal?.ticker} />
        </div>
      </div>
    </div>
  );
}
