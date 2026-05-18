import { useState, useEffect, useRef } from 'react';
import { api } from '../api.js';

const SECTORS = [
  { sym: 'XLK',  label: 'Technology',       color: '#3d9eff' },
  { sym: 'XLF',  label: 'Financials',        color: '#7b5ea7' },
  { sym: 'XLE',  label: 'Energy',            color: '#ff7f32' },
  { sym: 'XLV',  label: 'Health Care',       color: '#00d68f' },
  { sym: 'XLI',  label: 'Industrials',       color: '#06c8f0' },
  { sym: 'XLY',  label: 'Cons. Discret.',    color: '#ffb703' },
  { sym: 'XLP',  label: 'Cons. Staples',     color: '#a8dadc' },
  { sym: 'XLU',  label: 'Utilities',         color: '#e63946' },
  { sym: 'XLB',  label: 'Materials',         color: '#8ecae6' },
  { sym: 'XLRE', label: 'Real Estate',       color: '#f4a261' },
  { sym: 'XLC',  label: 'Comm. Services',    color: '#a7c957' },
];

const NEWS_SOURCES = ['All', 'MarketWatch', 'Reuters Biz', 'CNBC Markets',
  'BBC Business', 'Yahoo Finance', 'Seeking Alpha', 'Bloomberg'];
const POSITIVE = /surge|rally|beat|record|rise|gain|boost|jump|strong|bullish|upgrade|soar|exceed|growth|profit|boom/i;
const NEGATIVE = /crash|fall|drop|plunge|miss|decline|weak|bearish|downgrade|slump|cut|warn|risk|loss|concern|recession|layoff/i;

const sentOf = (t = '') => POSITIVE.test(t) ? 'bullish' : NEGATIVE.test(t) ? 'bearish' : 'neutral';
const sentColor = (s) => s === 'bullish' ? 'var(--green)' : s === 'bearish' ? 'var(--red)' : 'var(--dim)';
const pctColor = (v) => v > 0.3 ? 'var(--green)' : v < -0.3 ? 'var(--red)' : 'var(--yellow)';
const fmtPrice = (p) => p == null ? '—' : p >= 10000 ? p.toLocaleString('en-US', { maximumFractionDigits: 0 }) : p >= 1000 ? p.toLocaleString('en-US', { maximumFractionDigits: 2 }) : p.toFixed(2);
const fmtPct = (v) => v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
const timeSince = (ts) => { if (!ts) return ''; const d = Math.floor(Date.now()/1000 - ts); return d < 60 ? `${d}s` : d < 3600 ? `${Math.floor(d/60)}m` : `${Math.floor(d/3600)}h`; };

// ─── Global Market AI Chat ────────────────────────────────────────────────────

function MarketAIChat({ injectRef, marketContext }) {
  const [input, setInput] = useState('');
  const [msgs, setMsgs] = useState([]);
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState([]);
  const bodyRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [msgs]);

  useEffect(() => {
    if (injectRef) {
      injectRef.current = (text) => {
        setInput(text);
        setTimeout(() => inputRef.current?.focus(), 50);
      };
    }
  }, [injectRef]);

  async function send(q) {
    const question = (q || input).trim();
    if (!question || busy) return;
    setInput('');
    setBusy(true);
    const newHist = [...history, { role: 'user', content: question }];
    setHistory(newHist);
    setMsgs(prev => [...prev, { role: 'user', text: question }]);
    try {
      const res = await api.marketChat(question, newHist, marketContext);
      const reply = res.answer || res.response || '…';
      setHistory([...newHist, { role: 'assistant', content: reply }]);
      setMsgs(prev => [...prev, { role: 'assistant', text: reply }]);
    } catch (err) {
      setMsgs(prev => [...prev, { role: 'assistant', text: `Error: ${err.message}` }]);
    } finally { setBusy(false); }
  }

  const CHIPS = ['Market outlook?', 'Top sectors today?', 'Fed impact?', 'Risk-on or risk-off?', 'What\'s moving markets?'];

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 480 }}>
      <div className="card-header" style={{ flexShrink: 0, padding: '12px 16px' }}>
        <span className="card-title" style={{ fontSize: 13 }}>🌐 Global Market AI</span>
        {busy && <span style={{ fontSize: 10, color: 'var(--blue)', marginLeft: 8 }}>thinking…</span>}
        {!busy && marketContext && (
          <span style={{ fontSize: 9, color: 'var(--green)', marginLeft: 'auto' }}>● live context</span>
        )}
      </div>

      <div ref={bodyRef} style={{ flex: 1, overflowY: 'auto', padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 10, minHeight: 240 }}>
        {msgs.length === 0 && (
          <div style={{ fontSize: 12, color: 'var(--dim)', textAlign: 'center', marginTop: 20, lineHeight: 1.8 }}>
            Ask about markets, sectors, earnings, macro trends — I have live prices and news in context.<br/>
            Click 🤖 on any headline to get an AI summary.
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} style={{
            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '94%',
            background: m.role === 'user' ? 'rgba(61,158,255,.15)' : 'var(--card2)',
            border: `1px solid ${m.role === 'user' ? 'rgba(61,158,255,.3)' : 'var(--border)'}`,
            borderRadius: m.role === 'user' ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
            padding: '9px 12px', fontSize: 12, color: 'var(--text)', lineHeight: 1.65,
          }}
            dangerouslySetInnerHTML={{
              __html: m.text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>')
            }}
          />
        ))}
        {busy && (
          <div style={{ alignSelf: 'flex-start', background: 'var(--card2)', border: '1px solid var(--border)', borderRadius: '12px 12px 12px 4px', padding: '9px 14px', fontSize: 14, color: 'var(--blue)' }}>▌</div>
        )}
      </div>

      <div style={{ flexShrink: 0, padding: '6px 12px', borderTop: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 8 }}>
          {CHIPS.map(q => (
            <button key={q} className="chat-chip" style={{ fontSize: 10 }} onClick={() => send(q)}>{q}</button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            ref={inputRef}
            style={{ flex: 1, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)', padding: '8px 12px', fontSize: 12, outline: 'none' }}
            placeholder="Ask about any market, stock, or macro trend…"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && send()}
          />
          <button className="btn btn-primary" style={{ padding: '8px 16px' }} onClick={() => send()} disabled={busy || !input.trim()}>
            {busy ? '⏳' : '↑ Ask'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Left Panel: Sector Sidebar ───────────────────────────────────────────────

function SectorSidebar({ quotes, onAnalyze }) {
  const sorted = [...SECTORS].sort((a, b) =>
    (quotes[b.sym]?.change_pct ?? 0) - (quotes[a.sym]?.change_pct ?? 0)
  );
  const allPcts = sorted.map(s => quotes[s.sym]?.change_pct ?? 0);
  const maxAbs = Math.max(...allPcts.map(Math.abs), 1);
  return (
    <div className="card">
      <div className="card-header" style={{ padding: '10px 12px' }}>
        <span className="card-title" style={{ fontSize: 12 }}>📊 Sectors</span>
      </div>
      <div>
        {sorted.map(({ sym, label, color }) => {
          const q = quotes[sym];
          const pct = q?.change_pct ?? null;
          const barW = pct != null ? Math.abs(pct) / maxAbs * 100 : 0;
          const pc = pct != null ? pctColor(pct) : 'var(--dim)';
          return (
            <div key={sym}
              onClick={() => onAnalyze?.(sym)}
              title={`Analyze ${sym} — ${label}`}
              style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', transition: 'background .15s' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--card2)'}
              onMouseLeave={e => e.currentTarget.style.background = ''}
            >
              <div style={{ width: 5, height: 5, borderRadius: '50%', background: color, flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                  <span style={{ fontSize: 10, fontWeight: 700 }}>{sym}</span>
                  <span style={{ fontSize: 10, fontWeight: 700, color: pc, fontFamily: 'monospace' }}>
                    {pct != null ? fmtPct(pct) : '—'}
                  </span>
                </div>
                <div style={{ width: '100%', height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ width: `${barW}%`, height: 3, background: pc, borderRadius: 2, transition: 'width .4s' }} />
                </div>
                <div style={{ fontSize: 8, color: 'var(--dim)', marginTop: 1 }}>{label}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Market Price Box ─────────────────────────────────────────────────────────

const INDEX_ETF_MAP = {
  '^GSPC': 'SPY', '^IXIC': 'QQQ', '^DJI': 'DIA', '^RUT': 'IWM',
  '^VIX': null,
  '^FTSE': 'EWU', '^N225': 'EWJ', '^HSI': 'EWH',
  '^GDAXI': 'EWG', '^FCHI': 'EWQ', '^STOXX50E': 'EZU',
  '^NSEI': 'INDA', '^BSESN': 'INDA', '^AXJO': 'EWA',
};

function MarketCard({ item, onAnalyze }) {
  const pct = item.change_pct ?? 0;
  const color = pctColor(pct);
  const isPos = pct >= 0;
  const analysisSymbol = item.symbol?.startsWith('^')
    ? (INDEX_ETF_MAP[item.symbol] ?? null)
    : item.symbol;
  const canAnalyze = !!onAnalyze && !!analysisSymbol;
  return (
    <div
      onClick={() => canAnalyze && onAnalyze(analysisSymbol)}
      title={canAnalyze ? `Click to analyze ${analysisSymbol} (${item.label}) in Signal tab` : item.symbol?.startsWith('^') ? `${item.label} — not directly tradeable` : undefined}
      style={{
        background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10,
        padding: '12px 14px', borderTop: `2px solid ${color}`,
        transition: 'transform .15s, border-color .2s, box-shadow .15s',
        cursor: canAnalyze ? 'pointer' : 'default', position: 'relative',
      }}
      onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.borderColor = color; if (canAnalyze) e.currentTarget.style.boxShadow = `0 4px 16px ${color}28`; }}
      onMouseLeave={e => { e.currentTarget.style.transform = ''; e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.boxShadow = ''; }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text)', display: 'flex', alignItems: 'center', gap: 5 }}>
            {item.label}
            {canAnalyze && <span style={{ fontSize: 8, color: 'var(--dim)', opacity: .5 }} title={analysisSymbol !== item.symbol ? `→ ${analysisSymbol}` : ''}>📡</span>}
          </div>
          <div style={{ fontSize: 9, color: 'var(--dim)', marginTop: 2 }}>{item.symbol}</div>
        </div>
        <span style={{
          fontSize: 9, padding: '2px 6px', borderRadius: 8, fontWeight: 700,
          background: isPos ? 'rgba(0,214,143,.12)' : 'rgba(245,54,92,.12)', color,
        }}>{isPos ? '▲' : '▼'} {Math.abs(pct).toFixed(2)}%</span>
      </div>
      <div style={{ fontSize: 19, fontWeight: 800, fontFamily: 'monospace', color: 'var(--text)', letterSpacing: '-.01em' }}>
        {fmtPrice(item.price)}
      </div>
      <div style={{ fontSize: 10, color, fontFamily: 'monospace', marginTop: 2 }}>
        {isPos ? '+' : ''}{(item.change ?? 0).toFixed(2)} today
      </div>
      <div style={{ marginTop: 8, height: 3, background: 'var(--border)', borderRadius: 2 }}>
        <div style={{ width: `${Math.min(50 + pct * 4, 100)}%`, height: 3, background: color, borderRadius: 2, minWidth: 4 }} />
      </div>
    </div>
  );
}

function MarketSection({ title, items, cols, onAnalyze }) {
  if (!items?.length) return null;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--dim)', marginBottom: 8, letterSpacing: '.06em', textTransform: 'uppercase' }}>{title}</div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols || items.length}, 1fr)`, gap: 8 }}>
        {items.map(item => <MarketCard key={item.symbol} item={item} onAnalyze={onAnalyze} />)}
      </div>
    </div>
  );
}

// ─── Earnings Calendar ────────────────────────────────────────────────────────

function EarningsCalendar({ earnings, loading }) {
  return (
    <div className="card" style={{ marginBottom: 14 }}>
      <div className="card-header">
        <span className="card-title">📅 Upcoming Earnings</span>
        {!loading && <span style={{ fontSize: 10, color: 'var(--dim)', marginLeft: 8 }}>{earnings.length} in next 30 days</span>}
      </div>
      {loading ? (
        <div style={{ padding: '18px 14px', display: 'flex', gap: 10, alignItems: 'center', color: 'var(--dim)', fontSize: 12 }}>
          <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Loading earnings…
        </div>
      ) : earnings.length === 0 ? (
        <div style={{ padding: '16px 14px', color: 'var(--dim)', fontSize: 12 }}>No upcoming earnings found.</div>
      ) : (
        <div style={{ overflowX: 'auto', maxHeight: 260, overflowY: 'auto' }}>
          <table className="data-table">
            <thead><tr><th>Ticker</th><th>Date</th><th>In</th><th>Price</th><th>Chg%</th><th>EPS Est.</th></tr></thead>
            <tbody>
              {earnings.map((e, i) => {
                const pc = (e.change_pct ?? 0) >= 0 ? 'var(--green)' : 'var(--red)';
                const urg = e.days_until === 0 ? { bg: 'rgba(255,180,0,.15)', c: 'var(--yellow)' }
                          : e.days_until <= 3  ? { bg: 'rgba(245,54,92,.1)', c: 'var(--red)' }
                          : { bg: 'var(--card2)', c: 'var(--dim)' };
                return (
                  <tr key={i}>
                    <td style={{ fontWeight: 700 }}>{e.ticker}</td>
                    <td style={{ fontSize: 11, color: 'var(--dim)' }}>{e.earnings_date}</td>
                    <td><span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 10, fontWeight: 700, background: urg.bg, color: urg.c }}>{e.days_until === 0 ? 'Today' : `${e.days_until}d`}</span></td>
                    <td style={{ fontFamily: 'monospace' }}>{e.price != null ? `$${fmtPrice(e.price)}` : '—'}</td>
                    <td style={{ color: pc, fontFamily: 'monospace', fontWeight: 600 }}>{fmtPct(e.change_pct)}</td>
                    <td style={{ color: 'var(--dim2)', fontFamily: 'monospace' }}>{e.eps_estimate != null ? `$${e.eps_estimate}` : '—'}</td>
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

// ─── News Feed ────────────────────────────────────────────────────────────────

function NewsFeed({ news, loading, onAsk }) {
  const [srcFilter, setSrcFilter]   = useState('All');
  const [sentFilter, setSentFilter] = useState('all');
  const [summaries, setSummaries]   = useState({});
  const [summarizing, setSummarizing] = useState({});

  const filtered = news.filter(n => {
    if (srcFilter !== 'All' && n.publisher !== srcFilter) return false;
    if (sentFilter !== 'all' && sentOf(n.title) !== sentFilter) return false;
    return true;
  });

  const counts = { all: news.length, bullish: 0, bearish: 0, neutral: 0 };
  news.forEach(n => { counts[sentOf(n.title)]++; });

  async function summarizeNews(n, key) {
    if (summaries[key] || summarizing[key]) return;
    setSummarizing(s => ({ ...s, [key]: true }));
    try {
      const prompt = `In 2-3 sentences, summarize this financial news and its market impact. Start your first sentence with exactly one of: BULLISH, BEARISH, or NEUTRAL. Headline: "${n.title}"`;
      const res = await api.marketChat(prompt, [], null);
      const text = res.answer || res.response || '';
      const upper = text.toUpperCase();
      const sentiment = upper.startsWith('BULLISH') ? 'bullish' : upper.startsWith('BEARISH') ? 'bearish' : 'neutral';
      setSummaries(s => ({ ...s, [key]: { text, sentiment } }));
      onAsk?.(`Summarize this news and its market impact: "${n.title}"`);
    } catch {
      setSummaries(s => ({ ...s, [key]: { text: 'Failed to summarize.', sentiment: 'neutral' } }));
    } finally {
      setSummarizing(s => ({ ...s, [key]: false }));
    }
  }

  return (
    <div className="card">
      <div className="card-header" style={{ flexWrap: 'wrap', gap: 8 }}>
        <span className="card-title">📰 Market News</span>
        <div style={{ display: 'flex', gap: 5, marginLeft: 'auto', flexWrap: 'wrap', alignItems: 'center' }}>
          {[{ k: 'all', label: `All (${counts.all})`, color: 'var(--dim2)' },
            { k: 'bullish', label: `▲ ${counts.bullish}`, color: 'var(--green)' },
            { k: 'bearish', label: `▼ ${counts.bearish}`, color: 'var(--red)' }].map(s => (
            <button key={s.k} onClick={() => setSentFilter(s.k)} style={{
              fontSize: 10, padding: '3px 8px', borderRadius: 6, border: 'none', cursor: 'pointer', fontWeight: 600,
              background: sentFilter === s.k ? s.color : 'var(--card2)',
              color: sentFilter === s.k ? '#fff' : s.color, transition: 'all .15s',
            }}>{s.label}</button>
          ))}
          <select value={srcFilter} onChange={e => setSrcFilter(e.target.value)}
            style={{ fontSize: 10, background: 'var(--card2)', color: 'var(--dim2)', border: '1px solid var(--border)', borderRadius: 6, padding: '3px 7px', outline: 'none' }}>
            {NEWS_SOURCES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </div>
      {loading ? (
        <div style={{ padding: '18px 14px', display: 'flex', gap: 10, alignItems: 'center', color: 'var(--dim)', fontSize: 12 }}>
          <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Fetching latest news…
        </div>
      ) : (
        <div style={{ maxHeight: 520, overflowY: 'auto' }}>
          {filtered.length === 0 ? (
            <div style={{ padding: 24, textAlign: 'center', color: 'var(--dim)', fontSize: 12 }}>No news matching this filter</div>
          ) : filtered.map((n, i) => {
            const sent = sentOf(n.title);
            const sc   = sentColor(sent);
            const key  = `${n.published_at}-${i}`;
            const sum  = summaries[key];
            const busy = summarizing[key];
            const sumColor = sum?.sentiment === 'bullish' ? 'var(--green)' : sum?.sentiment === 'bearish' ? 'var(--red)' : 'var(--dim2)';
            return (
              <div key={i} style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', gap: 10 }}>
                  <div style={{ width: 3, flexShrink: 0, alignSelf: 'stretch', background: sc, borderRadius: 2, opacity: .7 }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <a href={n.link} target="_blank" rel="noopener noreferrer"
                      style={{ fontSize: 12, color: 'var(--text)', fontWeight: 500, lineHeight: 1.4, display: 'block', marginBottom: 5, textDecoration: 'none' }}
                      onMouseEnter={e => e.target.style.color = 'var(--blue)'}
                      onMouseLeave={e => e.target.style.color = 'var(--text)'}>{n.title}</a>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                      <span style={{ fontSize: 9, color: 'var(--blue)', fontWeight: 700 }}>{n.publisher}</span>
                      {n.ticker && <span style={{ fontSize: 9, color: 'var(--dim)', background: 'var(--card2)', padding: '1px 5px', borderRadius: 4 }}>{n.ticker}</span>}
                      <span style={{ fontSize: 9, color: sc, fontWeight: 600 }}>{sent === 'bullish' ? '▲ Bullish' : sent === 'bearish' ? '▼ Bearish' : '— Neutral'}</span>
                      <span style={{ fontSize: 9, color: 'var(--dim)', marginLeft: 'auto' }}>{timeSince(n.published_at)} ago</span>
                    </div>
                  </div>
                  <button
                    title={sum ? 'Click to collapse summary' : 'Summarize with AI'}
                    onClick={() => sum ? setSummaries(s => { const c = { ...s }; delete c[key]; return c; }) : summarizeNews(n, key)}
                    disabled={busy}
                    style={{ flexShrink: 0, background: sum ? 'rgba(61,158,255,.12)' : 'none', border: `1px solid ${sum ? 'rgba(61,158,255,.3)' : 'var(--border)'}`, borderRadius: 6, padding: '3px 8px', fontSize: 10, color: sum ? 'var(--blue)' : 'var(--dim)', cursor: busy ? 'wait' : 'pointer', alignSelf: 'flex-start', transition: 'all .15s', fontWeight: 600 }}
                    onMouseEnter={e => { if (!busy) { e.currentTarget.style.background = 'var(--card2)'; e.currentTarget.style.color = 'var(--blue)'; }}}
                    onMouseLeave={e => { if (!sum) { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = 'var(--dim)'; }}}>
                    {busy ? '⏳' : sum ? '✕ AI' : '🤖 Summarize'}
                  </button>
                </div>

                {/* Inline AI summary */}
                {busy && (
                  <div style={{ marginTop: 8, marginLeft: 13, padding: '8px 10px', background: 'var(--card2)', borderRadius: 6, fontSize: 11, color: 'var(--blue)', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div className="spinner" style={{ width: 10, height: 10, borderWidth: 1.5 }} /> Analyzing…
                  </div>
                )}
                {sum && !busy && (
                  <div style={{ marginTop: 8, marginLeft: 13, padding: '9px 11px', background: 'var(--card2)', borderRadius: 7, border: `1px solid ${sumColor}30`, position: 'relative' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
                      <span style={{ fontSize: 9, fontWeight: 800, padding: '2px 8px', borderRadius: 8, letterSpacing: '.05em', background: `${sumColor}18`, color: sumColor, border: `1px solid ${sumColor}30` }}>
                        {sum.sentiment === 'bullish' ? '▲ BULLISH' : sum.sentiment === 'bearish' ? '▼ BEARISH' : '— NEUTRAL'}
                      </span>
                      <span style={{ fontSize: 9, color: 'var(--dim)' }}>AI Summary</span>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--dim2)', lineHeight: 1.6 }}>{sum.text}</div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── Main Market Tab ──────────────────────────────────────────────────────────

export default function MarketTab({ isActive, onAnalyze }) {
  const [summary, setSummary]   = useState({ us: [], global: [], assets: [], fx: [] });
  const [sectors, setSectors]   = useState({});
  const [news, setNews]         = useState([]);
  const [earnings, setEarnings] = useState([]);
  const [loadingMkt, setLoadingMkt]  = useState(true);
  const [loadingNews, setLoadingNews] = useState(true);
  const [loadingErn, setLoadingErn]   = useState(true);
  const [lastRefresh, setLastRefresh] = useState(null);
  const timerRef   = useRef(null);
  const hasLoaded  = useRef(false);   // prevents reload on tab switch
  const injectRef  = useRef(null);

  useEffect(() => {
    // Only load data once on first activation — not on every tab switch
    if (isActive && !hasLoaded.current) {
      hasLoaded.current = true;
      loadAll();
    }
    if (isActive) {
      timerRef.current = setInterval(loadMarket, 60000);
    }
    return () => clearInterval(timerRef.current);
  }, [isActive]);

  async function loadMarket() {
    try {
      const [s, sec] = await Promise.all([api.marketSummary(), api.marketQuotes(SECTORS.map(x => x.sym))]);
      setSummary(s); setSectors(sec.quotes || {}); setLastRefresh(new Date());
    } catch {} finally { setLoadingMkt(false); }
  }

  async function loadAll() {
    setLoadingMkt(true); setLoadingNews(true); setLoadingErn(true);
    loadMarket();
    api.marketNews(50).then(d => setNews(d.news || [])).catch(() => setNews([])).finally(() => setLoadingNews(false));
    api.marketEarnings().then(d => setEarnings(d.earnings || [])).catch(() => setEarnings([])).finally(() => setLoadingErn(false));
  }

  // Build market context for the AI chatbot
  const marketContext = {
    us_markets: summary.us.map(m => `${m.label} (${m.symbol}): $${fmtPrice(m.price)} ${fmtPct(m.change_pct)}`).join(', '),
    global_markets: summary.global.map(m => `${m.label}: $${fmtPrice(m.price)} ${fmtPct(m.change_pct)}`).join(', '),
    assets: summary.assets.map(m => `${m.label}: $${fmtPrice(m.price)} ${fmtPct(m.change_pct)}`).join(', '),
    fx: summary.fx.map(m => `${m.label}: ${fmtPrice(m.price)} ${fmtPct(m.change_pct)}`).join(', '),
    sectors_up: Object.entries(sectors).filter(([,q]) => (q?.change_pct ?? 0) > 0).map(([sym]) => sym).join(', '),
    sectors_down: Object.entries(sectors).filter(([,q]) => (q?.change_pct ?? 0) < 0).map(([sym]) => sym).join(', '),
    upcoming_earnings: earnings.slice(0, 8).map(e => `${e.ticker} in ${e.days_until}d`).join(', '),
    news_count: news.length,
    bullish_news: news.filter(n => sentOf(n.title) === 'bullish').length,
    bearish_news: news.filter(n => sentOf(n.title) === 'bearish').length,
  };

  const allItems = [...summary.us, ...summary.global];
  const bullishSectors = Object.values(sectors).filter(q => (q?.change_pct ?? 0) > 0).length;

  return (
    <div style={{ maxWidth: 1700, margin: '0 auto' }}>
      {/* Header strip */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 14, flexWrap: 'wrap', alignItems: 'center' }}>
        {[
          { label: 'Markets Up', val: allItems.length ? `${allItems.filter(m => (m.change_pct ?? 0) > 0).length}/${allItems.length}` : '—', color: 'var(--green)' },
          { label: 'Sectors Up', val: `${bullishSectors}/${SECTORS.length}`, color: bullishSectors >= 6 ? 'var(--green)' : bullishSectors >= 4 ? 'var(--yellow)' : 'var(--red)' },
          { label: 'Earnings', val: loadingErn ? '…' : earnings.length, color: 'var(--cyan)' },
          { label: 'News', val: loadingNews ? '…' : news.length, color: 'var(--blue)' },
        ].map(m => (
          <div key={m.label} className="metric-card" style={{ minWidth: 100 }}>
            <div className="metric-lbl">{m.label}</div>
            <div className="metric-val" style={{ color: m.color, fontSize: 16 }}>{m.val}</div>
          </div>
        ))}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          {lastRefresh && <span style={{ fontSize: 10, color: 'var(--dim)' }}>{lastRefresh.toLocaleTimeString()}</span>}
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--green)', display: 'inline-block', animation: 'pulse 2s infinite' }} />
          <button className="btn btn-sm" onClick={loadAll}>↻ Refresh</button>
        </div>
      </div>

      {/* Three-column layout: Sector LEFT | Market CENTER | Earnings+AI RIGHT */}
      <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>

        {/* ── LEFT: Sector Sidebar (220px sticky) ─── */}
        <div style={{ width: 220, flexShrink: 0, position: 'sticky', top: 0, maxHeight: '92vh', overflowY: 'auto' }}>
          <SectorSidebar quotes={sectors} onAnalyze={onAnalyze} />
        </div>

        {/* ── CENTER: Market Data + News ─── */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {loadingMkt ? (
            <div className="spinner-wrap"><div className="spinner" /><div className="spinner-text">Loading market data…</div></div>
          ) : (
            <>
              <MarketSection title="🇺🇸 US Markets" items={summary.us} cols={summary.us.length} onAnalyze={onAnalyze} />
              <MarketSection title="🌏 Global Markets" items={summary.global} cols={4} onAnalyze={onAnalyze} />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 14 }}>
                <MarketSection title="₿ Crypto & Commodities" items={summary.assets} cols={3} onAnalyze={onAnalyze} />
                <MarketSection title="💱 FX Rates" items={summary.fx} cols={3} onAnalyze={onAnalyze} />
              </div>
            </>
          )}
          <NewsFeed news={news} loading={loadingNews} onAsk={text => injectRef.current?.(text)} />
        </div>

        {/* ── RIGHT: Earnings + Global Market AI (480px sticky) ─── */}
        <div style={{ width: 480, flexShrink: 0, position: 'sticky', top: 0, maxHeight: '92vh', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14 }}>
          <EarningsCalendar earnings={earnings} loading={loadingErn} />
          <div style={{ flex: 1, minHeight: 480 }}>
            <MarketAIChat injectRef={injectRef} marketContext={loadingMkt ? null : marketContext} />
          </div>
        </div>
      </div>
    </div>
  );
}
