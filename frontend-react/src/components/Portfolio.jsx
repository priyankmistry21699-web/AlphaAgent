import { useState, useEffect, useRef } from 'react';
import { api } from '../api.js';

const fmtUSD  = (v, dec = 0) => v == null ? '—' : `$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: dec, minimumFractionDigits: dec })}`;
const fmtPct  = (v, dec = 2) => v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(dec)}%`;
const pnlClr  = (v) => v >= 0 ? 'var(--green)' : 'var(--red)';
const renderMd = (t = '') => t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');

const todayISO = () => new Date().toISOString().slice(0, 10);

function isMarketOpen() {
  const now = new Date();
  const et  = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const day = et.getDay();
  const h   = et.getHours(), m = et.getMinutes();
  const mins = h * 60 + m;
  return day >= 1 && day <= 5 && mins >= 570 && mins < 960;
}

function detectRegion() {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
    if (tz.startsWith('America'))  return 'us';
    if (tz.includes('Kolkata') || tz.includes('Calcutta') || tz.includes('Colombo')) return 'india';
    if (tz.startsWith('Europe') || tz.includes('London') || tz.includes('Dublin')) return 'europe';
    if (tz === 'Asia/Tokyo') return 'japan';
    if (tz === 'Asia/Shanghai' || tz === 'Asia/Hong_Kong' || tz === 'Asia/Chongqing') return 'china';
    if (['Asia/Seoul','Asia/Singapore','Asia/Taipei','Asia/Jakarta',
         'Asia/Kuala_Lumpur','Pacific/Auckland','Australia/Sydney']
        .some(z => tz === z)) return 'asia';
  } catch {}
  return 'global';
}

const PORTFOLIO_CHIPS = [
  'Summarize my portfolio',
  'When should I sell?',
  'What is my biggest risk?',
  'Rebalancing advice',
  'How am I performing?',
  'Exit strategy for each position',
];

// ─── Portfolio AI Assistant ───────────────────────────────────────────────────

function PortfolioAIAssistant({ livePositions, budget, entryDate, entryTime, exitDate, exitTime, hasLive, onReset, externalQuery, onExternalQueryConsumed }) {
  const [msgs, setMsgs]       = useState([]);
  const [input, setInput]     = useState('');
  const [busy, setBusy]       = useState(false);
  const [history, setHistory] = useState([]);
  const bodyRef  = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [msgs]);

  useEffect(() => {
    if (externalQuery) {
      onExternalQueryConsumed?.();
      send(externalQuery);
    }
  }, [externalQuery]);

  async function send(q) {
    const question = (q || input).trim();
    if (!question || busy) return;
    setInput('');
    setBusy(true);
    const newHist = [...history, { role: 'user', content: question }];
    setHistory(newHist);
    setMsgs(prev => [...prev, { role: 'user', text: question }, { role: 'ai', text: null }]);

    try {
      const fullEntry = entryDate + (entryTime ? `T${entryTime}` : '');
      const fullExit  = exitDate ? exitDate + (exitTime ? `T${exitTime}` : '') : '';
      const res = await api.portfolioChat(question, newHist, livePositions || [], budget, fullEntry, fullExit || undefined);
      const answer = res.answer || '…';
      setHistory([...newHist, { role: 'assistant', content: answer }]);
      setMsgs(prev => {
        const next = [...prev];
        next[next.length - 1] = { role: 'ai', text: answer };
        return next;
      });
    } catch (err) {
      setMsgs(prev => {
        const next = [...prev];
        next[next.length - 1] = { role: 'ai', text: `Error: ${err.message}` };
        return next;
      });
    } finally { setBusy(false); }
  }

  function clearAll() {
    setHistory([]);
    setMsgs([]);
    onReset?.();
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ padding: '10px 14px 8px', borderBottom: '1px solid var(--border)', flexShrink: 0, background: 'var(--card2)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 28, height: 28, borderRadius: '50%', background: 'linear-gradient(135deg,#3d9eff,#7b5ea7)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, fontWeight: 800, color: '#fff', flexShrink: 0 }}>α</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>Portfolio AI</div>
            <div style={{ fontSize: 10, color: 'var(--dim)' }}>Advisor · Gemini + AlphaAgent</div>
          </div>
          {msgs.length > 0 && (
            <button onClick={clearAll} style={{ fontSize: 10, color: 'var(--dim)', background: 'none', border: '1px solid var(--border)', borderRadius: 6, padding: '3px 8px', cursor: 'pointer', flexShrink: 0 }}>✕ Clear</button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div ref={bodyRef} style={{ flex: 1, overflowY: 'auto', padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {msgs.length === 0 && (
          <div style={{ textAlign: 'center', padding: '20px 14px', color: 'var(--dim)', fontSize: 11, lineHeight: 1.9 }}>
            <div style={{ fontSize: 28, marginBottom: 8, opacity: .5 }}>α</div>
            <div style={{ color: 'var(--dim2)', fontWeight: 700, marginBottom: 10, fontSize: 12 }}>Portfolio AI Advisor</div>

            {hasLive ? (
              <>
                {/* Advisor mode — portfolio exists */}
                <div style={{ marginBottom: 14, padding: '10px 12px', background: 'rgba(0,196,118,.07)', border: '1px solid rgba(0,196,118,.2)', borderRadius: 8, fontSize: 10, color: 'var(--green)', lineHeight: 1.8, textAlign: 'left' }}>
                  <div style={{ fontWeight: 800, marginBottom: 4, fontSize: 10.5, letterSpacing: '.04em' }}>YOUR PORTFOLIO IS LIVE</div>
                  <div>Ask me about your positions, P&amp;L, or when to exit</div>
                  <div>I can see all your live data — ask anything specific</div>
                  <div>To add/remove stocks → use <strong>Signal Scan</strong> on the left</div>
                </div>
                <div style={{ textAlign: 'left', fontSize: 10, color: 'var(--dim)', lineHeight: 1.6 }}>
                  <div style={{ fontWeight: 700, color: 'var(--dim2)', marginBottom: 5, fontSize: 10, letterSpacing: '.04em' }}>TRY ASKING:</div>
                  {[
                    'Summarize my portfolio performance',
                    'When should I sell each position?',
                    'What is my biggest risk right now?',
                    'Which position should I trim first?',
                    'Give me exit price targets for all',
                  ].map(q => (
                    <div key={q} onClick={() => send(q)}
                      style={{ cursor: 'pointer', padding: '4px 8px', marginBottom: 3, borderRadius: 6, background: 'var(--surface)', border: '1px solid var(--border)', fontSize: 10 }}
                      onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(0,196,118,.4)'}
                      onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
                    >"{q}"</div>
                  ))}
                </div>
              </>
            ) : (
              <>
                {/* No portfolio yet — guide to Signal Scan */}
                <div style={{ marginBottom: 14, padding: '10px 12px', background: 'rgba(61,158,255,.08)', border: '1px solid rgba(61,158,255,.2)', borderRadius: 8, fontSize: 10, color: 'var(--blue)', lineHeight: 1.8, textAlign: 'left' }}>
                  <div style={{ fontWeight: 800, marginBottom: 4, fontSize: 10.5, letterSpacing: '.04em' }}>HOW TO BUILD YOUR PORTFOLIO</div>
                  <div>① Wait for <strong>CORE READY</strong> — top stocks warm in ~2 min</div>
                  <div>② Set <strong>Market · Asset Type · Budget</strong> in the filters</div>
                  <div>③ Hit <strong>▶ Run Signals</strong> — AlphaAgent scans live signals</div>
                  <div>④ Pick stocks using <strong>+ Add</strong>, then <strong>▶ Activate</strong></div>
                </div>
                <div style={{ textAlign: 'left', fontSize: 10, color: 'var(--dim)', lineHeight: 1.6 }}>
                  <div style={{ fontWeight: 700, color: 'var(--dim2)', marginBottom: 5, fontSize: 10, letterSpacing: '.04em' }}>OR ASK ME FOR ANALYSIS:</div>
                  {[
                    'Which US tech stocks look strong?',
                    'What sectors should I focus on?',
                    'Explain momentum vs value investing',
                  ].map(q => (
                    <div key={q} onClick={() => send(q)}
                      style={{ cursor: 'pointer', padding: '4px 8px', marginBottom: 3, borderRadius: 6, background: 'var(--surface)', border: '1px solid var(--border)', fontSize: 10 }}
                      onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(61,158,255,.4)'}
                      onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
                    >"{q}"</div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {msgs.map((m, i) => {
          if (m.role === 'user') return (
            <div key={i} style={{ alignSelf: 'flex-end', maxWidth: '88%', background: 'rgba(61,158,255,.15)', border: '1px solid rgba(61,158,255,.3)', borderRadius: '12px 12px 4px 12px', padding: '9px 13px', fontSize: 12, color: 'var(--text)', lineHeight: 1.6 }}>
              {m.text}
            </div>
          );
          if (m.text === null) return (
            <div key={i} style={{ alignSelf: 'flex-start', background: 'var(--card2)', border: '1px solid var(--border)', borderRadius: '12px 12px 12px 4px', padding: '9px 14px', fontSize: 14, color: 'var(--blue)' }}>▌</div>
          );
          return (
            <div key={i} style={{ alignSelf: 'flex-start', maxWidth: '98%' }}>
              <div style={{ background: 'var(--card2)', border: '1px solid var(--border)', borderRadius: '12px 12px 12px 4px', padding: '9px 13px', fontSize: 12, color: 'var(--text)', lineHeight: 1.65 }}
                dangerouslySetInnerHTML={{ __html: renderMd(m.text || '') }} />
            </div>
          );
        })}
      </div>

      {/* Quick chips */}
      <div style={{ padding: '6px 12px', borderTop: '1px solid var(--border)', flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 6 }}>
          {PORTFOLIO_CHIPS.map(c => (
            <button key={c} onClick={() => send(c)} disabled={busy}
              style={{ fontSize: 9, padding: '3px 8px', borderRadius: 20, border: '1px solid var(--border)', background: 'var(--card2)', color: 'var(--dim2)', cursor: busy ? 'wait' : 'pointer', transition: 'all .15s' }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--blue)'; e.currentTarget.style.color = 'var(--blue)'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--dim2)'; }}
            >{c}</button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <textarea ref={inputRef} rows={1} value={input}
            onChange={e => { setInput(e.target.value); e.target.style.height = ''; e.target.style.height = Math.min(e.target.scrollHeight, 72) + 'px'; }}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder={hasLive ? 'Ask about your positions, P&L, exit timing…' : 'Ask for market analysis or portfolio advice…'}
            disabled={busy}
            style={{ flex: 1, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)', padding: '8px 12px', fontSize: 12, outline: 'none', resize: 'none', fontFamily: 'inherit', lineHeight: 1.4 }}
          />
          <button onClick={() => send()} disabled={busy || !input.trim()}
            style={{ padding: '8px 14px', borderRadius: 8, border: 'none', background: busy ? 'var(--card2)' : 'var(--blue)', color: '#fff', cursor: busy ? 'wait' : 'pointer', fontSize: 14, fontWeight: 700, flexShrink: 0 }}>
            {busy ? '⏳' : '↑'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Live Position Row (compact table row + expandable detail) ────────────────

function PositionCard({ pos, idx, onSell, onBuyMore }) {
  const colors = ['var(--blue)', 'var(--green)', 'var(--purple)', 'var(--yellow)', 'var(--cyan)', '#f4a261', '#a8dadc', 'var(--red)'];
  const ac     = colors[idx % colors.length];
  const pnl    = pos.pnl ?? 0;
  const pnlPct = pos.pnl_pct ?? 0;
  const pc     = pnlClr(pnl);
  const entry  = pos.entry_price ?? 0;
  const cur    = pos.current_price ?? entry;
  const target = pos.target_price;
  const stop   = pos.stop_loss;
  const mval   = pos.market_value ?? 0;
  const alloc  = pos.allocated ?? 0;
  const riskPct   = entry > 0 && stop   ? ((entry - stop)   / entry * 100).toFixed(1) : null;
  const rewardPct = entry > 0 && target ? ((target - entry) / entry * 100).toFixed(1) : null;
  const toTarget  = target && entry > 0 ? Math.min(Math.max((cur - entry) / (target - entry) * 100, 0), 100) : 0;

  let entryDate = '—', entryTime = '—';
  if (pos.entry_datetime) {
    const d = new Date(pos.entry_datetime);
    entryDate = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    entryTime = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
  }

  const [open, setOpen]       = useState(false);
  const [buying, setBuying]   = useState(false);
  const [buyAmt, setBuyAmt]   = useState('');
  const [selling, setSelling] = useState(false);

  async function handleSell() {
    setSelling(true);
    try { await onSell(pos.ticker); } finally { setSelling(false); }
  }
  async function handleBuy() {
    const amt = parseFloat(buyAmt);
    if (!amt || amt <= 0) return;
    setBuying(true);
    try { await onBuyMore(pos.ticker, amt); setBuyAmt(''); } finally { setBuying(false); }
  }

  return (
    <div style={{ borderBottom: '1px solid var(--border)' }}>
      {/* ── Compact row ── */}
      <div onClick={() => setOpen(o => !o)}
        style={{ display: 'grid', gridTemplateColumns: '3px 110px 1fr 100px 80px 38px', alignItems: 'center', padding: '10px 14px', cursor: 'pointer', gap: 10, userSelect: 'none',
          background: open ? 'rgba(255,255,255,.03)' : 'transparent', transition: 'background .15s' }}
        onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,.03)'}
        onMouseLeave={e => e.currentTarget.style.background = open ? 'rgba(255,255,255,.03)' : 'transparent'}>

        {/* left accent bar */}
        <div style={{ width: 3, height: 36, borderRadius: 2, background: ac, alignSelf: 'center' }} />

        {/* ticker + name + price */}
        <div style={{ overflow: 'hidden' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 1 }}>
            <span style={{ fontSize: 13, fontWeight: 800, color: ac, fontFamily: 'monospace' }}>{pos.ticker}</span>
            <span style={{ fontSize: 9, padding: '1px 5px', borderRadius: 4, fontWeight: 700,
              background: (pos.trade_type || 'LONG') === 'LONG' ? 'rgba(34,197,94,.15)' : 'rgba(239,68,68,.15)',
              color: (pos.trade_type || 'LONG') === 'LONG' ? 'var(--green)' : 'var(--red)' }}>
              {pos.trade_type || 'LONG'}
            </span>
          </div>
          <div style={{ fontSize: 9, color: 'var(--dim2)', fontFamily: 'monospace' }}>${cur.toFixed(2)}</div>
          <div style={{ fontSize: 8, color: 'var(--dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{pos.name}</div>
        </div>

        {/* P&L + progress bar */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
            <span style={{ fontSize: 12, fontWeight: 800, color: pc, fontFamily: 'monospace' }}>{pnl >= 0 ? '+' : '−'}{fmtUSD(Math.abs(pnl))}</span>
            <span style={{ fontSize: 10, fontWeight: 700, color: pc }}>{pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%</span>
          </div>
          <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: 4, width: `${toTarget}%`, background: `linear-gradient(90deg,${ac},var(--green))`, borderRadius: 2 }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: 'var(--dim)', marginTop: 2 }}>
            <span>In ${entry.toFixed(0)}</span>
            {target && <span style={{ color: 'var(--green)' }}>Tgt ${target.toFixed(0)} {rewardPct ? `+${rewardPct}%` : ''}</span>}
          </div>
        </div>

        {/* invested + mkt val */}
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text)', fontFamily: 'monospace' }}>{fmtUSD(mval)}</div>
          <div style={{ fontSize: 8, color: 'var(--dim)', marginTop: 1 }}>mkt val</div>
          <div style={{ fontSize: 8, color: 'var(--dim)' }}>{fmtUSD(alloc)} in · {pos.pct ?? 0}%</div>
        </div>

        {/* today % */}
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 12, fontWeight: 700, fontFamily: 'monospace',
            color: (pos.change_pct ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {pos.change_pct != null ? `${pos.change_pct >= 0 ? '+' : ''}${pos.change_pct.toFixed(2)}%` : '—'}
          </div>
          <div style={{ fontSize: 8, color: 'var(--dim)', marginTop: 1 }}>today</div>
          {stop && <div style={{ fontSize: 8, color: 'var(--red)', marginTop: 2 }}>Stop ${stop.toFixed(0)}</div>}
        </div>

        {/* expand */}
        <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--dim)' }}>{open ? '▲' : '▼'}</div>
      </div>

      {/* ── Expanded detail panel ── */}
      {open && (
        <div style={{ padding: '10px 14px 12px', background: 'rgba(0,0,0,.15)', borderTop: '1px solid var(--border)' }}>
          {/* stat grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6,1fr)', gap: '6px 12px', marginBottom: 10 }}>
            {[
              ['Trade Type',  pos.trade_type ?? '—',              'var(--dim2)'],
              ['Shares',      pos.shares?.toFixed(4) ?? '—',      'var(--text)'],
              ['Entry',       `$${entry.toFixed(2)}`,             'var(--text)'],
              ['Current',     `$${cur.toFixed(2)}`,               pc],
              ['Target',      target ? `$${target.toFixed(2)}` : '—', 'var(--green)'],
              ['Stop Loss',   stop   ? `$${stop.toFixed(2)}`   : '—', 'var(--red)'],
              ['Entry Date',  entryDate,                          'var(--dim2)'],
              ['Entry Time',  entryTime,                          'var(--dim2)'],
              ['Exit Date',   pos.exit_date ?? '—',               'var(--yellow)'],
              ['Horizon',     pos.hold_horizon ?? '—',            'var(--dim2)'],
              ['Invested',    fmtUSD(alloc),                      'var(--blue)'],
              ['Mkt Value',   fmtUSD(mval),                       'var(--dim2)'],
            ].map(([l, v, c]) => (
              <div key={l}>
                <div style={{ fontSize: 8, color: 'var(--dim)', marginBottom: 2 }}>{l}</div>
                <div style={{ fontSize: 11, fontWeight: 700, color: c, fontFamily: 'monospace' }}>{v}</div>
              </div>
            ))}
          </div>

          {/* thesis / exit rationale */}
          {(pos.thesis || pos.exit_rationale) && (
            <div style={{ display: 'flex', gap: 12, marginBottom: 10, fontSize: 9 }}>
              {pos.thesis         && <div style={{ color: 'var(--dim)' }}><span style={{ color: 'var(--dim2)', fontWeight: 700 }}>Thesis: </span>{pos.thesis}</div>}
              {pos.exit_rationale && <div style={{ color: 'var(--yellow)' }}><span style={{ fontWeight: 700 }}>Exit: </span>{pos.exit_rationale}</div>}
            </div>
          )}

          {/* sell / buy more */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button onClick={handleSell} disabled={selling}
              style={{ padding: '5px 14px', borderRadius: 7, border: '1px solid rgba(239,68,68,.4)', background: 'rgba(239,68,68,.12)', color: 'var(--red)', fontSize: 11, fontWeight: 700, cursor: selling ? 'wait' : 'pointer' }}>
              {selling ? '…' : '✕ Sell All'}
            </button>
            <input type="number" value={buyAmt} onChange={e => setBuyAmt(e.target.value)}
              placeholder="$ add more"
              style={{ width: 110, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, color: 'var(--text)', padding: '5px 8px', fontSize: 11, outline: 'none', fontFamily: 'monospace' }}
              onKeyDown={e => { if (e.key === 'Enter') handleBuy(); }} />
            <button onClick={handleBuy} disabled={buying || !buyAmt}
              style={{ padding: '5px 14px', borderRadius: 7, border: `1px solid ${ac}55`, background: `${ac}22`, color: ac, fontSize: 11, fontWeight: 700, cursor: (buying || !buyAmt) ? 'not-allowed' : 'pointer' }}>
              {buying ? '…' : '+ Buy More'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}


// ─── Main Portfolio Tab ───────────────────────────────────────────────────────

// ─── Asset Search Autocomplete ────────────────────────────────────────────────
const ASSET_TYPES = [
  { id: 'all',         label: 'All Types' },
  { id: 'stocks',      label: 'Stocks' },
  { id: 'etfs',        label: 'ETFs' },
  { id: 'mutual_fund', label: 'Funds' },
  { id: 'index',       label: 'Indices' },
  { id: 'commodities', label: 'Commodities' },
  { id: 'forex',       label: 'Forex' },
  { id: 'crypto',      label: 'Crypto' },
];
const REGIONS = [
  { id: 'all',    label: 'All Markets' },
  { id: 'us',     label: 'US' },
  { id: 'india',  label: 'India' },
  { id: 'europe', label: 'Europe' },
  { id: 'asia',   label: 'Asia' },
  { id: 'japan',  label: 'Japan' },
  { id: 'china',  label: 'China' },
  { id: 'global', label: 'Global' },
];

function AssetSearch({ region, assetType, onSelect, compact }) {
  const [query, setQuery]     = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen]       = useState(false);
  const debounce              = useRef(null);
  const containerRef          = useRef(null);

  useEffect(() => {
    clearTimeout(debounce.current);
    if (query.trim().length < 1) { setResults([]); setOpen(false); return; }
    debounce.current = setTimeout(async () => {
      setLoading(true);
      try {
        const d = await api.marketSearch(query.trim(), region, assetType, 10);
        setResults(d.results || []);
        setOpen(true);
      } catch {} finally { setLoading(false); }
    }, 250);
    return () => clearTimeout(debounce.current);
  }, [query, region, assetType]);

  useEffect(() => {
    function onClick(e) { if (!containerRef.current?.contains(e.target)) setOpen(false); }
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, []);

  const dirClr = (d) => d === 'LONG' ? 'var(--green)' : d === 'SHORT' ? 'var(--red)' : 'var(--dim)';

  return (
    <div ref={containerRef} style={{ position: 'relative', minWidth: 0, ...(compact ? {} : { flex: 1 }) }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: compact ? 4 : 8, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, padding: compact ? '5px 8px' : '6px 12px' }}>
        <span style={{ fontSize: compact ? 10 : 13, color: 'var(--dim)', flexShrink: 0 }}>🔍</span>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onFocus={() => results.length > 0 && setOpen(true)}
          placeholder={compact ? 'Ticker / name…' : 'Search stocks, ETFs, indices, crypto, forex…'}
          style={{ flex: 1, background: 'none', border: 'none', color: 'var(--text)', fontSize: compact ? 10 : 12, outline: 'none', fontFamily: 'monospace', minWidth: 0 }}
        />
        {loading && <span style={{ fontSize: 9, color: 'var(--blue)', flexShrink: 0 }}>…</span>}
        {query && <button onClick={() => { setQuery(''); setResults([]); setOpen(false); }}
          style={{ background: 'none', border: 'none', color: 'var(--dim)', cursor: 'pointer', fontSize: 11, padding: 0, flexShrink: 0, lineHeight: 1 }}>✕</button>}
      </div>

      {open && results.length > 0 && (
        <div style={{ position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0, zIndex: 999,
          background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10,
          boxShadow: '0 8px 32px rgba(0,0,0,.4)', overflow: 'hidden' }}>

          {/* "You are searching for…" header */}
          <div style={{ padding: '7px 14px', background: 'rgba(61,158,255,.08)', borderBottom: '1px solid var(--border)', fontSize: 10, color: 'var(--blue)', fontWeight: 600, letterSpacing: '.04em' }}>
            You are searching for "{query}"
          </div>

          {results.map((r, i) => (
            <div key={r.ticker}
              onClick={() => { onSelect(r); setQuery(''); setOpen(false); }}
              style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 14px', cursor: 'pointer', borderBottom: i < results.length - 1 ? '1px solid var(--border)' : 'none', transition: 'background .1s' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--card2)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>

              {/* Signal dot */}
              {r.has_signal && (
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: dirClr(r.signal_dir), flexShrink: 0 }} title={`Signal: ${r.signal_dir}`} />
              )}
              {!r.has_signal && <span style={{ width: 6, height: 6, flexShrink: 0 }} />}

              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                  <span style={{ fontSize: 13, fontWeight: 800, color: 'var(--text)', fontFamily: 'monospace' }}>{r.ticker}</span>
                  <span style={{ fontSize: 10, color: 'var(--dim2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name}</span>
                </div>
                <div style={{ display: 'flex', gap: 8, marginTop: 1 }}>
                  <span style={{ fontSize: 9, color: 'var(--dim)', textTransform: 'capitalize' }}>{r.type}</span>
                  {r.sector && <span style={{ fontSize: 9, color: 'var(--dim)' }}>· {r.sector}</span>}
                  {r.regions?.length > 0 && <span style={{ fontSize: 9, color: 'var(--dim)' }}>· {r.regions.join(', ')}</span>}
                </div>
              </div>

              <div style={{ textAlign: 'right', flexShrink: 0, display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 }}>
                {r.price != null && <div style={{ fontSize: 12, fontWeight: 700, fontFamily: 'monospace', color: 'var(--text)' }}>${r.price.toFixed(2)}</div>}
                {r.change_pct != null && (
                  <div style={{ fontSize: 9, color: r.positive ? 'var(--green)' : 'var(--red)' }}>
                    {r.positive ? '▲' : '▼'} {Math.abs(r.change_pct).toFixed(2)}%
                  </div>
                )}
                {r.has_signal && (
                  <div style={{ fontSize: 8, color: dirClr(r.signal_dir), fontWeight: 700 }}>{r.signal_dir}</div>
                )}
                <div style={{ fontSize: 9, color: 'var(--blue)', fontWeight: 700, marginTop: 2, padding: '2px 7px', border: '1px solid rgba(61,158,255,.4)', borderRadius: 5, background: 'rgba(61,158,255,.08)' }}>
                  + Add →
                </div>
              </div>
            </div>
          ))}

          <div style={{ padding: '5px 14px', background: 'var(--card2)', borderTop: '1px solid var(--border)', fontSize: 9, color: 'var(--dim)' }}>
            Click any row to instantly add it to Portfolio AI · Green dot = signal ready
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main Portfolio Tab ───────────────────────────────────────────────────────

export default function PortfolioTab({ isActive }) {
  const [budget, setBudget]       = useState(100000);
  const [entryDate, setEntryDate] = useState(todayISO());
  const [entryTime, setEntryTime] = useState('09:30');
  const [exitDate, setExitDate]   = useState('');
  const [exitTime, setExitTime]   = useState('16:00');
  const [filterRegion, setFilterRegion]   = useState('all');
  const [filterType,   setFilterType]     = useState('all');
  const [liveData, setLiveData]   = useState(null);
  const [loadingLive, setLoadingLive] = useState(false);
  const [activating, setActivating]   = useState(false);
  const [activeView, setActiveView]   = useState('live');
  const [trades, setTrades]           = useState([]);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [marketOpen, setMarketOpen]   = useState(isMarketOpen());
  const [warmup, setWarmup]           = useState({ quotes_pct: 0, signals_pct: 0, total_tickers: 0, quotes_ready: 0, signals_ready: 0, priority_pct: 0, priority_ready: 0, priority_total: 0 });
  const [aiChatQuery, setAiChatQuery] = useState('');
  const [scanResults, setScanResults] = useState(null);
  const [scanning, setScanning]       = useState(false);
  const [autoScan, setAutoScan]       = useState(false);
  const [scanError, setScanError]     = useState('');
  const [discarded, setDiscarded]     = useState(new Set());
  const [selectedPicks, setSelectedPicks] = useState(new Set());
  const [scanMeta, setScanMeta]       = useState(null);  // {region, type, budget, entry, exit}
  const refreshTimer  = useRef(null);
  const marketTimer   = useRef(null);
  const warmupTimer   = useRef(null);
  const [scanComplete, setScanComplete] = useState(false);
  const scanInterval  = useRef(null);
  const scanRunning   = useRef(false);
  const ENOUGH_PICKS  = 25;

  // Strategy Builder state
  const [strategyResult,  setStrategyResult]  = useState(null);
  const [strategyLoading, setStrategyLoading] = useState(false);
  const [strategyError,   setStrategyError]   = useState('');
  const [strategyMode,    setStrategyMode]    = useState('auto');
  const [strategyCapital, setStrategyCapital] = useState(100000);

  function fetchWarmup() {
    api.warmupStatus().then(d => {
      setWarmup(d);
      // Reschedule: fast while warming, slow once done
      clearInterval(warmupTimer.current);
      const done = d.quotes_pct >= 100 && d.signals_pct >= 100;
      warmupTimer.current = setInterval(fetchWarmup, done ? 30000 : 2000);
    }).catch(() => {});
  }

  useEffect(() => {
    // Detect region from timezone and tell the backend to prioritize it
    const region = detectRegion();
    api.setRegion(region).catch(() => {});
  }, []);

  useEffect(() => {
    if (isActive) {
      fetchLive();
      fetchWarmup();
      api.trades().then(d => setTrades(Array.isArray(d) ? d : d.trades || [])).catch(() => {});
    }
    return () => clearInterval(warmupTimer.current);
  }, [isActive]);

  // Auto-refresh live positions every 5s
  useEffect(() => {
    clearInterval(refreshTimer.current);
    if (isActive && liveData?.positions?.length > 0) {
      refreshTimer.current = setInterval(fetchLive, 5000);
    }
    return () => clearInterval(refreshTimer.current);
  }, [isActive, liveData?.positions?.length, marketOpen]);

  // Update market-open status every minute
  useEffect(() => {
    marketTimer.current = setInterval(() => setMarketOpen(isMarketOpen()), 60000);
    return () => clearInterval(marketTimer.current);
  }, []);

  // Clean up scan auto-refresh on unmount
  useEffect(() => () => clearInterval(scanInterval.current), []);

  async function fetchLive() {
    setLoadingLive(true);
    try {
      const d = await api.aiPortfolioLive();
      setLiveData(d);
      setLastRefresh(new Date());
    } catch (err) {
      console.warn('ai-portfolio/live error:', err.message);
    } finally { setLoadingLive(false); }
  }

  async function handleActivate(positions, cap) {
    setActivating(true);
    try {
      await api.aiPortfolioCommit(positions, cap);
      await fetchLive();
      setActiveView('live');
    } finally { setActivating(false); }
  }

  async function handleSell(ticker) {
    await api.aiPortfolioSell(ticker);
    await fetchLive();
  }

  async function handleBuyMore(ticker, amount) {
    await api.aiPortfolioBuyMore(ticker, amount);
    await fetchLive();
  }

  async function handleReset() {
    await api.aiPortfolioReset();
    setLiveData(null);
    setScanResults(null);
    setScanMeta(null);
    setScanError('');
    setDiscarded(new Set());
    setSelectedPicks(new Set());
    setActiveView('live');
  }

  async function buildStrategy() {
    setStrategyLoading(true);
    setStrategyError('');
    try {
      // Pass region + asset_type so non-US markets get their own universe
      const result = await api.strategyBuild({
        capital:    strategyCapital,
        mode:       strategyMode,
        region:     filterRegion !== 'all' ? filterRegion : undefined,
        asset_type: filterType   !== 'all' ? filterType   : undefined,
      });
      setStrategyResult(result);
    } catch (err) {
      setStrategyError(`Build failed: ${err.message}`);
    } finally {
      setStrategyLoading(false);
    }
  }

  function sendScanToAI(data) {
    if (!data || data.picks.length === 0) return;
    const pickList = data.picks.slice(0, 20).map(p =>
      `${p.ticker} (${p.name}, ${p.direction}, prob=${Math.round(p.probability * 100)}%, price=$${p.price ? p.price.toFixed(2) : '?'})`
    ).join('; ');
    setAiChatQuery(
      `Signal scan found ${data.picks.length} signals. Budget $${(budget / 1000).toFixed(0)}k.\n` +
      `Picks: ${pickList}.\n` +
      `For each stock give me: entry price, target price, stop loss price, exit date, and 1-line reasoning. ` +
      `Format as a clean table or list. Be specific with prices and dates.`
    );
  }

  function stopAutoScan(finalData) {
    clearInterval(scanInterval.current);
    scanInterval.current = null;
    setAutoScan(false);
    if (finalData && finalData.picks.length >= ENOUGH_PICKS) {
      setScanComplete(true);
      sendScanToAI(finalData);
    } else if (finalData) {
      sendScanToAI(finalData);
    }
  }

  async function doScan() {
    if (scanRunning.current) return null;
    scanRunning.current = true;
    setScanning(true);
    try {
      const ownedTickers = (liveData?.positions ?? []).map(p => p.ticker);
      const data = await api.portfolioScan({ region: filterRegion, asset_type: filterType, budget, exclude: ownedTickers });
      setScanResults(data);
      if (data.picks.length >= ENOUGH_PICKS) {
        stopAutoScan(data);
      }
      return data;
    } catch (err) {
      setScanError(`Scan failed: ${err.message}. Check that the server is running.`);
      stopAutoScan(null);
      return null;
    } finally {
      setScanning(false);
      scanRunning.current = false;
    }
  }

  async function runScan() {
    clearInterval(scanInterval.current);
    setScanResults(null);
    setScanError('');
    setScanComplete(false);
    setDiscarded(new Set());
    setSelectedPicks(new Set());
    setScanMeta({ region: filterRegion, assetType: filterType, budget, entryDate, entryTime, exitDate, exitTime });
    setAutoScan(true);
    const first = await doScan();
    if (first && first.picks.length < ENOUGH_PICKS) {
      scanInterval.current = setInterval(doScan, 5000);
    }
  }

  const summary   = liveData?.summary ?? {};
  const positions = liveData?.positions ?? [];
  const hasLive   = positions.length > 0;

  const pnl     = summary.total_pnl ?? 0;
  const retPct  = summary.return_pct ?? 0;
  const pnlC    = pnlClr(pnl);

  const METRICS = [
    { label: 'Portfolio Value', val: fmtUSD(summary.portfolio_value || 0),             color: 'var(--text)' },
    { label: 'Total P&L',       val: `${pnl >= 0 ? '+' : ''}${fmtUSD(pnl)}`,          color: pnlC },
    { label: 'Return %',        val: fmtPct(retPct),                                   color: pnlC },
    { label: 'Positions',       val: summary.positions_count ?? positions.length,       color: 'var(--cyan)' },
    { label: 'Win Rate',        val: `${((summary.win_rate ?? 0) * 100).toFixed(1)}%`, color: 'var(--blue)' },
    { label: 'Sharpe Ratio',    val: (summary.sharpe ?? 0).toFixed(2),                 color: 'var(--purple)' },
  ];

  const warmupDone     = warmup.quotes_pct >= 100 && warmup.signals_pct >= 100;
  const priorityDone   = warmup.priority_pct >= 100;
  const warmupActive   = warmup.total_tickers > 0 && !warmupDone;

  return (
    <div style={{ maxWidth: 1920, margin: '0 auto', display: 'flex', gap: 12, alignItems: 'flex-start' }}>

      {/* ══ LEFT: Filter panel (sticky) ══ */}
      <div style={{ width: 215, flexShrink: 0, position: 'sticky', top: 16, alignSelf: 'flex-start', display: 'flex', flexDirection: 'column', gap: 8 }}>

        {/* Pre-loading / warmup status */}
        {warmup.total_tickers > 0 && (
          <div style={{ padding: '8px 10px', background: warmupDone ? 'rgba(0,196,118,.06)' : priorityDone ? 'rgba(0,196,118,.04)' : 'rgba(61,158,255,.06)', border: `1px solid ${warmupDone ? 'rgba(0,196,118,.2)' : priorityDone ? 'rgba(0,196,118,.15)' : 'rgba(61,158,255,.2)'}`, borderRadius: 8 }}>
            <div style={{ fontSize: 10, fontWeight: 800, color: warmupDone ? 'var(--green)' : priorityDone ? 'var(--green)' : 'var(--blue)', letterSpacing: '.05em', marginBottom: 5 }}>
              {warmupDone ? '✓ DATA READY' : priorityDone ? '✓ CORE READY' : '⟳ PRE-LOADING'}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: 'var(--dim)', marginBottom: 2 }}>
                  <span>Core signals</span><span style={{ fontFamily: 'monospace' }}>{warmup.priority_ready}/{warmup.priority_total}</span>
                </div>
                <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ height: 3, width: `${warmup.priority_pct}%`, background: priorityDone ? 'var(--green)' : 'var(--blue)', borderRadius: 2, transition: 'width .5s' }} />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: 'var(--dim)', marginBottom: 2 }}>
                  <span>All signals</span><span style={{ fontFamily: 'monospace' }}>{warmup.signals_ready}/{warmup.total_signable ?? warmup.total_tickers}</span>
                </div>
                <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ height: 3, width: `${warmup.signals_pct}%`, background: 'var(--purple)', borderRadius: 2, transition: 'width .5s' }} />
                </div>
              </div>
            </div>
            {warmupActive && !priorityDone && <div style={{ fontSize: 8, color: 'var(--dim)', marginTop: 4 }}>Warming core signals (~2 min)…</div>}
            {!warmupDone && priorityDone && <div style={{ fontSize: 8, color: 'var(--green)', marginTop: 4 }}>Top {warmup.priority_total} stocks ready</div>}
          </div>
        )}

        {/* Signal Scan controls */}
        <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 10px', display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: 10, fontWeight: 800, color: 'var(--dim2)', letterSpacing: '.06em' }}>SIGNAL SCAN</div>

          <div>
            <div style={{ fontSize: 8, color: 'var(--dim)', letterSpacing: '.05em', marginBottom: 3 }}>MARKET</div>
            <select value={filterRegion} onChange={e => setFilterRegion(e.target.value)}
              style={{ width: '100%', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, color: 'var(--text)', fontSize: 11, padding: '5px 8px', cursor: 'pointer', outline: 'none' }}>
              {REGIONS.map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
            </select>
          </div>

          <div>
            <div style={{ fontSize: 8, color: 'var(--dim)', letterSpacing: '.05em', marginBottom: 3 }}>ASSET TYPE</div>
            <select value={filterType} onChange={e => setFilterType(e.target.value)}
              style={{ width: '100%', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, color: 'var(--text)', fontSize: 11, padding: '5px 8px', cursor: 'pointer', outline: 'none' }}>
              {ASSET_TYPES.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
            </select>
          </div>

          <div>
            <div style={{ fontSize: 8, color: 'var(--dim)', letterSpacing: '.05em', marginBottom: 3 }}>BUDGET</div>
            <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, padding: '5px 8px', display: 'flex', alignItems: 'center', gap: 3 }}>
              <span style={{ fontSize: 10, color: 'var(--dim)' }}>$</span>
              <input type="number" value={budget} onChange={e => setBudget(Math.max(1000, Number(e.target.value)))}
                style={{ background: 'none', border: 'none', color: 'var(--text)', fontSize: 11, fontFamily: 'monospace', fontWeight: 700, outline: 'none', width: '100%' }} />
            </div>
          </div>

          <div>
            <div style={{ fontSize: 8, color: 'var(--dim)', letterSpacing: '.05em', marginBottom: 3 }}>SEARCH ASSET</div>
            <AssetSearch region={filterRegion} assetType={filterType} onSelect={() => {}} compact />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
            <div>
              <div style={{ fontSize: 8, color: 'var(--dim)', letterSpacing: '.05em', marginBottom: 3 }}>ENTRY DATE</div>
              <input type="date" value={entryDate} onChange={e => setEntryDate(e.target.value)}
                style={{ width: '100%', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, color: 'var(--text)', fontSize: 9, fontFamily: 'monospace', padding: '5px 4px', outline: 'none', cursor: 'pointer', boxSizing: 'border-box' }} />
            </div>
            <div>
              <div style={{ fontSize: 8, color: 'var(--dim)', letterSpacing: '.05em', marginBottom: 3 }}>TIME</div>
              <input type="time" value={entryTime} onChange={e => setEntryTime(e.target.value)}
                style={{ width: '100%', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, color: 'var(--text)', fontSize: 9, fontFamily: 'monospace', padding: '5px 4px', outline: 'none', cursor: 'pointer', boxSizing: 'border-box' }} />
            </div>
            <div>
              <div style={{ fontSize: 8, color: 'var(--dim)', letterSpacing: '.05em', marginBottom: 3 }}>EXIT DATE <span style={{ color: 'var(--blue)', fontSize: 7 }}>(opt)</span></div>
              <input type="date" value={exitDate} onChange={e => setExitDate(e.target.value)}
                style={{ width: '100%', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, color: exitDate ? 'var(--text)' : 'var(--dim)', fontSize: 9, fontFamily: 'monospace', padding: '5px 4px', outline: 'none', cursor: 'pointer', boxSizing: 'border-box' }} />
            </div>
            <div>
              <div style={{ fontSize: 8, color: 'var(--dim)', letterSpacing: '.05em', marginBottom: 3 }}>TIME</div>
              <input type="time" value={exitTime} onChange={e => setExitTime(e.target.value)}
                style={{ width: '100%', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, color: 'var(--text)', fontSize: 9, fontFamily: 'monospace', padding: '5px 4px', outline: 'none', cursor: 'pointer', boxSizing: 'border-box' }} />
            </div>
          </div>

          {/* Run / Stop */}
          {autoScan ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '5px 8px', borderRadius: 7, background: 'rgba(61,158,255,.12)', border: '1px solid rgba(61,158,255,.3)', fontSize: 9, color: 'var(--blue)' }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', border: '1.5px solid var(--blue)', borderTopColor: 'transparent', animation: 'spin 1s linear infinite', flexShrink: 0 }} />
                {scanResults ? `${scanResults.picks.length}/${ENOUGH_PICKS} best found` : 'searching…'}
              </div>
              <button onClick={() => stopAutoScan(scanResults)}
                style={{ padding: '7px', borderRadius: 8, border: '1px solid rgba(239,68,68,.4)', background: 'rgba(239,68,68,.12)', color: 'var(--red)', fontSize: 11, fontWeight: 700, cursor: 'pointer' }}>
                ■ Stop
              </button>
              <button onClick={runScan}
                style={{ padding: '7px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--dim2)', fontSize: 11, fontWeight: 700, cursor: 'pointer' }}>
                ↺ Restart
              </button>
            </div>
          ) : (
            <button onClick={runScan} disabled={scanning}
              style={{ padding: '9px', borderRadius: 8, border: 'none', background: scanning ? 'var(--card2)' : 'linear-gradient(135deg,#3d9eff,#7b5ea7)', color: '#fff', fontSize: 12, fontWeight: 800, cursor: scanning ? 'wait' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7, transition: 'opacity .15s' }}>
              {scanning ? <><div style={{ width: 11, height: 11, border: '2px solid rgba(255,255,255,.3)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} /> Scanning…</> : <>▶ Run Signals</>}
            </button>
          )}

          {scanError && <div style={{ fontSize: 9, color: 'var(--red)', padding: '5px 8px', background: 'rgba(239,68,68,.1)', border: '1px solid rgba(239,68,68,.3)', borderRadius: 6 }}>{scanError}</div>}
          {scanResults && scanResults.picks.length === 0 && <div style={{ fontSize: 9, color: 'var(--yellow)', padding: '5px 8px', background: 'rgba(234,179,8,.08)', border: '1px solid rgba(234,179,8,.3)', borderRadius: 6 }}>Warming up ({scanResults.signals_cached}/{scanResults.total_scanned})</div>}
          {scanComplete && (
            <div style={{ padding: '7px 8px', background: 'rgba(0,196,118,.1)', border: '1px solid rgba(0,196,118,.35)', borderRadius: 7 }}>
              <div style={{ fontSize: 10, fontWeight: 800, color: 'var(--green)' }}>✓ {scanResults?.picks?.length} signals found!</div>
              <div style={{ fontSize: 8, color: 'var(--dim2)', marginTop: 2 }}>Ranked by conviction. Use + in the scan panel.</div>
              <button onClick={() => setScanComplete(false)} style={{ fontSize: 8, color: 'var(--dim)', background: 'none', border: 'none', cursor: 'pointer', padding: '2px 0 0' }}>Dismiss</button>
            </div>
          )}
        </div>
      </div>

      {/* ══ CENTER: Portfolio ══ */}
      <div style={{ flex: 1, minWidth: 0 }}>

        {/* Metrics bar */}
        <div className="metrics-row" style={{ marginBottom: 16 }}>
          {METRICS.map(m => (
            <div key={m.label} className="metric-card">
              <div className="metric-lbl">{m.label}</div>
              <div className="metric-val" style={{ color: m.color, fontSize: 18 }}>{m.val}</div>
            </div>
          ))}
        </div>

        {/* ── Signal Scan Results ── */}
        {scanResults && (() => {
          const visible  = scanResults.picks.filter(p => !discarded.has(p.ticker));
          const longs    = visible.filter(p => p.direction === 'LONG');
          const shorts   = visible.filter(p => p.direction === 'SHORT');
          const fmtM     = v => v >= 1e6 ? `$${(v/1e6).toFixed(1)}M` : v >= 1e3 ? `$${(v/1e3).toFixed(0)}k` : `$${v}`;
          const totalScore = visible.reduce((s, p) => s + (p.score || 0.02), 0) || 1;
          const picksAlloc = visible.map(p => ({ ...p, _pct: Math.max(5, Math.min(25, ((p.score || 0.02) / totalScore) * 100)) }));
          const totalRaw   = picksAlloc.reduce((s, p) => s + p._pct, 0) || 1;
          const picks = picksAlloc.map(p => ({
            ...p,
            alloc_pct: parseFloat((p._pct / totalRaw * 100).toFixed(1)),
            alloc_amt: Math.round((p._pct / totalRaw) * budget),
          }));
          return (
            <div style={{ marginBottom: 14, background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, overflow: 'hidden' }}>
              {/* header */}
              <div style={{ padding: '9px 14px', background: 'var(--card2)', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontSize: 12, fontWeight: 800 }}>Signal Scan Results</span>
                <span style={{ fontSize: 10, color: 'var(--green)', fontWeight: 700 }}>{longs.length} LONG</span>
                <span style={{ fontSize: 10, color: 'var(--red)', fontWeight: 700 }}>{shorts.length} SHORT</span>
                <span style={{ fontSize: 9, color: 'var(--dim)' }}>{visible.length} picks</span>
                <span style={{ fontSize: 9, color: 'var(--dim)', marginLeft: 'auto' }}>{scanResults.signals_cached}/{scanResults.total_scanned} cached</span>
                <button onClick={() => { stopAutoScan(null); setScanResults(null); setDiscarded(new Set()); setSelectedPicks(new Set()); }}
                  style={{ fontSize: 12, color: 'var(--dim)', background: 'none', border: 'none', cursor: 'pointer', padding: '0 4px', lineHeight: 1 }}>✕</button>
              </div>
              {/* grid of pick rows */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 1, background: 'var(--border)' }}>
                {visible.length === 0
                  ? <div style={{ padding: 20, textAlign: 'center', color: 'var(--dim)', fontSize: 11, background: 'var(--card)', gridColumn: '1/-1' }}>All picks discarded</div>
                  : picks.map((p, i) => {
                    const isSel = selectedPicks.has(p.ticker);
                    return (
                      <div key={p.ticker} style={{ padding: '9px 12px', background: isSel ? 'rgba(0,196,118,.05)' : 'var(--card)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 4 }}>
                          <span style={{ fontSize: 12, fontWeight: 800, fontFamily: 'monospace', color: p.direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>{p.ticker}</span>
                          <span style={{ fontSize: 9, padding: '1px 5px', borderRadius: 4, fontWeight: 700, background: p.direction === 'LONG' ? 'rgba(0,196,118,.15)' : 'rgba(239,68,68,.15)', color: p.direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>{p.direction === 'LONG' ? 'BUY' : 'SHORT'}</span>
                          <span style={{ fontSize: 9, color: 'var(--dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{p.name}</span>
                          <button onClick={() => setSelectedPicks(prev => { const n = new Set(prev); isSel ? n.delete(p.ticker) : n.add(p.ticker); return n; })}
                            style={{ fontSize: 10, padding: '2px 7px', borderRadius: 4, border: `1px solid ${isSel ? 'rgba(0,196,118,.6)' : 'rgba(0,196,118,.35)'}`, background: isSel ? 'rgba(0,196,118,.2)' : 'rgba(0,196,118,.08)', color: 'var(--green)', cursor: 'pointer', fontWeight: 700, flexShrink: 0 }}>{isSel ? '✓' : '+'}</button>
                          <button onClick={() => { setDiscarded(prev => new Set([...prev, p.ticker])); setSelectedPicks(prev => { const n = new Set(prev); n.delete(p.ticker); return n; }); }}
                            style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, border: '1px solid rgba(239,68,68,.4)', background: 'rgba(239,68,68,.1)', color: 'var(--red)', cursor: 'pointer', fontWeight: 700, flexShrink: 0 }}>✕</button>
                        </div>
                        <div style={{ display: 'flex', gap: 8, fontSize: 9, color: 'var(--dim2)', marginBottom: 3, flexWrap: 'wrap' }}>
                          <span style={{ fontFamily: 'monospace' }}>${p.price?.toFixed(2) ?? '—'}</span>
                          <span style={{ color: p.positive ? 'var(--green)' : 'var(--red)' }}>{p.change_pct != null ? `${p.positive ? '+' : ''}${p.change_pct.toFixed(2)}%` : '—'}</span>
                          <span>Conv <strong style={{ color: p.conviction >= 0.6 ? 'var(--green)' : p.conviction >= 0.35 ? 'var(--yellow)' : 'var(--dim2)' }}>{(p.conviction * 100).toFixed(0)}%</strong></span>
                          <span>Alloc <strong style={{ color: 'var(--blue)' }}>{p.alloc_pct}%</strong> <span style={{ color: 'var(--dim)' }}>{fmtM(p.alloc_amt)}</span></span>
                        </div>
                        {p.reasoning && <div style={{ fontSize: 8, color: 'var(--dim)', lineHeight: 1.4, marginBottom: p.exit_hint ? 2 : 0 }}>{p.reasoning}</div>}
                        {p.exit_hint && <div style={{ fontSize: 8, color: 'var(--yellow)', lineHeight: 1.4, fontStyle: 'italic' }}>{p.exit_hint}</div>}
                      </div>
                    );
                  })}
              </div>
              {/* footer */}
              {selectedPicks.size > 0 && (() => {
                const selPicks = picks.filter(p => selectedPicks.has(p.ticker));
                const selCount = selPicks.length;
                const tooFew   = !hasLive && selCount < 3;
                const sTot     = selPicks.reduce((s, p) => s + (p.score || 0.02), 0) || 1;
                const sPos     = selPicks.map(p => ({ ticker: p.ticker, rp: Math.max(5, Math.min(25, ((p.score || 0.02) / sTot) * 100)) }));
                const sTotRaw  = sPos.reduce((s, p) => s + p.rp, 0) || 1;
                const sFinal   = sPos.map(p => ({ ticker: p.ticker, pct: Math.round(p.rp / sTotRaw * 100), amt: Math.round((p.rp / sTotRaw) * budget) }));
                const pSum = sFinal.reduce((s, p) => s + p.pct, 0);
                if (sFinal.length) sFinal[0].pct += 100 - pSum;
                const totAmt = sFinal.reduce((s, p) => s + p.amt, 0);
                return (
                  <div style={{ padding: '8px 14px', background: 'var(--card2)', borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontSize: 10, color: 'var(--green)', fontWeight: 700 }}>{selCount} pick{selCount > 1 ? 's' : ''} · {fmtM(totAmt)} risk-weighted</span>
                    {tooFew && <span style={{ fontSize: 9, color: 'var(--yellow)' }}>Select at least 3</span>}
                    <button disabled={tooFew || activating}
                      onClick={async () => {
                        setActivating(true);
                        try {
                          if (hasLive) {
                            for (const p of sFinal) await api.aiPortfolioBuyMore(p.ticker, p.amt);
                            await fetchLive();
                          } else {
                            await api.aiPortfolioCommit(sFinal.map(p => ({ ticker: p.ticker, pct: p.pct })), budget);
                            await fetchLive(); setActiveView('live');
                          }
                          setScanResults(null); setSelectedPicks(new Set());
                        } catch (e) { setScanError(`Activation failed: ${e.message}`); }
                        finally { setActivating(false); }
                      }}
                      style={{ marginLeft: 'auto', fontSize: 11, padding: '6px 16px', borderRadius: 6, border: 'none', background: tooFew ? 'var(--card2)' : 'linear-gradient(135deg,#00c476,#3d9eff)', color: '#fff', cursor: tooFew ? 'default' : 'pointer', fontWeight: 800 }}>
                      {activating ? '⏳…' : hasLive ? `+ Add ${selCount} to Portfolio` : `▶ Activate ${selCount} — Go Live`}
                    </button>
                    <button onClick={() => setSelectedPicks(new Set())}
                      style={{ fontSize: 10, color: 'var(--dim)', background: 'none', border: '1px solid var(--border)', borderRadius: 6, padding: '5px 8px', cursor: 'pointer' }}>Clear</button>
                  </div>
                );
              })()}
            </div>
          );
        })()}

        {/* View tabs + controls */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap', alignItems: 'center' }}>
          {[
            { id: 'live',     label: '📊 Live Positions' },
            { id: 'strategy', label: '🎯 Strategy Builder' },
            { id: 'trades',   label: '📋 Trade History' },
          ].map(v => (
            <button key={v.id} className={`btn ${activeView === v.id ? 'btn-primary' : ''}`} onClick={() => setActiveView(v.id)}>
              {v.label}
            </button>
          ))}

          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
            {lastRefresh && <span style={{ fontSize: 10, color: 'var(--dim)' }}>{lastRefresh.toLocaleTimeString()}</span>}
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, color: marketOpen ? 'var(--green)' : 'var(--dim)' }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: marketOpen ? 'var(--green)' : 'var(--dim)', display: 'inline-block', animation: marketOpen ? 'pulse 2s infinite' : 'none' }} />
              {marketOpen ? 'Market Open' : 'Market Closed'}
            </div>
            <button className="btn btn-sm" onClick={fetchLive} disabled={loadingLive}>
              {loadingLive ? '…' : '↻ Refresh'}
            </button>
          </div>
        </div>

        {/* Live Positions */}
        {activeView === 'live' && (
          !hasLive ? (
            <div className="empty-state" style={{ minHeight: 300 }}>
              <div className="empty-icon">▶</div>
              <div className="empty-text" style={{ fontSize: 15, marginBottom: 10 }}>Ready to build your portfolio</div>
              <div style={{ fontSize: 12, color: 'var(--dim)', lineHeight: 2, textAlign: 'center', maxWidth: 400 }}>
                <span style={{ display: 'inline-block', marginBottom: 6 }}>
                  Pick a <strong style={{ color: 'var(--text)' }}>Market</strong> and <strong style={{ color: 'var(--text)' }}>Asset Type</strong>, set your <strong style={{ color: 'var(--text)' }}>Budget</strong> and <strong style={{ color: 'var(--text)' }}>Entry Date</strong>, then hit
                </span><br />
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 14px', borderRadius: 8, background: 'linear-gradient(135deg,rgba(61,158,255,.15),rgba(123,94,167,.15))', border: '1px solid rgba(61,158,255,.3)', fontSize: 12, fontWeight: 800, color: 'var(--blue)', marginBottom: 10 }}>▶ Run Signals</span><br />
                <span style={{ fontSize: 11 }}>AlphaAgent scans live signals · Portfolio AI proposes 3 options<br />Pick one and <strong style={{ color: 'var(--text)' }}>Finalize</strong> to go live with real entry prices</span>
              </div>
            </div>
          ) : (
            <>
              {/* ── Portfolio summary card (rectangular, above positions) ── */}
              <div style={{ marginBottom: 12, padding: '14px 18px', background: 'linear-gradient(135deg,rgba(61,158,255,.08),rgba(123,94,167,.08))', border: '1px solid rgba(61,158,255,.25)', borderRadius: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
                  <div style={{ width: 24, height: 24, borderRadius: '50%', background: 'linear-gradient(135deg,#3d9eff,#7b5ea7)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 800, color: '#fff', flexShrink: 0 }}>α</div>
                  <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--text)' }}>Live Portfolio</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginLeft: 'auto', fontSize: 10, color: marketOpen ? 'var(--green)' : 'var(--dim)' }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: marketOpen ? 'var(--green)' : 'var(--dim)', display: 'inline-block', animation: marketOpen ? 'pulse 2s infinite' : 'none' }} />
                    {marketOpen ? 'Market Open' : 'Market Closed'}
                  </div>
                  {loadingLive && <span style={{ fontSize: 10, color: 'var(--blue)' }}>↻</span>}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '6px 14px' }}>
                  {[
                    ['Market', REGIONS.find(r => r.id === (scanMeta?.region || 'all'))?.label || 'All Markets'],
                    ['Asset Type', ASSET_TYPES.find(t => t.id === (scanMeta?.assetType || 'all'))?.label || 'All Types'],
                    ['Capital', fmtUSD(summary.capital)],
                    ['Invested', fmtUSD(positions.reduce((s, p) => s + (p.market_value ?? 0), 0))],
                    ['Entry', scanMeta ? `${scanMeta.entryDate} ${scanMeta.entryTime}` : '—'],
                    ['Exit Target', scanMeta?.exitDate ? `${scanMeta.exitDate} ${scanMeta.exitTime}` : 'AI Recommended'],
                    ['Positions', `${positions.length} (${positions.filter(p => (p.pnl ?? 0) > 0).length} winning)`],
                    ['Total P&L', `${pnl >= 0 ? '+' : ''}${fmtUSD(pnl)}`],
                  ].map(([label, val]) => (
                    <div key={label}>
                      <div style={{ fontSize: 8, color: 'var(--dim)', letterSpacing: '.05em', marginBottom: 1 }}>{label.toUpperCase()}</div>
                      <div style={{ fontSize: 11, fontWeight: 700, fontFamily: 'monospace', color: label === 'Total P&L' ? pnlC : 'var(--dim2)' }}>{val}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Positions table */}
              <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, overflow: 'hidden' }}>
                {/* column header */}
                <div style={{ display: 'grid', gridTemplateColumns: '3px 110px 1fr 100px 80px 38px', gap: 10, padding: '5px 14px', background: 'rgba(255,255,255,.03)', borderBottom: '1px solid var(--border)' }}>
                  <div />
                  <div style={{ fontSize: 8, color: 'var(--dim)', fontWeight: 700, letterSpacing: '.07em' }}>TICKER</div>
                  <div style={{ fontSize: 8, color: 'var(--dim)', fontWeight: 700, letterSpacing: '.07em' }}>P&amp;L · PROGRESS</div>
                  <div style={{ fontSize: 8, color: 'var(--dim)', fontWeight: 700, letterSpacing: '.07em', textAlign: 'right' }}>MKT VALUE</div>
                  <div style={{ fontSize: 8, color: 'var(--dim)', fontWeight: 700, letterSpacing: '.07em', textAlign: 'right' }}>TODAY</div>
                  <div />
                </div>
                {positions.map((pos, i) => (
                  <PositionCard key={pos.ticker} pos={pos} idx={i}
                    onSell={handleSell} onBuyMore={handleBuyMore} />
                ))}
              </div>

              {/* P&L summary row */}
              <div style={{ marginTop: 12, padding: '10px 14px', background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                <div>
                  <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 2 }}>TOTAL P&L</div>
                  <div style={{ fontSize: 16, fontWeight: 800, fontFamily: 'monospace', color: pnlC }}>{pnl >= 0 ? '+' : ''}{fmtUSD(pnl)}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 2 }}>RETURN</div>
                  <div style={{ fontSize: 16, fontWeight: 800, color: pnlC }}>{fmtPct(retPct)}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 2 }}>WIN RATE</div>
                  <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--blue)' }}>{((summary.win_rate ?? 0) * 100).toFixed(0)}%</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 2 }}>SHARPE</div>
                  <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--purple)' }}>{(summary.sharpe ?? 0).toFixed(2)}</div>
                </div>
                {/* Top gainer / loser */}
                {positions.length > 0 && (() => {
                  const sorted = [...positions].sort((a, b) => (b.pnl_pct ?? 0) - (a.pnl_pct ?? 0));
                  const top = sorted[0], bot = sorted[sorted.length - 1];
                  return (
                    <>
                      <div style={{ marginLeft: 'auto' }}>
                        <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 2 }}>TOP GAINER</div>
                        <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--green)' }}>{top.ticker} <span style={{ fontSize: 10 }}>+{top.pnl_pct?.toFixed(2)}%</span></div>
                      </div>
                      {bot.ticker !== top.ticker && (
                        <div>
                          <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 2 }}>TOP LOSER</div>
                          <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--red)' }}>{bot.ticker} <span style={{ fontSize: 10 }}>{bot.pnl_pct?.toFixed(2)}%</span></div>
                        </div>
                      )}
                    </>
                  );
                })()}
              </div>
            </>
          )
        )}

        {/* ── Strategy Builder ── */}
        {activeView === 'strategy' && (() => {
          const sr = strategyResult;
          const regime = sr?.regime || null;
          const REGIME_COLOR = { CALM: 'var(--green)', ELEVATED: 'var(--yellow)', HIGH: 'var(--orange)', EXTREME: 'var(--red)' };
          const MODE_COLOR   = { SNIPER: 'var(--red)', CONCENTRATED: 'var(--orange)', BALANCED: 'var(--blue)', DIVERSIFIED: 'var(--green)' };

          return (
            <div>
              {/* Controls */}
              <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, padding: '14px 16px', marginBottom: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 800, color: 'var(--text)', marginBottom: 10 }}>🎯 Regime-Adaptive Portfolio Builder</div>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>
                  <div>
                    <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 4, letterSpacing: '.05em' }}>CAPITAL</div>
                    <input
                      type="number" min="1000" step="1000"
                      value={strategyCapital}
                      onChange={e => setStrategyCapital(Number(e.target.value))}
                      style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, color: 'var(--text)', padding: '6px 10px', fontSize: 12, width: 120, outline: 'none' }}
                    />
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: 'var(--dim)', marginBottom: 4, letterSpacing: '.05em' }}>STRATEGY MODE</div>
                    <select
                      value={strategyMode}
                      onChange={e => setStrategyMode(e.target.value)}
                      style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, color: 'var(--text)', padding: '6px 10px', fontSize: 12, outline: 'none' }}
                    >
                      <option value="auto">Auto (recommended)</option>
                      <option value="SNIPER">SNIPER — Top 3, conv≥60</option>
                      <option value="CONCENTRATED">CONCENTRATED — Top 5, conv≥40 ★</option>
                      <option value="BALANCED">BALANCED — Top 10, conv≥25</option>
                      <option value="DIVERSIFIED">DIVERSIFIED — Top 20, conv≥5</option>
                    </select>
                  </div>
                  <button
                    onClick={buildStrategy} disabled={strategyLoading}
                    style={{ padding: '7px 18px', borderRadius: 8, border: 'none', background: strategyLoading ? 'var(--card2)' : 'linear-gradient(135deg,#3d9eff,#7b5ea7)', color: '#fff', fontSize: 12, fontWeight: 800, cursor: strategyLoading ? 'wait' : 'pointer', display: 'flex', alignItems: 'center', gap: 7 }}
                  >
                    {strategyLoading
                      ? <><div style={{ width: 11, height: 11, border: '2px solid rgba(255,255,255,.3)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} /> Building…</>
                      : '▶ Build Portfolio'}
                  </button>
                </div>
                {strategyError && <div style={{ fontSize: 10, color: 'var(--red)', marginTop: 8 }}>{strategyError}</div>}
              </div>

              {sr && (
                <>
                  {/* Regime band */}
                  <div style={{ background: 'var(--card)', border: `1px solid ${REGIME_COLOR[regime?.label] || 'var(--border)'}40`, borderRadius: 10, padding: '12px 16px', marginBottom: 12, display: 'flex', flexWrap: 'wrap', gap: 20, alignItems: 'center' }}>
                    <div>
                      <div style={{ fontSize: 9, color: 'var(--dim)', letterSpacing: '.05em' }}>MARKET REGIME</div>
                      <div style={{ fontSize: 16, fontWeight: 900, color: REGIME_COLOR[regime?.label] || 'var(--dim)' }}>{regime?.label || '—'}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: 'var(--dim)', letterSpacing: '.05em' }}>VIX</div>
                      <div style={{ fontSize: 15, fontWeight: 800, color: (regime?.vix || 0) > 25 ? 'var(--red)' : (regime?.vix || 0) > 18 ? 'var(--yellow)' : 'var(--green)', fontFamily: 'monospace' }}>{regime?.vix?.toFixed(1) || '—'}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: 'var(--dim)', letterSpacing: '.05em' }}>SPY</div>
                      <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--text)', fontFamily: 'monospace' }}>${regime?.spy_now?.toFixed(2) || '—'}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: 'var(--dim)', letterSpacing: '.05em' }}>SPY 1D</div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: (regime?.spy_1d || 0) >= 0 ? 'var(--green)' : 'var(--red)', fontFamily: 'monospace' }}>{(regime?.spy_1d || 0) >= 0 ? '+' : ''}{regime?.spy_1d?.toFixed(2) || '—'}%</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: 'var(--dim)', letterSpacing: '.05em' }}>SPY vs 20-SMA</div>
                      <div style={{ fontSize: 12, fontWeight: 700, color: regime?.above20 ? 'var(--green)' : 'var(--red)' }}>{regime?.above20 ? '▲ Above' : '▼ Below'}</div>
                    </div>
                    {sr.opportunity_day && (
                      <div style={{ padding: '5px 12px', background: 'rgba(255,183,3,.12)', border: '1px solid rgba(255,183,3,.4)', borderRadius: 8, fontSize: 10, fontWeight: 800, color: 'var(--yellow)' }}>
                        ⚡ OPPORTUNITY DAY — Market down {regime?.spy_1d?.toFixed(1)}% with VIX {regime?.vix}
                      </div>
                    )}
                    <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                      <div style={{ fontSize: 9, color: 'var(--dim)' }}>Signal cache coverage</div>
                      <div style={{ fontSize: 12, fontWeight: 700, color: (sr.cache_coverage || 0) > 30 ? 'var(--green)' : 'var(--yellow)' }}>{sr.cache_coverage || 0} tickers warmed</div>
                    </div>
                  </div>

                  {/* Mode cards */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 8, marginBottom: 12 }}>
                    {Object.entries(sr.modes || {}).map(([key, m]) => {
                      const isChosen = key === sr.chosen_mode;
                      const isRec    = key === sr.recommended_mode;
                      const mc       = MODE_COLOR[key] || 'var(--dim)';
                      return (
                        <div key={key} style={{ background: isChosen ? `${mc}12` : 'var(--card)', border: `1px solid ${isChosen ? mc : 'var(--border)'}`, borderRadius: 10, padding: '10px 13px', position: 'relative' }}>
                          {isRec && <div style={{ position: 'absolute', top: 7, right: 8, fontSize: 8, fontWeight: 800, color: mc, background: `${mc}15`, padding: '2px 6px', borderRadius: 4 }}>RECOMMENDED</div>}
                          <div style={{ fontSize: 12, fontWeight: 900, color: mc, marginBottom: 2 }}>{key}</div>
                          <div style={{ fontSize: 9, color: 'var(--dim2)', marginBottom: 6 }}>{m.positions} positions · {Math.round(m.pct_each * 100)}% each · conv≥{m.min_conv}</div>
                          <div style={{ fontSize: 9, color: 'var(--green)', fontWeight: 700 }}>YTD: {m.ytd}</div>
                          <div style={{ fontSize: 8, color: 'var(--dim)', marginTop: 2 }}>+{m.good_day} / −{m.bad_day}</div>
                          <div style={{ fontSize: 7, color: 'var(--dim)', marginTop: 4, lineHeight: 1.4 }}>{m.best_for}</div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Positions table */}
                  {sr.positions?.length > 0 ? (
                    <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, overflow: 'hidden', marginBottom: 12 }}>
                      <div style={{ padding: '10px 14px', background: 'var(--card2)', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 12, fontWeight: 800 }}>Portfolio — {sr.chosen_mode}</span>
                        <span style={{ fontSize: 10, color: 'var(--dim)', fontWeight: 600 }}>{sr.n_positions} positions</span>
                        <span style={{ fontSize: 10, color: 'var(--text)', fontFamily: 'monospace' }}>${(sr.allocated || 0).toLocaleString('en-US', { maximumFractionDigits: 0 })} invested</span>
                        <span style={{ fontSize: 10, color: 'var(--dim)' }}>${(sr.cash || 0).toLocaleString('en-US', { maximumFractionDigits: 0 })} cash</span>
                        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                          <div style={{ fontSize: 10, color: 'var(--dim)' }}>Expected P&amp;L</div>
                          <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--green)', fontFamily: 'monospace' }}>+${(sr.expected_pnl || 0).toLocaleString('en-US', { maximumFractionDigits: 0 })}</div>
                          <div style={{ fontSize: 8, color: 'var(--dim)' }}>range: +${(sr.exp_pnl_low || 0).toLocaleString()} → +${(sr.exp_pnl_high || 0).toLocaleString()}</div>
                        </div>
                      </div>
                      <div style={{ overflowX: 'auto' }}>
                        <table className="data-table">
                          <thead>
                            <tr>
                              {['#', 'Symbol', 'Sector', 'P(Up)', 'Conv', 'Agents', 'Kelly', '$ Alloc', '% Port', 'Stop', 'Target'].map(h => (
                                <th key={h} style={{ fontSize: 9 }}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {sr.positions.map((p, i) => (
                              <tr key={p.symbol}>
                                <td style={{ fontSize: 10, color: 'var(--dim)' }}>{i + 1}</td>
                                <td style={{ fontWeight: 800, fontFamily: 'monospace', color: 'var(--green)' }}>{p.symbol}</td>
                                <td style={{ fontSize: 10, color: 'var(--dim2)' }}>{p.sector}</td>
                                <td style={{ fontFamily: 'monospace', color: p.prob >= 62 ? 'var(--green)' : p.prob >= 55 ? 'var(--yellow)' : 'var(--red)' }}>{p.prob?.toFixed(1)}%</td>
                                <td style={{ fontFamily: 'monospace', color: p.conviction >= 60 ? 'var(--green)' : p.conviction >= 40 ? 'var(--yellow)' : 'var(--dim)' }}>{p.conviction?.toFixed(0)}</td>
                                <td style={{ fontSize: 10, color: 'var(--dim)' }}>{p.long_votes}/{p.n_agents}</td>
                                <td style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--dim2)' }}>{(p.kelly_f * 100).toFixed(1)}%</td>
                                <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>${(p.dollar_alloc || 0).toLocaleString('en-US', { maximumFractionDigits: 0 })}</td>
                                <td style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--cyan)' }}>{p.pct_portfolio?.toFixed(1)}%</td>
                                <td style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--red)' }}>{p.stop ? `$${p.stop}` : '—'}</td>
                                <td style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--green)' }}>{p.target ? `$${p.target}` : '—'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : (
                    <div style={{ padding: '24px', textAlign: 'center', background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10 }}>
                      <div style={{ fontSize: 13, color: 'var(--yellow)', marginBottom: 8 }}>⚠ No positions pass {sr.chosen_mode} thresholds yet</div>
                      <div style={{ fontSize: 11, color: 'var(--dim)' }}>
                        {sr.cache_coverage < 20
                          ? `Signal cache has only ${sr.cache_coverage} tickers warmed. Wait ~2 min for more signals to load, then rebuild.`
                          : `${sr.n_candidates} candidates found but none pass conv≥${sr.mode_info?.min_conv} + prob≥${Math.round((sr.mode_info?.min_prob || 0) * 100)}%. Try BALANCED or DIVERSIFIED mode.`}
                      </div>
                    </div>
                  )}
                </>
              )}

              {!sr && !strategyLoading && (
                <div className="empty-state" style={{ minHeight: 200 }}>
                  <div className="empty-icon">🎯</div>
                  <div className="empty-text">Regime-adaptive portfolio builder</div>
                  <div style={{ fontSize: 11, color: 'var(--dim)', lineHeight: 1.8, maxWidth: 400, textAlign: 'center' }}>
                    Detects live market regime (VIX + SPY trend), recommends a strategy mode,<br />
                    and sizes positions using Kelly criterion from the AlphaAgent signal cache.<br />
                    <strong style={{ color: 'var(--text)' }}>CONCENTRATED</strong> outperformed +15.5% Jan–May 2026.
                  </div>
                </div>
              )}
            </div>
          );
        })()}

        {/* Trade history */}
        {activeView === 'trades' && (
          <div className="card">
            <div className="card-header"><span className="card-title">Trade History (Last 50)</span></div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr><th>Date</th><th>Ticker</th><th>Direction</th><th>Shares</th><th>Entry $</th><th>Exit $</th><th>P&L</th><th>Return %</th></tr>
                </thead>
                <tbody>
                  {trades.length === 0 ? (
                    <tr><td colSpan={8} style={{ textAlign: 'center', color: 'var(--dim)', padding: 24 }}>No trades recorded yet</td></tr>
                  ) : trades.slice(0, 50).map((t, i) => {
                    const pnl = t.pnl ?? 0;
                    const c = pnlClr(pnl);
                    return (
                      <tr key={i}>
                        <td style={{ fontSize: 11, color: 'var(--dim)' }}>{t.date || t.timestamp || '—'}</td>
                        <td style={{ fontWeight: 700 }}>{t.ticker}</td>
                        <td><span className={`badge ${t.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>{t.direction}</span></td>
                        <td style={{ color: 'var(--dim2)' }}>{t.shares ?? t.quantity ?? '—'}</td>
                        <td style={{ fontFamily: 'monospace' }}>${(t.entry_price ?? 0).toFixed(2)}</td>
                        <td style={{ fontFamily: 'monospace' }}>${(t.exit_price ?? 0).toFixed(2)}</td>
                        <td style={{ color: c, fontWeight: 700, fontFamily: 'monospace' }}>{pnl >= 0 ? '+' : ''}${pnl.toFixed(0)}</td>
                        <td style={{ color: c }}>{((t.return_pct ?? 0) * 100).toFixed(2)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* ══ RIGHT: Portfolio AI (sticky, full height) ══ */}
      <div style={{
        width: 460, flexShrink: 0, position: 'sticky', top: 16, alignSelf: 'flex-start',
        height: 'calc(100vh - var(--header-h) - var(--nav-h) - 36px)',
        display: 'flex', flexDirection: 'column',
      }}>
        <PortfolioAIAssistant
          livePositions={positions}
          budget={budget}
          entryDate={entryDate}
          entryTime={entryTime}
          exitDate={exitDate}
          exitTime={exitTime}
          hasLive={hasLive}
          onReset={handleReset}
          externalQuery={aiChatQuery}
          onExternalQueryConsumed={() => setAiChatQuery('')}
        />
      </div>
    </div>
  );
}
