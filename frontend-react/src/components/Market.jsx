import { useState, useEffect } from 'react';
import { api, dirColor } from '../api.js';
import { TOP_TICKERS } from '../constants.js';

const WATCHLIST = TOP_TICKERS.slice(0, 30);

export default function MarketTab({ onAnalyze }) {
  const [signals, setSignals] = useState({});
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState(null);
  const [marketInput, setMarketInput] = useState('');
  const [marketResponse, setMarketResponse] = useState('');
  const [chatBusy, setChatBusy] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    loadMarketSignals();
    api.status().then(setStatus).catch(() => {});
  }, []);

  async function loadMarketSignals() {
    setLoading(true);
    const subset = WATCHLIST.slice(0, 12);
    const results = await Promise.allSettled(
      subset.map(t => api.signal(t.sym, '1d').then(d => ({ sym: t.sym, name: t.name, ...d })))
    );
    const map = {};
    results.forEach((r, i) => {
      if (r.status === 'fulfilled') map[subset[i].sym] = r.value;
    });
    setSignals(map);
    setLoading(false);
  }

  async function sendMarketChat() {
    if (!marketInput.trim() || chatBusy) return;
    const q = marketInput.trim();
    setMarketInput('');
    setChatBusy(true);
    const newHistory = [...history, { role: 'user', content: q }];
    setHistory(newHistory);
    try {
      const res = await api.marketChat(q, newHistory);
      const reply = res.answer || res.response || 'No response.';
      setMarketResponse(reply);
      setHistory([...newHistory, { role: 'assistant', content: reply }]);
    } catch (err) {
      setMarketResponse(`Error: ${err.message}`);
    } finally {
      setChatBusy(false);
    }
  }

  const longs = Object.values(signals).filter(s => s.direction === 'LONG').length;
  const shorts = Object.values(signals).filter(s => s.direction === 'SHORT').length;
  const holds = Object.values(signals).filter(s => s.direction === 'HOLD').length;
  const bullPct = Object.keys(signals).length ? ((longs / Object.keys(signals).length) * 100).toFixed(0) : '—';

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      {/* Market Breadth */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        {[
          { label: 'Market Breadth', val: `${bullPct}% Bullish`, color: 'var(--green)' },
          { label: 'Bullish Signals', val: longs, color: 'var(--green)' },
          { label: 'Bearish Signals', val: shorts, color: 'var(--red)' },
          { label: 'Neutral Signals', val: holds, color: 'var(--yellow)' },
          { label: 'API Status', val: status ? '● Online' : '● Loading', color: status ? 'var(--green)' : 'var(--dim)' },
        ].map(m => (
          <div key={m.label} className="metric-card" style={{ minWidth: 130 }}>
            <div className="metric-lbl">{m.label}</div>
            <div className="metric-val" style={{ color: m.color, fontSize: 18 }}>{m.val}</div>
          </div>
        ))}
      </div>

      {/* Live Market Cards */}
      <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ fontSize: 13, fontWeight: 700, color: 'var(--dim)' }}>
          <span className="live-dot" />Live Signal Overview
        </h3>
        <button className="btn btn-sm" onClick={loadMarketSignals}>↻ Refresh</button>
      </div>

      {loading ? (
        <div className="spinner-wrap"><div className="spinner" /><div className="spinner-text">Loading market signals…</div></div>
      ) : (
        <div className="market-grid">
          {WATCHLIST.slice(0, 12).map(({ sym, name }) => {
            const s = signals[sym];
            if (!s) return (
              <div key={sym} className="market-card" style={{ opacity: .5 }}>
                <div className="mc-sym">{sym}</div>
                <div className="mc-name">{name}</div>
                <div style={{ fontSize: 11, color: 'var(--dim)' }}>Loading…</div>
              </div>
            );
            const dir = (s.direction || 'HOLD').toUpperCase();
            const prob = s.probability || 0.5;
            const color = dirColor(dir);
            return (
              <div key={sym} className="market-card" onClick={() => onAnalyze(sym)}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <div className="mc-sym">{sym}</div>
                    <div className="mc-name">{name}</div>
                  </div>
                  <span className={`badge ${dir === 'LONG' ? 'badge-long' : dir === 'SHORT' ? 'badge-short' : 'badge-hold'}`}>{dir}</span>
                </div>
                <div className="mc-price" style={{ color }}>{(prob * 100).toFixed(1)}%</div>
                <div className="mc-chg" style={{ color }}>P(Up) · {(s.conviction || 0).toFixed(1)}% conviction</div>
                <div style={{ marginTop: 8 }}>
                  <div style={{ background: 'var(--border)', borderRadius: 3, height: 5, overflow: 'hidden' }}>
                    <div style={{ width: `${(prob * 100).toFixed(0)}%`, height: 5, background: color, borderRadius: 3, transition: 'width .4s' }} />
                  </div>
                </div>
                <div style={{ fontSize: 10, color: 'var(--dim)', marginTop: 6 }}>Click to analyze →</div>
              </div>
            );
          })}
        </div>
      )}

      {/* Global Market AI */}
      <div className="card" style={{ maxWidth: 700 }}>
        <div className="card-header">
          <span className="card-title">🌐 Global Market AI</span>
        </div>
        <div className="card-body">
          {marketResponse && (
            <div style={{ background: 'var(--card2)', border: '1px solid var(--border)', borderRadius: 8, padding: 14, marginBottom: 12, fontSize: 12, color: 'var(--dim2)', lineHeight: 1.7 }}
              dangerouslySetInnerHTML={{ __html: marketResponse.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>') }}
            />
          )}
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              style={{ flex: 1, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)', padding: '8px 12px', fontSize: 12, outline: 'none' }}
              placeholder="Ask about market conditions, sector rotation, macro trends…"
              value={marketInput}
              onChange={e => setMarketInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && sendMarketChat()}
            />
            <button className="btn btn-primary btn-sm" onClick={sendMarketChat} disabled={chatBusy || !marketInput.trim()}>
              {chatBusy ? '⏳' : '↑ Ask'}
            </button>
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
            {['Market outlook?', 'Top sectors?', 'Fed impact?', 'Risk-on or risk-off?'].map(q => (
              <button key={q} className="chat-chip" onClick={() => { setMarketInput(q); }}>
                {q}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
