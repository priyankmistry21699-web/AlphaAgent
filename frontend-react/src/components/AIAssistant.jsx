import { useState, useRef, useEffect } from 'react';
import { api } from '../api.js';
import { QUICK_CHIPS } from '../constants.js';

const DISCOVERY_CHIPS = [
  'How will gold do?', 'Analyze Bitcoin', 'EUR/USD outlook?',
  'Best Indian stocks?', 'S&P 500 trend?', 'Oil price forecast?',
];

const ANALYZE_RE = /\n?ANALYZE:([A-Z0-9.\-=^]+)\s*$/i;

function parseAnalyze(text) {
  const m = text.match(ANALYZE_RE);
  if (!m) return { cleanText: text, ticker: null };
  return {
    cleanText: text.replace(ANALYZE_RE, '').trim(),
    ticker: m[1].toUpperCase(),
  };
}

function renderText(text) {
  if (!text) return null;
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

export default function AIAssistant({ ticker, signalData, onAnalyze }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const messagesRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (ticker) {
      setMessages([{
        role: 'ai',
        text: `Signal loaded for **${ticker}**. Ask me anything — I have full context on all 9 agent votes, factor scores, risk overrides, and the ${signalData?.horizon?.toUpperCase() || '1M'} horizon analysis.`,
      }]);
    }
  }, [ticker]);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages]);

  async function sendChat(question) {
    const q = (question || input).trim();
    if (!q || busy) return;
    setInput('');
    setMessages(prev => [...prev,
      { role: 'user', text: q },
      { role: 'ai', text: null },
    ]);
    setBusy(true);
    try {
      const res = await api.chat(q, ticker || '', signalData);
      const raw = res.answer || res.response || 'No response.';
      const { cleanText, ticker: suggested } = parseAnalyze(raw);

      if (suggested && onAnalyze) {
        setMessages(prev => {
          const next = [...prev];
          next[next.length - 1] = { role: 'ai', text: cleanText };
          return [
            ...next,
            { role: 'action', text: `Launching analysis for **${suggested}**…`, ticker: suggested },
          ];
        });
        setTimeout(() => onAnalyze(suggested), 400);
      } else {
        setMessages(prev => {
          const next = [...prev];
          next[next.length - 1] = { role: 'ai', text: cleanText };
          return next;
        });
      }
    } catch (err) {
      setMessages(prev => {
        const next = [...prev];
        next[next.length - 1] = { role: 'ai', text: `Error: ${err.message}` };
        return next;
      });
    } finally {
      setBusy(false);
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
  }

  const chips = ticker ? QUICK_CHIPS : DISCOVERY_CHIPS;

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="chat-avatar">α</div>
          <div>
            <div className="chat-title-text">
              AI Assistant
              {ticker && <span> · <span style={{ color: 'var(--cyan)' }}>{ticker}</span></span>}
            </div>
            <div className="chat-sub">
              {ticker ? 'Signal-aware · Powered by AI' : 'Ask me what to analyze'}
            </div>
          </div>
        </div>
        <button
          style={{ fontSize: 11, color: 'var(--dim)', background: 'none', border: 'none', cursor: 'pointer' }}
          onClick={() => setMessages([])}
        >Clear</button>
      </div>

      <div className="chat-messages" ref={messagesRef}>
        {messages.length === 0 && !ticker && (
          <div style={{ color: 'var(--dim)', fontSize: 11, textAlign: 'center', padding: '20px 14px', lineHeight: 1.75 }}>
            <div style={{ fontSize: 26, marginBottom: 10, opacity: .7 }}>α</div>
            <div style={{ color: 'var(--dim2)', marginBottom: 6 }}>
              Ask me about any <strong>stock</strong>, <strong>crypto</strong>, <strong>forex pair</strong>, or <strong>commodity</strong>.
            </div>
            <div style={{ fontSize: 10, color: 'var(--dim)' }}>
              I'll identify the ticker and launch a deep analysis for you.
            </div>
          </div>
        )}
        {messages.length === 0 && ticker && (
          <div style={{ color: 'var(--dim)', fontSize: 11, textAlign: 'center', padding: '24px 12px', lineHeight: 1.7 }}>
            Ask me about the signal, agent votes, risk factors, or entry timing.
          </div>
        )}
        {messages.map((m, i) => {
          if (m.role === 'action') {
            return (
              <div key={i} style={{
                margin: '6px 0', padding: '7px 12px', background: 'rgba(61,158,255,.1)',
                border: '1px solid rgba(61,158,255,.25)', borderRadius: 8,
                fontSize: 11, color: 'var(--blue)', display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <div className="spinner" style={{ width: 10, height: 10, borderWidth: 1.5, flexShrink: 0 }} />
                <span dangerouslySetInnerHTML={{ __html: renderText(m.text) }} />
              </div>
            );
          }
          return (
            <div key={i} className={`chat-msg ${m.role === 'user' ? 'chat-user' : 'chat-ai'}`}>
              <div
                className={`chat-bubble ${m.text === null ? 'thinking' : ''}`}
                dangerouslySetInnerHTML={m.text !== null ? { __html: renderText(m.text) } : undefined}
              >
                {m.text === null ? '' : null}
              </div>
            </div>
          );
        })}
      </div>

      <div className="chat-quick">
        {chips.map(chip => (
          <button key={chip} className="chat-chip" onClick={() => sendChat(chip)} disabled={busy}>
            {chip}
          </button>
        ))}
      </div>

      <div className="chat-input-row">
        <textarea
          ref={inputRef}
          className="chat-input"
          placeholder={ticker ? `Ask about ${ticker}…` : 'Ask about any stock, crypto, forex, or commodity…'}
          value={input}
          onChange={e => { setInput(e.target.value); e.target.style.height = ''; e.target.style.height = Math.min(e.target.scrollHeight, 80) + 'px'; }}
          onKeyDown={handleKey}
          rows={1}
          disabled={busy}
        />
        <button className="chat-send" onClick={() => sendChat()} disabled={busy || !input.trim()}>↑</button>
      </div>
    </div>
  );
}
