import { useState, useRef, useEffect } from 'react';
import { api } from '../api.js';
import { QUICK_CHIPS } from '../constants.js';

export default function AIAssistant({ ticker, signalData }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const messagesRef = useRef(null);
  const inputRef = useRef(null);

  // Reset on ticker change
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
    const userMsg = { role: 'user', text: q };
    setMessages(prev => [...prev, userMsg, { role: 'ai', text: null }]);
    setBusy(true);
    try {
      const res = await api.chat(q, ticker, signalData);
      const reply = res.answer || res.response || 'No response.';
      setMessages(prev => {
        const next = [...prev];
        next[next.length - 1] = { role: 'ai', text: reply };
        return next;
      });
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

  function renderText(text) {
    if (!text) return null;
    return text
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="chat-avatar">α</div>
          <div>
            <div className="chat-title-text">AI Assistant · <span style={{ color: 'var(--cyan)' }}>{ticker}</span></div>
            <div className="chat-sub">Powered by Claude · Signal-aware</div>
          </div>
        </div>
        <button
          style={{ fontSize: 11, color: 'var(--dim)', background: 'none', border: 'none', cursor: 'pointer' }}
          onClick={() => setMessages([])}
        >
          Clear
        </button>
      </div>

      <div className="chat-messages" ref={messagesRef}>
        {messages.length === 0 && (
          <div style={{ color: 'var(--dim)', fontSize: 11, textAlign: 'center', padding: '24px 12px', lineHeight: 1.7 }}>
            Ask me about the signal, agent votes, risk factors, or entry timing.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role === 'user' ? 'chat-user' : 'chat-ai'}`}>
            <div
              className={`chat-bubble ${m.text === null ? 'thinking' : ''}`}
              dangerouslySetInnerHTML={m.text !== null ? { __html: renderText(m.text) } : undefined}
            >
              {m.text === null ? '' : null}
            </div>
          </div>
        ))}
      </div>

      <div className="chat-quick">
        {QUICK_CHIPS.map(chip => (
          <button key={chip} className="chat-chip" onClick={() => sendChat(chip)} disabled={busy}>
            {chip}
          </button>
        ))}
      </div>

      <div className="chat-input-row">
        <textarea
          ref={inputRef}
          className="chat-input"
          placeholder="Ask about this signal…"
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
