import { useState } from 'react';
import { priceColor } from '../api.js';
import { AGENT_THEORIES, FACTOR_ENC } from '../constants.js';

function ProbBar({ prob }) {
  const color = priceColor(prob);
  return (
    <div className="prob-bar-wrap">
      <div className="prob-bar">
        <div className="prob-bar-fill" style={{ width: `${(prob * 100).toFixed(0)}%`, background: color }} />
      </div>
      <span className="prob-val" style={{ color }}>{(prob * 100).toFixed(1)}%</span>
    </div>
  );
}

function FactorCard({ factorKey, f }) {
  const enc = FACTOR_ENC[factorKey] || {};
  const sc = f.score ?? 50;
  const scColor = sc >= 66 ? 'var(--green)' : sc <= 33 ? 'var(--red)' : 'var(--yellow)';
  const signal = sc >= 66 ? 'BULLISH' : sc <= 33 ? 'BEARISH' : 'NEUTRAL';
  const bgColor = sc >= 66 ? 'rgba(0,214,143,0.06)' : sc <= 33 ? 'rgba(245,54,92,0.06)' : 'rgba(255,183,3,0.04)';
  const borderColor = sc >= 66 ? 'rgba(0,214,143,0.22)' : sc <= 33 ? 'rgba(245,54,92,0.22)' : 'rgba(255,183,3,0.14)';

  const rawVal = typeof f.value === 'number'
    ? (Math.abs(f.value) >= 1000 ? f.value.toLocaleString('en-US', { maximumFractionDigits: 0 })
      : Math.abs(f.value) < 0.01 && f.value !== 0 ? f.value.toExponential(2)
      : f.value.toFixed(3))
    : String(f.value ?? '—');

  const RANGE_COLORS = { great: 'var(--green)', good: '#22c55e', neutral: 'var(--yellow)', bad: '#f97316', terrible: 'var(--red)' };

  return (
    <div style={{ background: bgColor, border: `1px solid ${borderColor}`, borderRadius: 10, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '11px 13px 6px', gap: 10 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>{f.name || factorKey}</div>
          {enc.fullName && enc.fullName !== (f.name || factorKey) && (
            <div style={{ fontSize: 9, color: 'var(--dim)', marginTop: 1 }}>{enc.fullName}</div>
          )}
          {enc.category && (
            <span style={{ fontSize: 8, fontWeight: 700, color: 'var(--cyan)', background: 'rgba(6,200,240,.1)', padding: '2px 6px', borderRadius: 8, border: '1px solid rgba(6,200,240,.2)', marginTop: 4, display: 'inline-block', letterSpacing: '.05em' }}>
              {enc.category}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4, flexShrink: 0 }}>
          <span style={{ fontSize: 8, fontWeight: 800, padding: '2px 7px', borderRadius: 7, letterSpacing: '.06em', color: scColor, background: `${scColor}18`, border: `1px solid ${scColor}30` }}>{signal}</span>
          <span style={{ fontSize: 22, fontWeight: 900, fontFamily: 'monospace', color: scColor, lineHeight: 1 }}>{sc.toFixed(0)}<span style={{ fontSize: 10, color: 'var(--dim)', fontWeight: 400 }}>/100</span></span>
        </div>
      </div>

      {/* Score bar */}
      <div style={{ padding: '2px 13px 8px', position: 'relative' }}>
        <div style={{ height: 7, borderRadius: 4, display: 'flex', overflow: 'hidden' }}>
          <div style={{ width: '33%', background: 'rgba(245,54,92,.18)' }} />
          <div style={{ width: '34%', background: 'rgba(255,183,3,.13)' }} />
          <div style={{ width: '33%', background: 'rgba(0,214,143,.18)' }} />
        </div>
        <div style={{ position: 'absolute', top: 2, left: `calc(13px + ${sc}% * (100% - 26px) / 100)`, width: 3, height: 10, background: scColor, borderRadius: 2, transform: 'translateX(-50%)', boxShadow: `0 0 5px ${scColor}80` }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: 'var(--dim)', marginTop: 3 }}>
          <span>0 Bearish</span><span>50 Neutral</span><span>Bullish 100</span>
        </div>
      </div>

      {/* Body */}
      <div style={{ padding: '0 13px 11px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 9, color: 'var(--dim)', fontWeight: 600 }}>Current Value</span>
          <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', fontFamily: 'monospace' }}>
            {rawVal}{enc.valueUnit && <span style={{ fontSize: 9, color: 'var(--dim)', fontWeight: 400 }}> {enc.valueUnit}</span>}
          </span>
        </div>
        {enc.formula && (
          <div style={{ fontSize: 9, color: 'var(--dim2)', background: 'var(--surface)', padding: '5px 8px', borderRadius: 5, borderLeft: '2px solid var(--blue)', fontFamily: 'monospace', lineHeight: 1.5 }}>📐 {enc.formula}</div>
        )}
        {f.interpretation && (
          <div style={{ fontSize: 10, color: 'var(--dim2)', fontStyle: 'italic', lineHeight: 1.55 }}>{f.interpretation}</div>
        )}
        {enc.great && (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 9 }}>
            <tbody>
              {[['great', '●', 'GREAT'], ['good', '●', 'GOOD'], ['neutral', '●', 'NEUTRAL'], ['bad', '●', 'BAD'], ['terrible', '●', 'TERRIBLE']].map(([k, dot, label]) =>
                enc[k] ? (
                  <tr key={k}>
                    <td style={{ width: 14, color: RANGE_COLORS[k], paddingRight: 4 }}>{dot}</td>
                    <td style={{ width: 56, fontWeight: 700, color: 'var(--dim)', letterSpacing: '.04em', paddingRight: 6 }}>{label}</td>
                    <td style={{ color: 'var(--dim2)' }}>{enc[k]}</td>
                  </tr>
                ) : null
              )}
            </tbody>
          </table>
        )}
        {enc.note && (
          <div style={{ fontSize: 9, color: 'var(--dim)', padding: '5px 8px', background: 'var(--surface)', borderRadius: 5, borderLeft: '2px solid var(--purple)', lineHeight: 1.5 }}>💡 {enc.note}</div>
        )}
      </div>
    </div>
  );
}

export default function AgentRow({ agent, ticker, hzLabel }) {
  const [expanded, setExpanded] = useState(false);
  const [deepOpen, setDeepOpen] = useState(false);
  const [reasonOpen, setReasonOpen] = useState(false);
  const name = agent.agent_name || agent.name || 'Unknown';
  const vote = (agent.vote || 'HOLD').toUpperCase();
  const p = agent.probability_up ?? 0.5;
  const conf = agent.confidence ?? 0;
  const ms = agent.computation_time_ms ?? 0;
  const reasoning = (agent.reasoning || '').slice(0, 160);
  const fullReasoning = agent.reasoning || '';
  const factors = Object.entries(agent.factor_scores || {});
  const nf = factors.length;
  const wt = nf > 0 ? (100 / nf).toFixed(1) : 0;
  const pColor = priceColor(p);
  const voteClass = vote === 'LONG' ? 'badge-long' : vote === 'SHORT' ? 'badge-short' : 'badge-hold';
  const voteColor = vote === 'LONG' ? 'var(--green)' : vote === 'SHORT' ? 'var(--red)' : 'var(--yellow)';

  const bullishFactors = factors.filter(([, f]) => (f.score ?? 50) >= 66).length;
  const bearishFactors = factors.filter(([, f]) => (f.score ?? 50) <= 33).length;

  return (
    <>
      <tr onClick={() => setExpanded(e => !e)} style={{ cursor: 'pointer' }} title="Click to expand factor breakdown">
        <td>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ color: 'var(--dim)', fontSize: 11 }}>{expanded ? '▼' : '▶'}</span>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <strong style={{ color: 'var(--text)' }}>{name}</strong>
                <button
                  className="deep-dive-btn"
                  onClick={(e) => { e.stopPropagation(); setDeepOpen(d => !d); if (!expanded) setExpanded(true); }}
                  title="Toggle inline Factor Deep Dive"
                  style={{ background: deepOpen ? 'rgba(61,158,255,.2)' : undefined, color: deepOpen ? 'var(--blue)' : undefined }}
                >
                  {deepOpen ? '🔬 Close' : '🔬 Deep Dive'}
                </button>
              </div>
              <div style={{ fontSize: 9, color: 'var(--dim)', marginTop: 1 }}>{nf} factors</div>
              <div style={{ marginTop: 4, lineHeight: 2 }}>
                {AGENT_THEORIES[name]?.tags.slice(0, 5).map(tag => (
                  <span key={tag} className={`tag ${AGENT_THEORIES[name]?.color || 'tag-blue'}`}>{tag}</span>
                ))}
              </div>
            </div>
          </div>
        </td>
        <td><span className={`badge ${voteClass}`}>{vote}</span></td>
        <td style={{ minWidth: 130 }}>
          <div className="prob-bar-wrap">
            <div className="prob-bar">
              <div className="prob-bar-fill" style={{ width: `${(p * 100).toFixed(0)}%`, background: pColor }} />
            </div>
            <span className="prob-val" style={{ color: pColor }}>{(p * 100).toFixed(1)}%</span>
          </div>
        </td>
        <td style={{ color: 'var(--dim2)' }}>{(conf * 100).toFixed(0)}%</td>
        <td style={{ color: 'var(--dim)', fontSize: 11 }}>{ms.toFixed(0)}ms</td>
        <td style={{ fontSize: 11, maxWidth: 260 }} onClick={e => e.stopPropagation()}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6 }}>
            <span style={{ color: 'var(--dim2)', flex: 1, lineHeight: 1.55 }}>
              {reasonOpen ? fullReasoning : reasoning}
              {!reasonOpen && fullReasoning.length > 160 ? '…' : ''}
            </span>
            {fullReasoning.length > 160 && (
              <button
                onClick={() => setReasonOpen(o => !o)}
                style={{
                  flexShrink: 0, fontSize: 9, padding: '2px 6px', borderRadius: 4,
                  border: '1px solid var(--border)', background: reasonOpen ? 'var(--blue)' : 'var(--card2)',
                  color: reasonOpen ? '#fff' : 'var(--dim)', cursor: 'pointer', lineHeight: 1,
                  marginTop: 1,
                }}
              >{reasonOpen ? '▲ less' : '▼ more'}</button>
            )}
          </div>
        </td>
      </tr>

      {/* ── Simple factor table (row click expand) ── */}
      {expanded && !deepOpen && nf > 0 && (
        <tr className="fd-row">
          <td colSpan={6}>
            <div className="fd-content">
              <div className="fd-header">
                FACTOR BREAKDOWN
                <span style={{ color: 'var(--dim2)', fontWeight: 400 }}>{nf} factors · equal weight {wt}% each</span>
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <th style={{ textAlign: 'left', padding: '4px 8px', fontSize: 10, color: 'var(--dim)', fontWeight: 600 }}>Factor</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px', fontSize: 10, color: 'var(--dim)', fontWeight: 600 }}>Raw Value</th>
                    <th style={{ padding: '4px 8px', fontSize: 10, color: 'var(--dim)', fontWeight: 600, minWidth: 160 }}>Score</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px', fontSize: 10, color: 'var(--dim)', fontWeight: 600 }}>Weight</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px', fontSize: 10, color: 'var(--dim)', fontWeight: 600 }}>Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {factors.map(([key, f]) => {
                    const sc = f.score ?? 50;
                    const scColor = sc >= 66 ? 'var(--green)' : sc <= 33 ? 'var(--red)' : 'var(--yellow)';
                    const rawVal = typeof f.value === 'number'
                      ? (Math.abs(f.value) >= 1000 ? f.value.toLocaleString('en-US', { maximumFractionDigits: 0 })
                        : Math.abs(f.value) < 0.01 && f.value !== 0 ? f.value.toExponential(2)
                        : f.value.toFixed(3))
                      : String(f.value ?? '—');
                    return (
                      <tr key={key} style={{ borderBottom: '1px solid #0d1825' }}>
                        <td style={{ padding: '5px 8px', fontSize: 11, color: 'var(--text)', whiteSpace: 'nowrap' }}>{f.name || key}</td>
                        <td style={{ padding: '5px 8px', fontSize: 11, color: 'var(--dim2)', textAlign: 'right', fontFamily: 'JetBrains Mono, monospace' }}>{rawVal}</td>
                        <td style={{ padding: '5px 8px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <div style={{ flex: 1, background: 'var(--border)', borderRadius: 3, height: 7, overflow: 'hidden' }}>
                              <div style={{ width: `${sc}%`, height: 7, background: scColor, borderRadius: 3, transition: 'width .4s' }} />
                            </div>
                            <span style={{ color: scColor, fontSize: 10, fontWeight: 700, width: 28, textAlign: 'right' }}>{sc.toFixed(0)}</span>
                          </div>
                        </td>
                        <td style={{ padding: '5px 8px', textAlign: 'right', fontSize: 10, color: 'var(--dim2)' }}>{wt}%</td>
                        <td style={{ padding: '5px 8px', fontSize: 10, color: 'var(--dim)', maxWidth: 240 }}>{f.interpretation || '—'}</td>
                      </tr>
                    );
                  })}
                  <tr style={{ borderTop: '1px solid var(--border)', background: 'var(--card2)' }}>
                    <td colSpan={2} style={{ padding: '5px 8px', fontSize: 10, color: 'var(--dim)', fontWeight: 700 }}>AGENT COMPOSITE</td>
                    <td style={{ padding: '5px 8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ flex: 1, background: 'var(--border)', borderRadius: 3, height: 7, overflow: 'hidden' }}>
                          <div style={{ width: `${(p * 100).toFixed(0)}%`, height: 7, background: pColor, borderRadius: 3 }} />
                        </div>
                        <span style={{ color: pColor, fontSize: 10, fontWeight: 700, width: 28, textAlign: 'right' }}>{(p * 100).toFixed(0)}</span>
                      </div>
                    </td>
                    <td style={{ padding: '5px 8px', textAlign: 'right', fontSize: 10, color: 'var(--dim2)' }}>100%</td>
                    <td style={{ padding: '5px 8px', fontSize: 10, color: 'var(--dim2)' }}>Conf: {(conf * 100).toFixed(0)}% · {ms.toFixed(0)}ms</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </td>
        </tr>
      )}

      {/* ── Deep Dive inline panel ── */}
      {deepOpen && (
        <tr className="fd-row">
          <td colSpan={6}>
            <div style={{ padding: '16px 14px', background: 'var(--surface)' }}>
              {/* Agent header strip */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 14, padding: '12px 16px', background: 'var(--card)', borderRadius: 10, border: '1px solid var(--border)', flexWrap: 'wrap' }}>
                <span style={{ padding: '5px 14px', borderRadius: 20, fontSize: 12, fontWeight: 800, letterSpacing: '.05em', color: voteColor, background: `${voteColor}18`, border: `1.5px solid ${voteColor}44` }}>{vote}</span>
                {[
                  { val: `${(p * 100).toFixed(1)}%`, label: 'P(Up)', color: pColor },
                  { val: `${(conf * 100).toFixed(0)}%`, label: 'Confidence', color: 'var(--cyan)' },
                  { val: `${ms.toFixed(0)}ms`, label: 'Compute', color: 'var(--dim)' },
                  { val: `${bullishFactors}/${nf}`, label: 'Bullish Factors', color: 'var(--green)' },
                  { val: `${bearishFactors}/${nf}`, label: 'Bearish Factors', color: 'var(--red)' },
                ].map(({ val, label, color }) => (
                  <div key={label} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'monospace', color }}>{val}</div>
                    <div style={{ fontSize: 9, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '.07em', marginTop: 1 }}>{label}</div>
                  </div>
                ))}
                <div style={{ flex: 1, minWidth: 200 }}>
                  <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 5 }}>Full Reasoning</div>
                  <div style={{ fontSize: 11, color: 'var(--dim2)', lineHeight: 1.65, background: 'var(--surface)', padding: '8px 10px', borderRadius: 6, borderLeft: '3px solid var(--blue)' }}>{fullReasoning || 'No reasoning provided.'}</div>
                </div>
              </div>

              {/* Factor cards grid */}
              <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 10 }}>
                📊 Factor Analysis — {nf} factors · scored 0–100 (0=Bearish · 50=Neutral · 100=Bullish)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 10 }}>
                {factors.map(([key, f]) => (
                  <FactorCard key={key} factorKey={key} f={f} />
                ))}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
