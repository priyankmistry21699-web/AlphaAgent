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

function AgentTags({ agentName }) {
  const info = AGENT_THEORIES[agentName];
  if (!info) return null;
  return (
    <div style={{ marginTop: 4, lineHeight: 2 }}>
      {info.tags.slice(0, 5).map(tag => (
        <span key={tag} className={`tag ${info.color}`}>{tag}</span>
      ))}
    </div>
  );
}

function openDeepDive(agent, ticker) {
  const factors = Object.entries(agent.factor_scores || {});
  const vote = (agent.vote || 'HOLD').toUpperCase();
  const voteColor = vote === 'LONG' ? '#00d68f' : vote === 'SHORT' ? '#f5365c' : '#ffb703';
  const prob = ((agent.probability_up || 0.5) * 100).toFixed(1);
  const conf = ((agent.confidence || 0) * 100).toFixed(0);
  const ms = (agent.computation_time_ms || 0).toFixed(0);
  const name = agent.agent_name || agent.name || 'Unknown';
  const reasoning = agent.reasoning || 'No reasoning provided.';

  const factorCards = factors.map(([key, f]) => {
    const sc = f.score ?? 50;
    const scColor = sc >= 66 ? '#00d68f' : sc <= 33 ? '#f5365c' : '#ffb703';
    const enc = FACTOR_ENC[key] || {};
    const signal = sc >= 66 ? 'BULLISH' : sc <= 33 ? 'BEARISH' : 'NEUTRAL';
    const rawVal = typeof f.value === 'number'
      ? (Math.abs(f.value) >= 1000 ? f.value.toLocaleString('en-US', { maximumFractionDigits: 0 })
        : Math.abs(f.value) < 0.01 && f.value !== 0 ? f.value.toExponential(2)
        : f.value.toFixed(3))
      : String(f.value ?? '—');
    const bgColor = sc >= 66 ? 'rgba(0,214,143,0.06)' : sc <= 33 ? 'rgba(245,54,92,0.06)' : 'rgba(255,183,3,0.04)';
    const borderColor = sc >= 66 ? 'rgba(0,214,143,0.25)' : sc <= 33 ? 'rgba(245,54,92,0.25)' : 'rgba(255,183,3,0.15)';
    const rangeRows = enc.great
      ? `<table class="range-table">
          <tr><td class="rb great">●</td><td class="rl">GREAT</td><td>${enc.great}</td></tr>
          <tr><td class="rb good">●</td><td class="rl">GOOD</td><td>${enc.good || '—'}</td></tr>
          <tr><td class="rb neut">●</td><td class="rl">NEUTRAL</td><td>${enc.neutral || '—'}</td></tr>
          <tr><td class="rb bad">●</td><td class="rl">BAD</td><td>${enc.bad || '—'}</td></tr>
          <tr><td class="rb terr">●</td><td class="rl">TERRIBLE</td><td>${enc.terrible || '—'}</td></tr>
        </table>`
      : `<div style="font-size:10px;color:#5d7ba0;font-style:italic">Range data not available for this factor.</div>`;

    return `<div class="fc" style="background:${bgColor};border:1px solid ${borderColor}">
      <div class="fc-hdr">
        <div>
          <div class="fc-name">${f.name || key}</div>
          ${enc.fullName && enc.fullName !== (f.name || key) ? `<div class="fc-full">${enc.fullName}</div>` : ''}
          ${enc.category ? `<span class="fc-cat">${enc.category}</span>` : ''}
        </div>
        <div class="fc-right">
          <div class="fc-sig" style="color:${scColor};background:${scColor}18;border:1px solid ${scColor}33">${signal}</div>
          <div class="fc-score" style="color:${scColor}">${sc.toFixed(0)}<span class="fc-sm">/100</span></div>
        </div>
      </div>
      <div class="fc-bar-wrap">
        <div class="fc-bar-bg">
          <div class="bz bad-z"></div><div class="bz neut-z"></div><div class="bz good-z"></div>
        </div>
        <div class="fc-fill" style="width:${sc}%;background:${scColor}"></div>
        <div class="fc-ptr" style="left:${sc}%"></div>
        <div class="fc-lbl"><span>0 Bearish</span><span>50 Neutral</span><span>Bullish 100</span></div>
      </div>
      <div class="fc-body">
        <div class="fc-val-row">
          <span class="fc-vl">Current Value</span>
          <span class="fc-vn">${rawVal}${enc.valueUnit ? ` <span class="fc-unit">${enc.valueUnit}</span>` : ''}</span>
        </div>
        ${enc.formula ? `<div class="fc-form">📐 ${enc.formula}</div>` : ''}
        <div class="fc-interp">${f.interpretation || '—'}</div>
        ${rangeRows}
        ${enc.note ? `<div class="fc-note">💡 ${enc.note}</div>` : ''}
      </div>
    </div>`;
  }).join('');

  const html = `<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Factor Deep Dive — ${name} — ${ticker}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#07111f;color:#e4ecf7;font-size:13px;line-height:1.6}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:#07111f}::-webkit-scrollbar-thumb{background:#1c2f4a;border-radius:2px}
.ph{background:linear-gradient(135deg,#0c1a2e,#111f35);border-bottom:1px solid #1c2f4a;padding:14px 24px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;backdrop-filter:blur(12px)}
.ph-l{display:flex;align-items:center;gap:12px}
.ph-brand{font-size:18px;font-weight:900;background:linear-gradient(135deg,#3d9eff,#7b5ea7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.ph-agent{font-size:15px;font-weight:700}
.ph-ticker{font-size:11px;color:#5d7ba0;background:#0c1a2e;padding:3px 10px;border-radius:20px;border:1px solid #1c2f4a}
.ac{margin:18px 24px;background:#0c1a2e;border-radius:14px;border:1px solid #1c2f4a;overflow:hidden}
.ac-hdr{padding:16px 20px;border-bottom:1px solid #1c2f4a;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.vb{padding:6px 16px;border-radius:20px;font-size:13px;font-weight:800;letter-spacing:.05em}
.ac-ms{display:flex;gap:20px;flex:1;flex-wrap:wrap}
.ac-m{text-align:center}
.ac-mv{font-size:20px;font-weight:800;font-family:'JetBrains Mono',monospace}
.ac-ml{font-size:10px;color:#5d7ba0;text-transform:uppercase;letter-spacing:.08em}
.ac-body{padding:16px 20px}
.rl-lbl{font-size:10px;font-weight:700;color:#5d7ba0;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.rl-text{font-size:12.5px;color:#8da3c0;line-height:1.7;background:#07111f;padding:14px;border-radius:8px;border-left:3px solid #3d9eff}
.st{padding:0 24px 12px;font-size:11px;font-weight:700;color:#5d7ba0;text-transform:uppercase;letter-spacing:.1em;margin-top:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;padding:0 24px 24px}
.fc{border-radius:12px;overflow:hidden;transition:transform .2s,box-shadow .2s}
.fc:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.5)}
.fc-hdr{display:flex;justify-content:space-between;align-items:flex-start;padding:14px 16px 8px;gap:12px}
.fc-name{font-size:13px;font-weight:700;color:#e4ecf7}
.fc-full{font-size:10px;color:#5d7ba0;margin-top:2px}
.fc-cat{font-size:9px;font-weight:700;color:#06c8f0;background:#06c8f015;padding:2px 7px;border-radius:10px;border:1px solid #06c8f025;margin-top:5px;display:inline-block;letter-spacing:.06em}
.fc-right{display:flex;flex-direction:column;align-items:flex-end;gap:6px;flex-shrink:0}
.fc-sig{font-size:9px;font-weight:800;padding:3px 8px;border-radius:8px;letter-spacing:.07em}
.fc-score{font-size:26px;font-weight:900;line-height:1;font-family:'JetBrains Mono',monospace}
.fc-sm{font-size:12px;color:#5d7ba0;font-weight:400}
.fc-bar-wrap{padding:4px 16px 10px;position:relative}
.fc-bar-bg{height:8px;border-radius:4px;display:flex;overflow:hidden}
.bz{height:100%}
.bad-z{width:33%;background:rgba(245,54,92,.2)}
.neut-z{width:34%;background:rgba(255,183,3,.15)}
.good-z{width:33%;background:rgba(0,214,143,.2)}
.fc-fill{position:absolute;top:4px;left:16px;height:8px;border-radius:4px;transition:width .6s;max-width:calc(100% - 32px)}
.fc-ptr{position:absolute;top:1px;width:3px;height:14px;background:#fff;border-radius:2px;transform:translateX(-50%);box-shadow:0 0 6px rgba(255,255,255,.4)}
.fc-lbl{display:flex;justify-content:space-between;font-size:9px;color:#5d7ba0;margin-top:4px}
.fc-body{padding:0 16px 14px;display:flex;flex-direction:column;gap:8px}
.fc-val-row{display:flex;align-items:center;justify-content:space-between}
.fc-vl{font-size:10px;color:#5d7ba0;font-weight:600}
.fc-vn{font-size:14px;font-weight:700;color:#e4ecf7;font-family:'JetBrains Mono',monospace}
.fc-unit{font-size:10px;color:#5d7ba0;font-weight:400;font-family:inherit}
.fc-form{font-size:10.5px;color:#8da3c0;background:#07111f;padding:8px 10px;border-radius:6px;border-left:3px solid #3d9eff;font-family:'JetBrains Mono',monospace;line-height:1.5}
.fc-interp{font-size:11px;color:#8da3c0;font-style:italic}
.range-table{width:100%;border-collapse:collapse;font-size:10.5px}
.range-table td{padding:3px 5px;vertical-align:top}
.rb{width:18px;font-size:11px}
.rl{font-weight:700;width:60px;color:#5d7ba0;text-transform:uppercase;font-size:9px;letter-spacing:.05em}
.rb.great{color:#00d68f}.rb.good{color:#22c55e}.rb.neut{color:#ffb703}.rb.bad{color:#f97316}.rb.terr{color:#f5365c}
.fc-note{font-size:10.5px;color:#5d7ba0;padding:7px 10px;background:#0d1825;border-radius:6px;border-left:3px solid #7b5ea7}
.sf{margin:0 24px 24px;background:linear-gradient(135deg,#0c1a2e,#111f35);border-radius:12px;border:1px solid #1c2f4a;padding:16px 20px}
.sf-title{font-size:11px;font-weight:700;color:#5d7ba0;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
.sf-row{display:flex;gap:12px;margin-top:10px;flex-wrap:wrap}
.sf-sc{text-align:center;flex:1;min-width:100px;background:#07111f;border-radius:8px;padding:10px}
.sf-sv{font-size:18px;font-weight:800;font-family:'JetBrains Mono',monospace}
.sf-sl{font-size:9px;color:#5d7ba0;text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
</style>
</head>
<body>
<div class="ph">
  <div class="ph-l">
    <span class="ph-brand">AlphaAgent</span>
    <span style="color:#1c2f4a;font-size:18px">|</span>
    <span class="ph-agent">${name}</span>
    <span class="ph-ticker">${ticker}</span>
  </div>
  <span style="font-size:10px;color:#5d7ba0">Factor Deep Dive · ${factors.length} factors analyzed</span>
</div>
<div class="ac">
  <div class="ac-hdr">
    <div class="vb" style="color:${voteColor};background:${voteColor}18;border:1.5px solid ${voteColor}44">${vote}</div>
    <div class="ac-ms">
      <div class="ac-m"><div class="ac-mv" style="color:${voteColor}">${prob}%</div><div class="ac-ml">P(Up)</div></div>
      <div class="ac-m"><div class="ac-mv" style="color:#06c8f0">${conf}%</div><div class="ac-ml">Confidence</div></div>
      <div class="ac-m"><div class="ac-mv" style="color:#5d7ba0">${ms}ms</div><div class="ac-ml">Compute</div></div>
      <div class="ac-m"><div class="ac-mv" style="color:#a78bfa">${factors.length}</div><div class="ac-ml">Factors</div></div>
    </div>
  </div>
  <div class="ac-body">
    <div class="rl-lbl">Full Agent Reasoning</div>
    <div class="rl-text">${reasoning}</div>
  </div>
</div>
<div class="st">📊 Factor Analysis — Each scored 0–100 (0=Bearish · 50=Neutral · 100=Bullish)</div>
<div class="grid">${factorCards}</div>
<div class="sf">
  <div class="sf-title">Agent Composite Summary</div>
  <div style="font-size:12px;color:#8da3c0;line-height:1.7">${reasoning}</div>
  <div class="sf-row">
    <div class="sf-sc"><div class="sf-sv" style="color:${voteColor}">${vote}</div><div class="sf-sl">Direction</div></div>
    <div class="sf-sc"><div class="sf-sv" style="color:${voteColor}">${prob}%</div><div class="sf-sl">P(Up)</div></div>
    <div class="sf-sc"><div class="sf-sv" style="color:#06c8f0">${conf}%</div><div class="sf-sl">Confidence</div></div>
    <div class="sf-sc"><div class="sf-sv" style="color:#a78bfa">${factors.filter(([, f]) => (f.score ?? 50) >= 66).length}/${factors.length}</div><div class="sf-sl">Bullish Factors</div></div>
  </div>
</div>
</body></html>`;

  const w = window.open('', '_blank');
  if (!w) { alert('Please allow popups to view Factor Deep Dive.'); return; }
  w.document.write(html);
  w.document.close();
}

export default function AgentRow({ agent, ticker, hzLabel }) {
  const [expanded, setExpanded] = useState(false);
  const name = agent.agent_name || agent.name || 'Unknown';
  const vote = (agent.vote || 'HOLD').toUpperCase();
  const p = agent.probability_up ?? 0.5;
  const conf = agent.confidence ?? 0;
  const ms = agent.computation_time_ms ?? 0;
  const reasoning = (agent.reasoning || '').slice(0, 160);
  const factors = Object.entries(agent.factor_scores || {});
  const nf = factors.length;
  const wt = nf > 0 ? (100 / nf).toFixed(1) : 0;
  const pColor = priceColor(p);
  const voteClass = vote === 'LONG' ? 'badge-long' : vote === 'SHORT' ? 'badge-short' : 'badge-hold';

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
                  onClick={(e) => { e.stopPropagation(); openDeepDive(agent, ticker); }}
                  title="Open Factor Deep Dive in new tab"
                >
                  🔬 Deep Dive
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
        <td style={{ color: 'var(--dim2)', fontSize: 11, maxWidth: 260 }}>
          {reasoning}{(agent.reasoning || '').length > 160 ? '…' : ''}
        </td>
      </tr>

      {expanded && nf > 0 && (
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
    </>
  );
}
