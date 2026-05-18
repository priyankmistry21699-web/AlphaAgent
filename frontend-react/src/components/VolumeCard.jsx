import { useState, useEffect } from 'react';
import { api } from '../api.js';

const PERIODS = [
  { label: '1D', period: '1d',  interval: '5m'  },
  { label: '1W', period: '5d',  interval: '30m' },
  { label: '1M', period: '1mo', interval: '1d'  },
  { label: '3M', period: '3mo', interval: '1d'  },
  { label: '1Y', period: '1y',  interval: '1wk' },
];

function fmtVol(v) {
  if (v == null || isNaN(v)) return '—';
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B';
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
  return v.toFixed(0);
}

export default function VolumeCard({ ticker }) {
  const [pIdx, setPIdx]     = useState(2);       // default 1M
  const [data, setData]     = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!ticker) { setData([]); return; }
    const p = PERIODS[pIdx];
    setLoading(true);
    setData([]);
    api.tickerChart(ticker, p.period, p.interval)
      .then(d => setData(d.data || []))
      .catch(() => setData([]))
      .finally(() => setLoading(false));
  }, [ticker, pIdx]);

  // ── Aggregate stats ──────────────────────────────────────────────────
  let totalVol = 0, buyVol = 0, sellVol = 0;
  const bars = data.filter(d => d.volume > 0 && d.high != null && d.low != null && d.close != null);

  bars.forEach(d => {
    const range = d.high - d.low;
    const bFrac = range > 0 ? Math.max(0, Math.min(1, (d.close - d.low) / range)) : 0.5;
    const bv = d.volume * bFrac;
    const sv = d.volume - bv;
    totalVol += d.volume;
    buyVol   += bv;
    sellVol  += sv;
  });

  const buyPct  = totalVol > 0 ? (buyVol  / totalVol * 100) : 50;
  const sellPct = totalVol > 0 ? (sellVol / totalVol * 100) : 50;
  const days    = bars.length;
  const avgDaily = days > 0 ? totalVol / days : 0;
  const colorHex = buyPct >= 50 ? '#00d68f' : '#f5365c';

  // ── Mini bar chart (up to 30 most recent bars) ──────────────────────
  const chartBars = bars.slice(-30);
  const maxBarVol = Math.max(...chartBars.map(d => d.volume), 1);

  return (
    <div className="card" style={{ background: 'var(--surface)' }}>
      {/* Header */}
      <div style={{ padding: '10px 12px 0' }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--cyan)', marginBottom: 7 }}>
          📊 Volume Analysis{ticker ? ` — ${ticker}` : ''}
        </div>
        {/* Period buttons */}
        <div style={{ display: 'flex', gap: 4 }}>
          {PERIODS.map((p, i) => (
            <button key={p.label} onClick={() => setPIdx(i)} style={{
              padding: '2px 8px', fontSize: 10, borderRadius: 4, border: 'none', cursor: 'pointer',
              background: pIdx === i ? colorHex : 'var(--card2)',
              color: pIdx === i ? '#fff' : 'var(--dim)',
              fontWeight: pIdx === i ? 700 : 400,
              transition: 'all .12s',
            }}>{p.label}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: '10px 12px 12px' }}>
        {loading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 80 }}>
            <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
          </div>
        ) : data.length === 0 ? (
          <div style={{ color: 'var(--dim)', fontSize: 11, textAlign: 'center', padding: '16px 0' }}>No data</div>
        ) : (
          <>
            {/* Total + avg */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 10px', marginBottom: 10 }}>
              {[
                { label: 'Total Volume', val: fmtVol(totalVol), color: 'var(--text)' },
                { label: 'Avg per Bar',  val: fmtVol(avgDaily), color: 'var(--dim2)' },
              ].map(({ label, val, color }) => (
                <div key={label} style={{ background: 'var(--card2)', borderRadius: 6, padding: '6px 8px' }}>
                  <div style={{ fontSize: 8, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 2 }}>{label}</div>
                  <div style={{ fontSize: 14, fontWeight: 800, fontFamily: 'monospace', color }}>{val}</div>
                </div>
              ))}
            </div>

            {/* Buy/Sell breakdown bars */}
            <div style={{ marginBottom: 10 }}>
              {/* Buy */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 5 }}>
                <div style={{ width: 28, fontSize: 8, fontWeight: 700, color: 'var(--green)', textTransform: 'uppercase', flexShrink: 0 }}>Buy</div>
                <div style={{ flex: 1, height: 12, background: 'var(--card2)', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ width: `${buyPct}%`, height: '100%', background: 'var(--green)', borderRadius: 3, opacity: .85, transition: 'width .4s' }} />
                </div>
                <div style={{ width: 34, fontSize: 9, fontFamily: 'monospace', color: 'var(--green)', fontWeight: 700, flexShrink: 0, textAlign: 'right' }}>{buyPct.toFixed(1)}%</div>
              </div>
              {/* Sell */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <div style={{ width: 28, fontSize: 8, fontWeight: 700, color: 'var(--red)', textTransform: 'uppercase', flexShrink: 0 }}>Sell</div>
                <div style={{ flex: 1, height: 12, background: 'var(--card2)', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ width: `${sellPct}%`, height: '100%', background: 'var(--red)', borderRadius: 3, opacity: .85, transition: 'width .4s' }} />
                </div>
                <div style={{ width: 34, fontSize: 9, fontFamily: 'monospace', color: 'var(--red)', fontWeight: 700, flexShrink: 0, textAlign: 'right' }}>{sellPct.toFixed(1)}%</div>
              </div>
            </div>

            {/* Buy/sell volume amounts */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
              <div style={{ background: 'rgba(0,214,143,.07)', border: '1px solid rgba(0,214,143,.2)', borderRadius: 6, padding: '5px 8px', textAlign: 'center' }}>
                <div style={{ fontSize: 8, color: 'var(--green)', fontWeight: 700, marginBottom: 2 }}>BUY VOLUME</div>
                <div style={{ fontSize: 13, fontWeight: 800, fontFamily: 'monospace', color: 'var(--green)' }}>{fmtVol(buyVol)}</div>
              </div>
              <div style={{ background: 'rgba(245,54,92,.07)', border: '1px solid rgba(245,54,92,.2)', borderRadius: 6, padding: '5px 8px', textAlign: 'center' }}>
                <div style={{ fontSize: 8, color: 'var(--red)', fontWeight: 700, marginBottom: 2 }}>SELL VOLUME</div>
                <div style={{ fontSize: 13, fontWeight: 800, fontFamily: 'monospace', color: 'var(--red)' }}>{fmtVol(sellVol)}</div>
              </div>
            </div>

            {/* Mini stacked bar chart */}
            {chartBars.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 8, color: 'var(--dim)', marginBottom: 4 }}>Volume per bar (green=buy · red=sell)</div>
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height: 42 }}>
                  {chartBars.map((d, i) => {
                    const range  = d.high - d.low;
                    const bFrac  = range > 0 ? Math.max(0, Math.min(1, (d.close - d.low) / range)) : 0.5;
                    const h      = Math.max(2, Math.round((d.volume / maxBarVol) * 42));
                    const bH     = Math.round(h * bFrac);
                    const sH     = h - bH;
                    return (
                      <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'stretch', height: h }}>
                        <div style={{ height: sH, background: 'rgba(245,54,92,.7)', borderRadius: '1px 1px 0 0', minHeight: sH > 0 ? 1 : 0 }} />
                        <div style={{ height: bH, background: 'rgba(0,214,143,.7)', borderRadius: '0 0 1px 1px', minHeight: bH > 0 ? 1 : 0 }} />
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Disclaimer */}
            <div style={{ fontSize: 8, color: 'var(--dim)', lineHeight: 1.5 }}>
              ⚠ Buy/sell split estimated from price action (close position within high-low range). Not actual order flow.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
