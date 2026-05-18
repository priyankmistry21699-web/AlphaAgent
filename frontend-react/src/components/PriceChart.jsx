import { useState, useEffect } from 'react';
import {
  ComposedChart, Area, Bar, XAxis, YAxis,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { api } from '../api.js';

const RANGES = [
  { label: '1D', period: '1d',  interval: '5m'  },
  { label: '1W', period: '5d',  interval: '30m' },
  { label: '1M', period: '1mo', interval: '1d'  },
  { label: '3M', period: '3mo', interval: '1d'  },
  { label: '6M', period: '6mo', interval: '1d'  },
  { label: '1Y', period: '1y',  interval: '1wk' },
  { label: '5Y', period: '5y',  interval: '1mo' },
];

function fmtDate(dateStr, interval) {
  if (!dateStr) return '';
  const d = new Date(dateStr);
  if (interval === '5m' || interval === '30m' || interval === '1h')
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
  if (interval === '1d')
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  if (interval === '1wk')
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
}

function fmtP(v) {
  if (v == null) return '—';
  if (Math.abs(v) >= 10000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
  if (Math.abs(v) >= 1000)  return v.toLocaleString('en-US', { maximumFractionDigits: 2 });
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(3);
  return Number(v).toFixed(2);
}

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  const up = (d.close ?? 0) >= (d.open ?? d.close ?? 0);
  const c  = up ? 'var(--green)' : 'var(--red)';
  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '8px 12px', fontSize: 10, lineHeight: 1.9,
      boxShadow: '0 4px 16px rgba(0,0,0,.4)',
    }}>
      <div style={{ color: 'var(--dim)', marginBottom: 2, fontSize: 9 }}>{d.date}</div>
      <div style={{ color: c, fontWeight: 800, fontSize: 13, fontFamily: 'monospace' }}>
        ${fmtP(d.close)}
      </div>
      {d.open  != null && <div style={{ display: 'flex', gap: 10 }}><span style={{ color: 'var(--dim)', width: 12 }}>O</span><span style={{ fontFamily: 'monospace' }}>{fmtP(d.open)}</span></div>}
      {d.high  != null && <div style={{ display: 'flex', gap: 10 }}><span style={{ color: 'var(--green)', width: 12 }}>H</span><span style={{ fontFamily: 'monospace', color: 'var(--green)' }}>{fmtP(d.high)}</span></div>}
      {d.low   != null && <div style={{ display: 'flex', gap: 10 }}><span style={{ color: 'var(--red)', width: 12 }}>L</span><span style={{ fontFamily: 'monospace', color: 'var(--red)' }}>{fmtP(d.low)}</span></div>}
      {d.volume > 0    && <div style={{ color: 'var(--dim)', marginTop: 2 }}>Vol {d.volume >= 1e6 ? (d.volume / 1e6).toFixed(2) + 'M' : (d.volume / 1e3).toFixed(1) + 'K'}</div>}
    </div>
  );
}

export default function PriceChart({ ticker, sector, currentPrice }) {
  const [rangeIdx, setRangeIdx]   = useState(2); // default 1M
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading]     = useState(false);

  useEffect(() => {
    if (!ticker) { setChartData([]); return; }
    const r = RANGES[rangeIdx];
    setLoading(true);
    setChartData([]);
    api.tickerChart(ticker, r.period, r.interval)
      .then(d => setChartData(d.data || []))
      .catch(() => setChartData([]))
      .finally(() => setLoading(false));
  }, [ticker, rangeIdx]);

  const interval = RANGES[rangeIdx].interval;
  const prices   = chartData.map(d => d.close).filter(v => v != null);
  const first    = prices[0] ?? 0;
  const last     = prices[prices.length - 1] ?? 0;
  const isUp     = last >= first;
  const pctChg   = first ? ((last - first) / first * 100) : 0;
  const colorHex = isUp ? '#00d68f' : '#f5365c';
  const colorVar = isUp ? 'var(--green)' : 'var(--red)';

  const minP  = Math.min(...prices);
  const maxP  = Math.max(...prices);
  const pad   = (maxP - minP) * 0.1 || Math.abs(maxP) * 0.03;
  const maxVol = Math.max(...chartData.map(d => d.volume || 0), 1);

  // Sparse X-axis ticks
  const tickCount = 5;
  const step = Math.max(1, Math.floor(chartData.length / tickCount));
  const ticks = chartData.filter((_, i) => i % step === 0).map(d => d.date);

  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden' }}>

      {/* Header row */}
      <div style={{ padding: '8px 12px 0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--cyan)' }}>📈 {ticker}</span>
          {sector && (
            <span style={{ fontSize: 9, color: 'var(--dim)', background: 'var(--card2)', padding: '1px 5px', borderRadius: 4 }}>
              {sector}
            </span>
          )}
        </div>
        {prices.length >= 2 && (
          <span style={{ fontSize: 11, fontWeight: 700, fontFamily: 'monospace', color: colorVar }}>
            {isUp ? '▲' : '▼'} {Math.abs(pctChg).toFixed(2)}%
          </span>
        )}
      </div>

      {/* Timeframe buttons */}
      <div style={{ display: 'flex', gap: 3, padding: '5px 8px' }}>
        {RANGES.map((r, i) => (
          <button key={r.label} onClick={() => setRangeIdx(i)} style={{
            padding: '2px 7px', fontSize: 10, borderRadius: 4, border: 'none', cursor: 'pointer',
            background: rangeIdx === i ? colorHex : 'var(--card2)',
            color: rangeIdx === i ? '#fff' : 'var(--dim)',
            fontWeight: rangeIdx === i ? 700 : 400,
            transition: 'all .12s',
          }}>{r.label}</button>
        ))}
      </div>

      {/* Chart area */}
      <div style={{ height: 185, padding: '0 2px 4px' }}>
        {loading ? (
          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
          </div>
        ) : chartData.length === 0 ? (
          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--dim)', fontSize: 11 }}>
            No chart data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 4, right: 6, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id={`cg_${ticker.replace(/[^a-z0-9]/gi, '')}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={colorHex} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={colorHex} stopOpacity={0.01} />
                </linearGradient>
              </defs>

              <XAxis
                dataKey="date"
                ticks={ticks}
                tickFormatter={d => fmtDate(d, interval)}
                tick={{ fontSize: 8, fill: 'var(--dim)' }}
                axisLine={false} tickLine={false}
              />
              <YAxis
                domain={[minP - pad, maxP + pad]}
                tick={{ fontSize: 8, fill: 'var(--dim)' }}
                axisLine={false} tickLine={false}
                width={46}
                tickFormatter={v => fmtP(v)}
              />
              {/* Volume on a separate hidden axis scaled to bottom 18% */}
              <YAxis yAxisId="vol" orientation="right" hide domain={[0, maxVol * 5.5]} />

              <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'var(--dim)', strokeWidth: 1, strokeDasharray: '3 3' }} />

              {/* Current price reference line */}
              {currentPrice && (
                <ReferenceLine
                  y={currentPrice}
                  stroke={colorHex} strokeDasharray="4 3" strokeOpacity={0.6} strokeWidth={1}
                />
              )}

              {/* Volume bars */}
              <Bar
                yAxisId="vol"
                dataKey="volume"
                fill={colorHex}
                opacity={0.18}
                isAnimationActive={false}
                radius={[1, 1, 0, 0]}
              />

              {/* Price line + area fill */}
              <Area
                type="monotone"
                dataKey="close"
                stroke={colorHex}
                strokeWidth={1.6}
                fill={`url(#cg_${ticker.replace(/[^a-z0-9]/gi, '')})`}
                dot={false}
                activeDot={{ r: 3.5, fill: colorHex, stroke: 'var(--surface)', strokeWidth: 1.5 }}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
