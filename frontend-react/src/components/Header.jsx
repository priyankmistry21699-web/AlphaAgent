export default function Header({ livePrices }) {
  const fmtPrice = (p) => p == null ? '—' : p >= 10000 ? p.toLocaleString('en-US', { maximumFractionDigits: 0 }) : p >= 1000 ? p.toLocaleString('en-US', { maximumFractionDigits: 1 }) : p.toFixed(2);
  const fmtPct   = (v) => v == null ? '' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;

  // Duplicate for seamless infinite marquee
  const pills = livePrices.length > 0 ? [...livePrices, ...livePrices] : [];

  return (
    <header className="header">
      <div className="brand">
        <div className="brand-icon">α</div>
        <div>
          <div className="brand-name">AlphaAgent</div>
          <div className="brand-sub">Quantitative Trading Intelligence</div>
        </div>
      </div>
      <div className="header-sep" />

      <div className="ticker-bar">
        <div className="ticker-track">
          {pills.map((p, idx) => {
            const up    = (p.change_pct ?? 0) >= 0;
            const color = up ? 'var(--green)' : 'var(--red)';
            return (
              <div key={idx} className="ticker-pill">
                <span className="sym">{p.sym}</span>
                {p.label && <span style={{ fontSize: 9, color: 'var(--dim)', fontWeight: 500 }}>{p.label}</span>}
                <span className="price" style={{ fontFamily: 'monospace', fontSize: 11 }}>{fmtPrice(p.price)}</span>
                <span className="chg" style={{ color }}>{up ? '▲' : '▼'} {fmtPct(p.change_pct)}</span>
              </div>
            );
          })}
          {livePrices.length === 0 && (
            <span style={{ color: 'var(--dim)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span className="live-dot" />Loading market prices…
            </span>
          )}
        </div>
      </div>

      <div className="header-actions">
        <button className="btn-icon" title="Refresh" onClick={() => window.location.reload()}>↻</button>
      </div>
    </header>
  );
}
