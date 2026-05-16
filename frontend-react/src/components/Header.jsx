export default function Header({ livePrices }) {
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
        {livePrices.map(p => {
          const color = p.dir === 'LONG' ? 'var(--green)' : p.dir === 'SHORT' ? 'var(--red)' : 'var(--yellow)';
          const pct = ((p.prob - 0.5) * 100).toFixed(1);
          const sign = p.prob >= 0.5 ? '+' : '';
          return (
            <div key={p.sym} className="ticker-pill">
              <span className="sym">{p.sym}</span>
              <span className="price" style={{ color }}>{p.dir}</span>
              <span className="chg" style={{ color }}>{sign}{pct}%</span>
            </div>
          );
        })}
        {livePrices.length === 0 && (
          <span style={{ color: 'var(--dim)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span className="live-dot" />Loading live signals…
          </span>
        )}
      </div>
      <div className="header-actions">
        <button className="btn-icon" title="Refresh" onClick={() => window.location.reload()}>↻</button>
      </div>
    </header>
  );
}
