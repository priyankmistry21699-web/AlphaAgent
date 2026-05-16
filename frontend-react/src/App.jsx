import { useState, useEffect, useCallback } from 'react';
import Header from './components/Header.jsx';
import SignalTab from './components/Signal.jsx';
import MarketTab from './components/Market.jsx';
import PortfolioTab from './components/Portfolio.jsx';
import BacktestTab from './components/Backtest.jsx';
import { PaperTab, WalkFwdTab, StressTab, QuantLabTab, LeaderboardTab, OptimizerTab, SettingsTab } from './components/OtherTabs.jsx';

const TABS = [
  { id: 'market',      label: '🌐 Market' },
  { id: 'signal',      label: '📡 Signal' },
  { id: 'portfolio',   label: '💼 Portfolio' },
  { id: 'paper',       label: '📄 Paper' },
  { id: 'backtest',    label: '🔬 Backtest' },
  { id: 'walkfwd',     label: '🔄 Walk-Fwd' },
  { id: 'stress',      label: '⚡ Stress' },
  { id: 'quantlab',    label: '🔭 Quant Lab' },
  { id: 'leaderboard', label: '🏆 Leaderboard' },
  { id: 'optimizer',   label: '⚖️ Optimizer' },
  { id: 'settings',    label: '⚙️ Settings' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('market');
  const [livePrices, setLivePrices] = useState([]);

  // Live ticker bar — polls a few key symbols
  useEffect(() => {
    const syms = ['SPY', 'QQQ', 'BTC-USD', 'NVDA', 'AAPL', 'TSLA', 'GLD'];
    let cancelled = false;

    async function fetchPrices() {
      const results = await Promise.allSettled(
        syms.map(s => fetch(`/api/v1/signal/${s}?horizon=1d`).then(r => r.json()))
      );
      if (cancelled) return;
      const prices = results
        .map((r, i) => {
          if (r.status !== 'fulfilled') return null;
          const d = r.value;
          const p = d.probability || 0.5;
          return { sym: syms[i], prob: p, dir: d.direction || 'HOLD' };
        })
        .filter(Boolean);
      setLivePrices(prices);
    }

    fetchPrices();
    const timer = setInterval(fetchPrices, 60000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  const switchTab = useCallback((id) => setActiveTab(id), []);

  return (
    <div className="app">
      <Header livePrices={livePrices} onTabSwitch={switchTab} />

      <nav className="nav" style={{ top: 'var(--header-h)' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            className={`nav-tab ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => switchTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="content">
        <div className={`panel ${activeTab === 'market' ? 'active' : ''}`}>
          <MarketTab onAnalyze={(ticker) => { setActiveTab('signal'); window._pendingTicker = ticker; }} />
        </div>
        <div className={`panel ${activeTab === 'signal' ? 'active' : ''}`}>
          <SignalTab isActive={activeTab === 'signal'} />
        </div>
        <div className={`panel ${activeTab === 'portfolio' ? 'active' : ''}`}>
          <PortfolioTab isActive={activeTab === 'portfolio'} />
        </div>
        <div className={`panel ${activeTab === 'paper' ? 'active' : ''}`}>
          <PaperTab isActive={activeTab === 'paper'} />
        </div>
        <div className={`panel ${activeTab === 'backtest' ? 'active' : ''}`}>
          <BacktestTab isActive={activeTab === 'backtest'} />
        </div>
        <div className={`panel ${activeTab === 'walkfwd' ? 'active' : ''}`}>
          <WalkFwdTab isActive={activeTab === 'walkfwd'} />
        </div>
        <div className={`panel ${activeTab === 'stress' ? 'active' : ''}`}>
          <StressTab isActive={activeTab === 'stress'} />
        </div>
        <div className={`panel ${activeTab === 'quantlab' ? 'active' : ''}`}>
          <QuantLabTab isActive={activeTab === 'quantlab'} />
        </div>
        <div className={`panel ${activeTab === 'leaderboard' ? 'active' : ''}`}>
          <LeaderboardTab isActive={activeTab === 'leaderboard'} />
        </div>
        <div className={`panel ${activeTab === 'optimizer' ? 'active' : ''}`}>
          <OptimizerTab isActive={activeTab === 'optimizer'} />
        </div>
        <div className={`panel ${activeTab === 'settings' ? 'active' : ''}`}>
          <SettingsTab isActive={activeTab === 'settings'} />
        </div>
      </div>
    </div>
  );
}
