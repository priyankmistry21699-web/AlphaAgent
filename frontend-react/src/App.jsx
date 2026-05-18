import { useState, useEffect, useCallback } from 'react';
import Header from './components/Header.jsx';
import SignalTab from './components/Signal.jsx';
import MarketTab from './components/Market.jsx';
import PortfolioTab from './components/Portfolio.jsx';
import BacktestTab from './components/Backtest.jsx';
import { PaperTab, WalkFwdTab, StressTab, QuantLabTab, LeaderboardTab, OptimizerTab, SettingsTab } from './components/OtherTabs.jsx';
import ErrorBoundary from './components/ErrorBoundary.jsx';

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

// US indices & ETFs | Tech | Crypto | Metals/Commodities | Asian | European
const TICKER_SYMS = [
  // US Indices
  { sym: 'SPY',     label: 'S&P 500'   },
  { sym: 'QQQ',     label: 'NASDAQ'    },
  { sym: 'DIA',     label: 'Dow Jones' },
  { sym: 'IWM',     label: 'Russell'   },
  // Tech
  { sym: 'AAPL',    label: 'Apple'     },
  { sym: 'MSFT',    label: 'Microsoft' },
  { sym: 'NVDA',    label: 'NVIDIA'    },
  { sym: 'TSLA',    label: 'Tesla'     },
  { sym: 'AMZN',    label: 'Amazon'    },
  { sym: 'GOOGL',   label: 'Google'    },
  { sym: 'META',    label: 'Meta'      },
  // Crypto
  { sym: 'BTC-USD', label: 'Bitcoin'   },
  { sym: 'ETH-USD', label: 'Ethereum'  },
  // Metals & Commodities
  { sym: 'GLD',     label: 'Gold'      },
  { sym: 'SLV',     label: 'Silver'    },
  { sym: 'USO',     label: 'Oil'       },
  { sym: 'UNG',     label: 'Nat Gas'   },
  { sym: 'PDBC',    label: 'Commodities'},
  // Asian Indices (ETFs)
  { sym: 'EWJ',     label: 'Nikkei'    },
  { sym: 'FXI',     label: 'China'     },
  { sym: 'INDA',    label: 'Nifty/India'},
  { sym: 'EWH',     label: 'HK/HSI'   },
  { sym: 'EWY',     label: 'Korea'     },
  { sym: 'EWT',     label: 'Taiwan'    },
  // European Indices (ETFs)
  { sym: 'EWU',     label: 'FTSE/UK'   },
  { sym: 'EWG',     label: 'DAX/Germany'},
  { sym: 'EWQ',     label: 'CAC/France'},
  { sym: 'EZU',     label: 'Eurozone'  },
  // Bonds & Dollar
  { sym: 'TLT',     label: 'Bonds'     },
  { sym: 'UUP',     label: 'USD Index' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('market');
  const [livePrices, setLivePrices] = useState([]);

  // Live ticker bar — uses fast quotes endpoint (price + % change, ~2s load)
  useEffect(() => {
    let cancelled = false;
    const syms = TICKER_SYMS.map(t => t.sym);

    async function fetchPrices() {
      try {
        const res = await fetch('/api/v1/market/quotes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tickers: syms }),
        });
        if (!res.ok || cancelled) return;
        const data = await res.json();
        const quotes = data.quotes || {};
        const prices = TICKER_SYMS
          .map(({ sym, label }) => {
            const q = quotes[sym];
            if (!q) return null;
            return { sym, label, price: q.price, change_pct: q.change_pct ?? 0 };
          })
          .filter(Boolean);
        if (!cancelled) setLivePrices(prices);
      } catch {}
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
          <ErrorBoundary key="market">
            <MarketTab onAnalyze={(ticker) => { setActiveTab('signal'); window._pendingTicker = ticker; }} isActive={activeTab === 'market'} />
          </ErrorBoundary>
        </div>
        <div className={`panel ${activeTab === 'signal' ? 'active' : ''}`}>
          <ErrorBoundary key="signal">
            <SignalTab isActive={activeTab === 'signal'} />
          </ErrorBoundary>
        </div>
        <div className={`panel ${activeTab === 'portfolio' ? 'active' : ''}`}>
          <ErrorBoundary key="portfolio">
            <PortfolioTab isActive={activeTab === 'portfolio'} />
          </ErrorBoundary>
        </div>
        <div className={`panel ${activeTab === 'paper' ? 'active' : ''}`}>
          <ErrorBoundary key="paper">
            <PaperTab isActive={activeTab === 'paper'} />
          </ErrorBoundary>
        </div>
        <div className={`panel ${activeTab === 'backtest' ? 'active' : ''}`}>
          <ErrorBoundary key="backtest">
            <BacktestTab isActive={activeTab === 'backtest'} />
          </ErrorBoundary>
        </div>
        <div className={`panel ${activeTab === 'walkfwd' ? 'active' : ''}`}>
          <ErrorBoundary key="walkfwd">
            <WalkFwdTab isActive={activeTab === 'walkfwd'} />
          </ErrorBoundary>
        </div>
        <div className={`panel ${activeTab === 'stress' ? 'active' : ''}`}>
          <ErrorBoundary key="stress">
            <StressTab isActive={activeTab === 'stress'} />
          </ErrorBoundary>
        </div>
        <div className={`panel ${activeTab === 'quantlab' ? 'active' : ''}`}>
          <ErrorBoundary key="quantlab">
            <QuantLabTab isActive={activeTab === 'quantlab'} />
          </ErrorBoundary>
        </div>
        <div className={`panel ${activeTab === 'leaderboard' ? 'active' : ''}`}>
          <ErrorBoundary key="leaderboard">
            <LeaderboardTab isActive={activeTab === 'leaderboard'} />
          </ErrorBoundary>
        </div>
        <div className={`panel ${activeTab === 'optimizer' ? 'active' : ''}`}>
          <ErrorBoundary key="optimizer">
            <OptimizerTab isActive={activeTab === 'optimizer'} />
          </ErrorBoundary>
        </div>
        <div className={`panel ${activeTab === 'settings' ? 'active' : ''}`}>
          <ErrorBoundary key="settings">
            <SettingsTab isActive={activeTab === 'settings'} />
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
}
