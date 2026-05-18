import { Component } from 'react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('[AlphaAgent] Render error:', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', minHeight: 300, gap: 16, padding: 32,
        }}>
          <div style={{ fontSize: 32 }}>⚠</div>
          <div style={{ color: 'var(--text)', fontWeight: 700, fontSize: 15 }}>
            Something went wrong rendering this panel
          </div>
          <div style={{ color: 'var(--dim)', fontSize: 12, maxWidth: 420, textAlign: 'center' }}>
            {this.state.error?.message || 'Unknown error'}
          </div>
          <button
            className="btn btn-primary"
            onClick={() => this.setState({ error: null })}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
