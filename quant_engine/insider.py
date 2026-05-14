"""
AlphaAgent — Insider & Institutional Math Engine

Calculates net insider buying/selling pressure and institutional 
ownership concentration.
"""

import pandas as pd
from dataclasses import dataclass

@dataclass
class InsiderResult:
    net_insider_shares: int
    insider_sentiment_score: float  # 0-100
    institutional_ownership_pct: float # 0-100
    whale_sentiment_score: float    # 0-100
    warnings: list[str]

def analyze_insider_data(insider_df: pd.DataFrame, major_holders_df: pd.DataFrame) -> InsiderResult:
    """
    Analyzes SEC Form 4 and 13F proxy data from yfinance.
    """
    warnings = []
    
    # ── 1. Insider Transactions (Form 4) ──
    net_shares = 0
    insider_score = 50.0
    
    if not insider_df.empty and 'Shares' in insider_df.columns:
        # yfinance insider_df format can be inconsistent, but typically positive shares = buy, negative = sell
        # We try to sum the recent transaction shares
        try:
            # Drop NaN shares
            valid_shares = insider_df['Shares'].dropna()
            
            # Sum up net shares
            if len(valid_shares) > 0:
                net_shares = int(valid_shares.sum())
                
                # Simple scoring: 
                # Lots of buying is very bullish (> +100k shares)
                # Lots of selling is slightly bearish (insiders sell for many reasons, but buy for only one)
                if net_shares > 100000:
                    insider_score = 90.0
                elif net_shares > 10000:
                    insider_score = 75.0
                elif net_shares < -500000:
                    insider_score = 25.0
                    warnings.append(f"Heavy insider selling detected: {net_shares:,} shares.")
                elif net_shares < 0:
                    insider_score = 40.0
        except Exception as e:
            warnings.append("Could not parse insider shares correctly.")

    # ── 2. Institutional Ownership ──
    inst_own_pct = 50.0
    whale_score = 50.0
    
    if not major_holders_df.empty:
        try:
            # yfinance major_holders column layout varies by version.
            # Try the label column (index 1 or named 'Breakdown') then any string column.
            label_col = None
            for col in major_holders_df.columns:
                try:
                    if major_holders_df[col].astype(str).str.contains('Institution', case=False, na=False).any():
                        label_col = col
                        break
                except Exception:
                    continue

            if label_col is not None:
                inst_row = major_holders_df[
                    major_holders_df[label_col].astype(str).str.contains('Institution', case=False, na=False)
                ]
                if not inst_row.empty:
                    # Value column is the one that isn't the label column
                    val_col = [c for c in major_holders_df.columns if c != label_col][0]
                    val = inst_row.iloc[0][val_col]
                    if isinstance(val, str) and '%' in val:
                        inst_own_pct = float(val.replace('%', ''))
                    else:
                        inst_own_pct = float(val) * 100

                    if inst_own_pct > 80.0:
                        whale_score = 80.0
                    elif inst_own_pct < 20.0:
                        whale_score = 30.0
                        warnings.append(f"Very low institutional ownership ({inst_own_pct:.1f}%).")
        except Exception:
            warnings.append("Could not parse institutional ownership.")
            
    return InsiderResult(
        net_insider_shares=net_shares,
        insider_sentiment_score=insider_score,
        institutional_ownership_pct=inst_own_pct,
        whale_sentiment_score=whale_score,
        warnings=warnings
    )
