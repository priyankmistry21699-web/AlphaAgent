"""
AlphaAgent — Insider & Whale Agent

Tracks SEC Form 4 filings to see what company executives are doing,
and tracks institutional ownership via yfinance and SEC EDGAR.

Factors:
  1. Insider transaction sentiment (net buy/sell direction)
  2. Institutional ownership % and stability
  3. SEC EDGAR Form 4 recent filing count (signal of insider activity)
  4. 8-K material event count (news catalyst risk)
  5. Short-term insider cluster buying signal
"""

from typing import Any

from agents.base import BaseAgent
from agents.state import AgentResult, FactorScore
from quant_engine.insider import analyze_insider_data
from data.institutional import InstitutionalData


class InsiderAgent(BaseAgent):
    """
    Evaluates corporate insider transactions, institutional ownership,
    and EDGAR filing activity.
    """
    name = "insider"

    def __init__(self):
        super().__init__()
        self.edgar = InstitutionalData()

    def _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult:
        # ── 1. yfinance Insider & Institutional Data ──────────────────────
        insider_df = data.get_insider_transactions()
        major_holders_df = data.get_major_holders()

        result = analyze_insider_data(insider_df, major_holders_df)

        factor_scores = {
            "insider_sentiment": FactorScore(
                name="Insider Sentiment",
                value=result.net_insider_shares,
                score=result.insider_sentiment_score,
                interpretation=f"Net Shares: {result.net_insider_shares:,}",
            ),
            "institutional_backing": FactorScore(
                name="Institutional Ownership",
                value=result.institutional_ownership_pct,
                score=result.whale_sentiment_score,
                interpretation=f"Owned by Funds: {result.institutional_ownership_pct:.1f}%",
            ),
        }

        # ── 2. EDGAR Form 4 Filing Activity ──────────────────────────────
        warnings = list(result.warnings)
        edgar_reasoning = []

        try:
            edgar_snap = self.edgar.get_snapshot(ticker)

            form4_count = len(edgar_snap.recent_form4)
            eightk_count = edgar_snap.recent_8k_count
            edgar_inst_pct = edgar_snap.institutional_ownership_pct

            # Form 4 cluster signal: many filings in past 30d = insider activity
            if form4_count >= 5:
                edgar_reasoning.append(
                    f"EDGAR: {form4_count} Form 4 filings in 60d — high insider activity."
                )
                # We can't tell buy/sell without parsing XML, so treat as "active"
                form4_score = 65.0  # Slight positive (insiders usually buy when active)
            elif form4_count == 0:
                edgar_reasoning.append("EDGAR: No recent Form 4 activity.")
                form4_score = 50.0
            else:
                edgar_reasoning.append(f"EDGAR: {form4_count} Form 4 filings in 60d.")
                form4_score = 55.0

            factor_scores["edgar_form4_activity"] = FactorScore(
                name="EDGAR Form 4 Activity",
                value=float(form4_count),
                score=form4_score,
                interpretation=f"{form4_count} Form 4 filings | {eightk_count} 8-K events (30d)",
            )

            # 8-K count: many material events = heightened risk
            if eightk_count >= 5:
                warnings.append(f"High 8-K activity: {eightk_count} material events in 30d — elevated news risk.")
                factor_scores["material_events"] = FactorScore(
                    name="Material Events (8-K)",
                    value=float(eightk_count),
                    score=30.0,
                    interpretation=f"{eightk_count} 8-K filings in last 30 days",
                )
            elif eightk_count > 0:
                factor_scores["material_events"] = FactorScore(
                    name="Material Events (8-K)",
                    value=float(eightk_count),
                    score=50.0,
                    interpretation=f"{eightk_count} 8-K filings in last 30 days",
                )

            # EDGAR institutional ownership cross-check
            if edgar_inst_pct > 0:
                inst_score = min(80.0, edgar_inst_pct * 0.8)
                factor_scores["edgar_institutional"] = FactorScore(
                    name="Institutional Ownership (EDGAR)",
                    value=edgar_inst_pct,
                    score=inst_score,
                    interpretation=f"Institutions hold {edgar_inst_pct:.1f}% via EDGAR",
                )

        except Exception as e:
            edgar_reasoning.append(f"EDGAR data unavailable ({e}).")

        # ── New: Cluster Buys Detection ───────────────────────────────────
        try:
            import yfinance as yf
            insider_holders = yf.Ticker(ticker).insider_purchases
            if insider_holders is not None and not insider_holders.empty:
                buy_count = len(insider_holders[insider_holders.get("Shares", insider_holders.iloc[:,0]) > 0])
                cluster_score = min(85.0, 40.0 + buy_count * 12)
                factor_scores["cluster_buys"] = FactorScore(
                    name="Insider Cluster Buys",
                    value=float(buy_count),
                    score=cluster_score,
                    interpretation=f"{buy_count} insider purchase transaction(s) recently ({'strong cluster' if buy_count >= 3 else 'moderate' if buy_count >= 1 else 'none'})",
                )
        except Exception:
            pass

        # ── New: Institutional 13F Net Change ─────────────────────────────
        try:
            import yfinance as yf
            inst_holders = yf.Ticker(ticker).institutional_holders
            if inst_holders is not None and not inst_holders.empty and "% Out" in inst_holders.columns:
                inst_pct = float(inst_holders["% Out"].iloc[0]) * 100 if inst_holders["% Out"].iloc[0] < 1 else float(inst_holders["% Out"].iloc[0])
                inst_score = (80.0 if inst_pct > 70 else 60.0 if inst_pct > 40 else 40.0)
                factor_scores["institutional_ownership"] = FactorScore(
                    name="Institutional Ownership %",
                    value=round(inst_pct, 2),
                    score=inst_score,
                    interpretation=f"Institutions hold {inst_pct:.1f}% of float ({'high conviction' if inst_pct > 70 else 'moderate' if inst_pct > 40 else 'low institutional interest'})",
                )
        except Exception:
            pass

        # ── New: Short Squeeze Potential ──────────────────────────────────
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            short_pct = info.get("shortPercentOfFloat", 0) or 0
            short_ratio = info.get("shortRatio", 0) or 0  # days to cover
            if short_pct > 0:
                # High short + rising price = squeeze potential
                squeeze_score = min(85.0, 40.0 + float(short_pct) * 100 * 1.2)
                factor_scores["short_squeeze"] = FactorScore(
                    name="Short Squeeze Potential",
                    value=round(float(short_pct) * 100, 2),
                    score=squeeze_score,
                    interpretation=f"Short float: {float(short_pct)*100:.1f}% | Days to cover: {float(short_ratio):.1f}d ({'squeeze risk' if float(short_pct) > 0.15 else 'low short'})",
                )
        except Exception:
            pass

        # ── New: Shares Outstanding Trend (Float Reduction / Dilution) ───
        try:
            import yfinance as yf
            yf_tkr  = yf.Ticker(ticker)
            inf     = yf_tkr.info
            shares_now = inf.get("sharesOutstanding") or inf.get("impliedSharesOutstanding")
            # Try quarterly balance sheet for trend
            bs_q = yf_tkr.quarterly_balance_sheet
            if bs_q is not None and not bs_q.empty:
                share_rows = [r for r in bs_q.index
                              if 'ordinary' in str(r).lower() or ('share' in str(r).lower() and 'issued' not in str(r).lower() and 'preferred' not in str(r).lower())]
                if share_rows and len(bs_q.columns) >= 2:
                    s_now = float(bs_q.loc[share_rows[0]].iloc[0])
                    s_old = float(bs_q.loc[share_rows[0]].iloc[-1])
                    if s_old > 0:
                        share_chg = (s_now - s_old) / s_old * 100
                        float_score = (85.0 if share_chg < -2 else 65.0 if share_chg < 0 else 45.0 if share_chg < 2 else 20.0)
                        factor_scores["float_reduction"] = FactorScore(
                            name="Float Reduction (Buyback/Dilution)",
                            value=round(share_chg, 2),
                            score=float_score,
                            interpretation=f"Shares outstanding chg: {share_chg:+.1f}% ({'buyback — bullish supply squeeze' if share_chg < -1 else 'dilution — supply flood' if share_chg > 2 else 'stable float'})",
                        )
                        if share_chg > 3:
                            warnings.append(f"Share dilution detected: shares up {share_chg:.1f}% — secondary offering risk.")
        except Exception:
            pass

        # ── New: Kyle's Lambda (Price Impact / Market Microstructure) ────
        try:
            import yfinance as yf
            import numpy as _np
            hist_df = yf.Ticker(ticker).history(period="3mo", interval="1d")
            if not hist_df.empty and len(hist_df) >= 30:
                closes = hist_df["Close"].dropna().values
                volumes = hist_df["Volume"].dropna().values
                min_len = min(len(closes), len(volumes))
                closes  = closes[-min_len:]
                volumes = volumes[-min_len:]
                price_chg = _np.diff(closes)
                # signed volume: positive on up-days, negative on down-days
                signed_vol = volumes[1:] * _np.sign(price_chg)
                # OLS: price_change = lambda * signed_volume
                if len(signed_vol) >= 20 and _np.std(signed_vol) > 0:
                    cov  = _np.cov(price_chg, signed_vol)
                    kyle_lambda = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 0.0
                    # Normalize: low lambda = liquid (institutions can hide); high = thin/illiquid
                    avg_close = float(_np.mean(closes))
                    lambda_norm = abs(kyle_lambda) * 1e6 / avg_close if avg_close > 0 else 0
                    liq_score = max(10.0, min(90.0, 70.0 - lambda_norm * 10))
                    factor_scores["kyle_lambda"] = FactorScore(
                        name="Kyle's Lambda (Price Impact)",
                        value=round(kyle_lambda * 1e6, 4),
                        score=liq_score,
                        interpretation=(
                            f"λ={kyle_lambda*1e6:.4f} ×10⁻⁶ $/share "
                            f"({'thin — whale moves easily detected' if lambda_norm > 2 else 'liquid — institutions can hide' if lambda_norm < 0.5 else 'moderate depth'})"
                        ),
                    )
        except Exception:
            pass

        # ── New: Activist / 13D Proxy (High-Stakes Insider Buy via EDGAR)─
        try:
            edgar_snap2 = self.edgar.get_snapshot(ticker)
            form4_filings = edgar_snap2.recent_form4
            # Detect cluster: multiple insiders buying within 30d window
            if form4_filings:
                from collections import Counter
                import datetime
                today = __import__("datetime").date.today()
                recent_30d = [f for f in form4_filings
                              if hasattr(f, 'filed_at') and
                              (today - __import__("datetime").date.fromisoformat(str(f.filed_at)[:10])).days <= 30]
                if len(recent_30d) >= 3:
                    factor_scores["insider_cluster_30d"] = FactorScore(
                        name="Insider Cluster (30-Day)",
                        value=float(len(recent_30d)),
                        score=75.0,
                        interpretation=f"{len(recent_30d)} insider filings in 30 days — cluster buying signal",
                    )
        except Exception:
            pass

        # ── 3. Composite Probability ──────────────────────────────────────
        all_scores = [fs.score for fs in factor_scores.values()]
        composite_score = sum(all_scores) / len(all_scores) if all_scores else 50.0
        prob_up = self._map_score_to_probability(composite_score, min_val=0, max_val=100)

        # ── 4. Confidence ─────────────────────────────────────────────────
        confidence = 0.5
        if result.institutional_ownership_pct > 80:
            confidence += 0.2
        if abs(result.net_insider_shares) > 50000:
            confidence += 0.2
        confidence = min(1.0, max(0.0, confidence))

        # ── 5. Reasoning ──────────────────────────────────────────────────
        direction = "BULLISH" if prob_up > self.long_threshold else "BEARISH" if prob_up < self.short_threshold else "NEUTRAL"
        reasoning = (
            f"Insider & Whale outlook is {direction} ({prob_up * 100:.1f}% probability). "
            f"Institutions hold {result.institutional_ownership_pct:.1f}% of shares. "
            f"Recent insider net transaction volume: {result.net_insider_shares:,} shares. "
        )
        if edgar_reasoning:
            reasoning += " ".join(edgar_reasoning)

        return AgentResult(
            agent_name=self.name,
            probability_up=prob_up,
            confidence=confidence,
            reasoning=reasoning,
            factor_scores=factor_scores,
            warnings=warnings,
        )
