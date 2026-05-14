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
        direction = "BULLISH" if prob_up > 0.55 else "BEARISH" if prob_up < 0.45 else "NEUTRAL"
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
