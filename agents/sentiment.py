"""
AlphaAgent — Sentiment & Behavioral Agent (RAG Pipeline)

Uses Retrieval-Augmented Generation (RAG) to pull the most relevant
news from the ChromaDB vector store, and asks an LLM (Gemini) to
analyze the sentiment, panic levels, and behavioral factors.

Additional quantitative sentiment signals:
  - Short interest ratio (days-to-cover)
  - Fear & Greed approximation (from AlternativeData)
  - Analyst recommendation consensus
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

import yfinance as yf

from agents.base import BaseAgent
from agents.state import AgentResult, Direction, FactorScore
from data.news import NewsDatabase
from data.alternative import AlternativeData

try:
    import google.genai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    name = "sentiment"

    def __init__(self):
        super().__init__()
        self.news_db = NewsDatabase()
        self.alt_data = AlternativeData()

    def _run_analysis(self, ticker: str, data: Dict[str, Any]) -> AgentResult:
        prob_up = 0.5
        confidence = 0.3
        reasoning = []
        factor_scores = {}
        warnings = []

        # ── 1. RAG News Sentiment ────────────────────────────────────────
        self.news_db.fetch_and_store_news(ticker)

        now_ts = int(datetime.now().timestamp())
        twelve_hours_ago = now_ts - (12 * 3600)

        query = f"lawsuits, earnings, bankruptcy, growth, new products, {ticker}"
        retrieved_articles = self.news_db.search_news(
            query,
            ticker=ticker,
            n_results=5,
            where={"timestamp": {"$gte": twelve_hours_ago}},
        )

        rag_prob = 0.5
        rag_reasoning = "No recent news found — sentiment neutral."
        rag_confidence = 0.2

        if retrieved_articles:
            context = "\n".join(retrieved_articles)
            api_key = os.getenv("GEMINI_API_KEY")

            if api_key and _GENAI_AVAILABLE:
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
You are a quantitative behavioral finance analyst.
Review the following retrieved news headlines for {ticker}:

{context}

Analyze the market sentiment and behavioral factors.
Respond in EXACTLY the following format:
SCORE: [a number from 0 to 100, where 0=extreme panic/bearish, 100=extreme euphoria/bullish, 50=neutral]
REASON: [A strict 2-sentence explanation focusing on behavioral economics and sentiment.]
"""
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(temperature=0.0),
                    )
                    text = response.text.strip()
                    score_line = [l for l in text.split("\n") if l.startswith("SCORE:")][0]
                    reason_line = [l for l in text.split("\n") if l.startswith("REASON:")][0]
                    rag_prob = float(score_line.split(":")[1].strip()) / 100.0
                    rag_reasoning = reason_line.split(":", 1)[1].strip()
                    rag_confidence = 0.8 if any(
                        w in context.lower()
                        for w in ("lawsuit", "earnings", "bankrupt", "fda", "sec")
                    ) else 0.55
                except Exception as e:
                    logger.warning(f"Gemini RAG failed: {e}")
                    rag_prob, rag_reasoning, rag_confidence = self._keyword_scan(context)
            else:
                rag_prob, rag_reasoning, rag_confidence = self._keyword_scan(context)

        reasoning.append(f"[News RAG] {rag_reasoning}")
        factor_scores["news_sentiment"] = FactorScore(
            name="News Sentiment (RAG)",
            value=rag_prob,
            score=rag_prob * 100.0,
            interpretation=f"RAG score: {rag_prob * 100:.0f}/100",
        )
        prob_up += (rag_prob - 0.5) * 0.4
        confidence = max(confidence, rag_confidence * 0.6)

        # ── 2. Short Interest ────────────────────────────────────────────
        try:
            info = data.get_info()
            short_pct = float(info.get("shortPercentOfFloat", 0) or 0) * 100
            short_ratio = float(info.get("shortRatio", 0) or 0)

            if short_pct > 20:
                prob_up -= 0.08
                confidence += 0.05
                reasoning.append(
                    f"HIGH short interest: {short_pct:.1f}% of float ({short_ratio:.1f}d to cover) — "
                    "bearish conviction, but squeeze risk elevated."
                )
                warnings.append(f"Short interest {short_pct:.1f}% — potential short squeeze risk.")
            elif short_pct > 10:
                prob_up -= 0.03
                reasoning.append(f"Elevated short interest: {short_pct:.1f}% of float.")
            elif short_pct < 3:
                prob_up += 0.03
                reasoning.append(f"Low short interest ({short_pct:.1f}%) — weak bearish conviction.")
            else:
                reasoning.append(f"Short interest neutral: {short_pct:.1f}% of float.")

            si_score = max(10.0, min(90.0, 80.0 - short_pct * 2.0))
            factor_scores["short_interest"] = FactorScore(
                name="Short Interest",
                value=short_pct,
                score=si_score,
                interpretation=f"Short float: {short_pct:.1f}% | Days to cover: {short_ratio:.1f}",
            )
        except Exception as e:
            reasoning.append(f"Short interest data unavailable ({e}).")

        # ── 3. Fear & Greed Approximation ────────────────────────────────
        try:
            alt_snap = self.alt_data.get_snapshot()
            fg = alt_snap.fear_greed_score
            fg_label = alt_snap.fear_greed_label

            if fg <= 25:
                prob_up += 0.05  # Contrarian — extreme fear = buying opportunity
                reasoning.append(f"Fear & Greed: {fg:.0f}/100 ({fg_label}) — contrarian BUY signal.")
            elif fg >= 75:
                prob_up -= 0.05
                reasoning.append(f"Fear & Greed: {fg:.0f}/100 ({fg_label}) — overbought euphoria.")
            else:
                reasoning.append(f"Fear & Greed: {fg:.0f}/100 ({fg_label}).")

            factor_scores["fear_greed"] = FactorScore(
                name="Fear & Greed Index",
                value=fg,
                score=float(fg),
                interpretation=f"{fg_label} ({fg:.0f}/100)",
            )
        except Exception as e:
            reasoning.append(f"Fear & Greed data unavailable ({e}).")

        # ── 4. Analyst Recommendations ───────────────────────────────────
        try:
            ticker_obj = yf.Ticker(ticker)
            recs = ticker_obj.recommendations
            if recs is not None and not recs.empty:
                # Get most recent 3 months of recommendations
                recent = recs.tail(30)
                strong_buy = int(recent.get("strongBuy", recent.get("Strong Buy", 0)).sum()
                                 if hasattr(recent.get("strongBuy", 0), "sum")
                                 else recent.get("strongBuy", recent.get("Strong Buy", 0)) if not hasattr(recent.get("strongBuy", 0), "sum") else 0)

                # Simplified: use the recommendation summary
                summary = ticker_obj.recommendations_summary
                if summary is not None and not summary.empty:
                    row = summary.iloc[0]
                    sb = float(row.get("strongBuy", 0))
                    b = float(row.get("buy", 0))
                    h = float(row.get("hold", 0))
                    sell = float(row.get("sell", 0))
                    ss = float(row.get("strongSell", 0))
                    total = sb + b + h + sell + ss
                    if total > 0:
                        bull_pct = (sb + b) / total
                        bear_pct = (sell + ss) / total
                        analyst_score = bull_pct * 100

                        if bull_pct > 0.7:
                            prob_up += 0.05
                            reasoning.append(f"Analysts bullish: {bull_pct*100:.0f}% buy/strong-buy.")
                        elif bear_pct > 0.3:
                            prob_up -= 0.04
                            reasoning.append(f"Analysts cautious: {bear_pct*100:.0f}% sell/strong-sell.")
                        else:
                            reasoning.append(f"Analysts mixed: {bull_pct*100:.0f}% bullish.")

                        factor_scores["analyst_consensus"] = FactorScore(
                            name="Analyst Consensus",
                            value=bull_pct,
                            score=analyst_score,
                            interpretation=(
                                f"Strong Buy: {sb:.0f} | Buy: {b:.0f} | "
                                f"Hold: {h:.0f} | Sell: {sell:.0f} | Strong Sell: {ss:.0f}"
                            ),
                        )
        except Exception as e:
            reasoning.append(f"Analyst data unavailable ({e}).")

        # ── 5. Market Breadth Signal ──────────────────────────────────────
        try:
            alt_snap = self.alt_data.get_snapshot()
            breadth = alt_snap.breadth_score
            if breadth > 65:
                prob_up += 0.03
                reasoning.append(f"Market breadth healthy ({breadth:.0f}/100) — broad rally.")
            elif breadth < 35:
                prob_up -= 0.03
                reasoning.append(f"Market breadth weak ({breadth:.0f}/100) — narrow/concentrated rally.")
            factor_scores["market_breadth"] = FactorScore(
                name="Market Breadth (RSP vs SPY)",
                value=breadth,
                score=breadth,
                interpretation=f"Breadth score: {breadth:.0f}/100",
            )
        except Exception:
            pass

        # ── New: Put/Call Skew (Options Sentiment) ────────────────────────
        try:
            import yfinance as yf
            tkr_obj = yf.Ticker(ticker)
            exps = tkr_obj.options
            if exps:
                chain = tkr_obj.option_chain(exps[0])
                total_put_vol  = float(chain.puts["volume"].fillna(0).sum())
                total_call_vol = float(chain.calls["volume"].fillna(0).sum())
                if total_call_vol > 0:
                    pc_skew = total_put_vol / total_call_vol
                    pcs_score = (70.0 if pc_skew < 0.7 else 50.0 if pc_skew < 1.2 else 25.0)
                    factor_scores["options_pc_skew"] = FactorScore(
                        name="Options Put/Call Skew",
                        value=round(pc_skew, 3),
                        score=pcs_score,
                        interpretation=f"P/C ratio: {pc_skew:.2f} ({'bullish skew' if pc_skew < 0.7 else 'bearish hedge demand' if pc_skew > 1.2 else 'balanced'})",
                    )
        except Exception:
            pass

        # ── New: Social Sentiment Proxy (search trend proxy via news volume)
        try:
            import yfinance as yf
            news = yf.Ticker(ticker).news or []
            recent_count = len([n for n in news if n.get("providerPublishTime", 0) > (__import__("time").time() - 86400 * 3)])
            # High news volume = social interest spike
            social_score = min(80.0, 40.0 + recent_count * 4)
            factor_scores["social_momentum"] = FactorScore(
                name="News / Social Momentum",
                value=float(recent_count),
                score=social_score,
                interpretation=f"{recent_count} articles in last 3 days — {'high' if recent_count > 8 else 'moderate' if recent_count > 3 else 'low'} buzz",
            )
        except Exception:
            pass

        # ── New: Earnings Revision Score ──────────────────────────────────
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            eps_fwd  = info.get("forwardEps")
            eps_curr = info.get("trailingEps")
            if eps_fwd and eps_curr and eps_curr != 0:
                revision_pct = (float(eps_fwd) - float(eps_curr)) / abs(float(eps_curr)) * 100
                rev_score = (85.0 if revision_pct > 15 else 60.0 if revision_pct > 5 else 50.0 if revision_pct > -5 else 25.0)
                factor_scores["earnings_revision"] = FactorScore(
                    name="EPS Forward Revision",
                    value=round(revision_pct, 2),
                    score=rev_score,
                    interpretation=f"Fwd EPS revision vs trailing: {revision_pct:+.1f}% ({'positive' if revision_pct > 5 else 'negative' if revision_pct < -5 else 'neutral'})",
                )
        except Exception:
            pass

        # ── New: Unusual Options Activity Score ───────────────────────────
        try:
            import yfinance as yf
            tkr_obj2 = yf.Ticker(ticker)
            exps2 = tkr_obj2.options
            if exps2:
                chain2 = tkr_obj2.option_chain(exps2[0])
                call_oi = float(chain2.calls["openInterest"].fillna(0).sum())
                put_oi  = float(chain2.puts["openInterest"].fillna(0).sum())
                call_vol = float(chain2.calls["volume"].fillna(0).sum())
                # Unusual activity: volume >> open interest (fresh bets)
                if call_oi > 0:
                    vol_oi_ratio = call_vol / call_oi
                    uoa_score = min(85.0, 40.0 + vol_oi_ratio * 30)
                    factor_scores["unusual_options"] = FactorScore(
                        name="Unusual Options Activity",
                        value=round(vol_oi_ratio, 3),
                        score=uoa_score,
                        interpretation=f"Call Vol/OI ratio: {vol_oi_ratio:.2f} ({'unusual buying' if vol_oi_ratio > 1.5 else 'normal activity'})",
                    )
        except Exception:
            pass

        # ── Clamp & Vote ──────────────────────────────────────────────────
        prob_up = max(0.01, min(0.99, prob_up))
        confidence = max(0.0, min(1.0, confidence))

        vote = (Direction.LONG if prob_up > 0.55
                else Direction.SHORT if prob_up < 0.45
                else Direction.HOLD)

        return AgentResult(
            agent_name=self.name,
            vote=vote,
            probability_up=prob_up,
            confidence=confidence,
            reasoning=" ".join(reasoning),
            factor_scores=factor_scores,
            warnings=warnings,
        )

    def _keyword_scan(self, context: str) -> tuple:
        """Fallback keyword scanner if Gemini fails."""
        bull_words = ["surge", "jump", "up", "beat", "growth", "upgrade", "profit", "record", "strong"]
        bear_words = ["drop", "fall", "down", "miss", "decline", "downgrade", "loss", "warning", "cut"]
        ctx_lower = context.lower()
        bull_score = sum(1 for w in bull_words if w in ctx_lower)
        bear_score = sum(1 for w in bear_words if w in ctx_lower)
        if bull_score > bear_score:
            return 0.65, "Sentiment BULLISH based on positive keywords in news.", 0.4
        if bear_score > bull_score:
            return 0.35, "Sentiment BEARISH based on negative keywords in news.", 0.4
        return 0.5, "Sentiment NEUTRAL — no strong keyword signal.", 0.3
