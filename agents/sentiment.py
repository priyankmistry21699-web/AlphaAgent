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
        two_days_ago = now_ts - (48 * 3600)

        query = f"lawsuits, earnings, bankruptcy, growth, new products, {ticker}"
        retrieved_articles = self.news_db.search_news(
            query,
            ticker=ticker,
            n_results=8,
            where={"timestamp": {"$gte": two_days_ago}},
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
            _ss_base = settings.get("sentiment.social_score_base", 40.0)
            _ss_mult = settings.get("sentiment.social_score_mult", 4.0)
            social_score = min(80.0, _ss_base + recent_count * _ss_mult)
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

        # ── New: Transfer Entropy Proxy (News → Price Causality) ─────────
        # Approximation: Granger-style test — does lagged news volume predict returns?
        try:
            import numpy as _np
            tkr_obj_te = yf.Ticker(ticker)
            news_items  = tkr_obj_te.news or []
            hist_te = tkr_obj_te.history(period="3mo", interval="1d")
            if not hist_te.empty and len(news_items) >= 5:
                import pandas as pd
                # Build daily news count series
                import time as _time
                today_ts = int(_time.time())
                daily_counts = {}
                for n in news_items:
                    ts = n.get("providerPublishTime", 0)
                    day_key = pd.Timestamp(ts, unit="s").normalize()
                    daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

                news_ser = pd.Series(daily_counts).reindex(hist_te.index, fill_value=0)
                rets_ser = hist_te["Close"].pct_change().dropna()
                news_ser = news_ser.reindex(rets_ser.index, fill_value=0)

                if len(rets_ser) >= 20:
                    # Lag-1 cross-correlation: does yesterday's news predict today's return?
                    lagged_news = news_ser.shift(1).dropna()
                    aligned_ret = rets_ser.reindex(lagged_news.index).dropna()
                    lagged_news = lagged_news.reindex(aligned_ret.index)

                    if len(aligned_ret) >= 15 and lagged_news.std() > 0:
                        corr = float(aligned_ret.corr(lagged_news))
                        te_score = max(10.0, min(90.0, 50.0 + corr * 40.0))
                        factor_scores["transfer_entropy_proxy"] = FactorScore(
                            name="Transfer Entropy (News→Price)",
                            value=round(corr, 3),
                            score=te_score,
                            interpretation=(
                                f"Lag-1 news-return corr: {corr:+.3f} "
                                f"({'news reliably leads price' if abs(corr) > 0.3 else 'weak causal link' if abs(corr) > 0.1 else 'no detectable causality'})"
                            ),
                        )
                        if corr > 0.3:
                            reasoning.append(f"Transfer entropy: news flow significantly leads price ({corr:+.2f} corr) — sentiment is causal.")
        except Exception:
            pass

        # ── New: Shannon Entropy of Factor Scores (Signal Clarity) ────────
        try:
            import numpy as _np
            if len(factor_scores) >= 3:
                scores = _np.array([fs.score for fs in factor_scores.values()]) / 100.0
                scores = _np.clip(scores, 1e-9, 1.0 - 1e-9)
                # Normalize to probability distribution
                scores_n = scores / scores.sum()
                shannon_h = float(-_np.sum(scores_n * _np.log(scores_n)))
                max_entropy = _np.log(len(scores_n))
                norm_entropy = shannon_h / max_entropy if max_entropy > 0 else 0.5
                # Low entropy = agents agree (high clarity); high entropy = dispersed signals
                entropy_score = max(10.0, min(90.0, (1.0 - norm_entropy) * 80.0 + 10.0))
                factor_scores["signal_entropy"] = FactorScore(
                    name="Signal Entropy (Clarity Index)",
                    value=round(norm_entropy, 3),
                    score=entropy_score,
                    interpretation=(
                        f"Normalized Shannon entropy: {norm_entropy:.3f} "
                        f"({'high clarity — signals aligned' if norm_entropy < 0.7 else 'mixed signals' if norm_entropy < 0.9 else 'maximum disagreement'})"
                    ),
                )
        except Exception:
            pass

        # ── New: Price Target vs Current Price ────────────────────────────
        try:
            info_obj = yf.Ticker(ticker).info
            target_mean = info_obj.get("targetMeanPrice")
            current_p   = info_obj.get("currentPrice") or info_obj.get("regularMarketPrice")
            if target_mean and current_p and float(current_p) > 0:
                upside_pct = (float(target_mean) - float(current_p)) / float(current_p) * 100
                pt_score = (85.0 if upside_pct > 20 else 65.0 if upside_pct > 5 else 50.0 if upside_pct > -5 else 25.0)
                if upside_pct > 10:
                    prob_up += 0.04
                    reasoning.append(f"Analyst price target implies {upside_pct:+.1f}% upside (target ${target_mean:.2f}).")
                elif upside_pct < -10:
                    prob_up -= 0.04
                    reasoning.append(f"Analyst price target implies {upside_pct:.1f}% downside from current.")
                factor_scores["price_target_upside"] = FactorScore(
                    name="Analyst Price Target Upside",
                    value=round(upside_pct, 2),
                    score=pt_score,
                    interpretation=f"Mean target ${target_mean:.2f} vs current ${current_p:.2f} → {upside_pct:+.1f}% upside",
                )
        except Exception:
            pass

        # ── New: AAII Sentiment Survey (Contrarian) ───────────────────────
        try:
            from data.macro import MacroData
            macro_d = MacroData()
            aaii_bull = macro_d.get_series("AAIIBULL", years_back=1)
            aaii_bear = macro_d.get_series("AAIIBEAR", years_back=1)
            if (aaii_bull is not None and len(aaii_bull) >= 2 and
                    aaii_bear is not None and len(aaii_bear) >= 2):
                bull_pct = float(aaii_bull.iloc[-1].iloc[0])
                bear_pct = float(aaii_bear.iloc[-1].iloc[0])
                bull_bear_spread = bull_pct - bear_pct
                # Contrarian: extreme bearishness = buy signal
                if bull_bear_spread < -20:
                    aaii_score = 75.0
                    prob_up += 0.04
                    reasoning.append(f"AAII extreme bearishness (bull-bear: {bull_bear_spread:+.0f}%) — contrarian buy signal.")
                elif bull_bear_spread > 30:
                    aaii_score = 25.0
                    prob_up -= 0.03
                    reasoning.append(f"AAII extreme bullishness (bull-bear: {bull_bear_spread:+.0f}%) — contrarian caution.")
                else:
                    aaii_score = 50.0
                factor_scores["aaii_sentiment"] = FactorScore(
                    name="AAII Sentiment Survey",
                    value=round(bull_bear_spread, 1),
                    score=aaii_score,
                    interpretation=f"Bull: {bull_pct:.0f}% | Bear: {bear_pct:.0f}% | Spread: {bull_bear_spread:+.0f}% (contrarian indicator)",
                )
        except Exception:
            pass

        # ── New: Margin Debt Proxy (FRED) ─────────────────────────────────
        try:
            from data.macro import MacroData
            macro_d2 = MacroData()
            # FINNWILSHIRE = Wilshire 5000 as margin debt proxy alternative
            # Use RBUSBIS: reserve balances or check BOGZ1FL664220005Q margin accounts
            # Use consumer credit as proxy for leveraged retail demand
            cc = macro_d2.get_series("TOTALSL", years_back=1)   # Total Consumer Credit
            if cc is not None and len(cc) >= 3:
                cc_now  = float(cc.iloc[-1].iloc[0])
                cc_prev = float(cc.iloc[-3].iloc[0])
                cc_chg  = (cc_now / cc_prev - 1) * 100
                # Rising consumer credit = leveraged demand (short-term bullish, long-term risk)
                md_score = (60.0 if cc_chg > 2 else 50.0 if cc_chg > 0 else 40.0)
                factor_scores["consumer_credit_proxy"] = FactorScore(
                    name="Consumer Credit (Margin Proxy)",
                    value=round(cc_chg, 2),
                    score=md_score,
                    interpretation=f"Consumer credit 3M change: {cc_chg:+.2f}% ({'leverage expanding' if cc_chg > 2 else 'deleveraging' if cc_chg < -1 else 'stable'})",
                )
        except Exception:
            pass

        # ── New: Short Interest Change (MoM) ─────────────────────────────
        try:
            _si_info = yf.Ticker(ticker).info
            _shares_short       = _si_info.get("sharesShort")
            _shares_short_prior = _si_info.get("sharesShortPriorMonth")
            if _shares_short and _shares_short_prior and float(_shares_short_prior) > 0:
                _si_chg = (float(_shares_short) - float(_shares_short_prior)) / float(_shares_short_prior) * 100
                _sic_score = (75.0 if _si_chg < -10 else 55.0 if _si_chg < 0 else 40.0 if _si_chg < 10 else 20.0)
                factor_scores["short_interest_change"] = FactorScore(
                    name="Short Interest Change (MoM)",
                    value=round(_si_chg, 2),
                    score=_sic_score,
                    interpretation=f"Short interest MoM: {_si_chg:+.1f}% ({'shorts covering — bullish' if _si_chg < -10 else 'short interest rising — bearish pressure' if _si_chg > 10 else 'stable'})",
                )
                if _si_chg > 20:
                    reasoning.append(f"Short interest surging {_si_chg:+.0f}% MoM — bearish conviction building.")
                elif _si_chg < -15:
                    reasoning.append(f"Shorts covering hard ({_si_chg:.0f}% MoM) — potential short squeeze catalyst.")
        except Exception:
            pass

        # ── New: Analyst Revision Direction ──────────────────────────────
        try:
            _tkr_ard   = yf.Ticker(ticker)
            _summ_ard  = _tkr_ard.recommendations_summary
            if _summ_ard is not None and not _summ_ard.empty and len(_summ_ard) >= 2:
                def _bull_frac(row):
                    _t = (float(row.get("strongBuy", 0)) + float(row.get("buy", 0))
                          + float(row.get("hold", 0)) + float(row.get("sell", 0))
                          + float(row.get("strongSell", 0)))
                    return (float(row.get("strongBuy", 0)) + float(row.get("buy", 0))) / _t if _t > 0 else 0.5
                _bp_now  = _bull_frac(_summ_ard.iloc[0])
                _bp_prev = _bull_frac(_summ_ard.iloc[1])
                _ard_delta = _bp_now - _bp_prev
                _ard_score = (75.0 if _ard_delta > 0.05 else 55.0 if _ard_delta > 0 else 40.0 if _ard_delta > -0.05 else 20.0)
                factor_scores["analyst_revision_direction"] = FactorScore(
                    name="Analyst Revision Direction",
                    value=round(_ard_delta * 100, 1),
                    score=_ard_score,
                    interpretation=f"Analyst bullish% change: {_ard_delta*100:+.1f}pp ({'upgrade cycle' if _ard_delta > 0.05 else 'downgrade cycle' if _ard_delta < -0.05 else 'stable consensus'})",
                )
                if _ard_delta < -0.10:
                    reasoning.append(f"Analyst consensus deteriorating ({_ard_delta*100:.0f}pp) — downgrade cycle.")
                elif _ard_delta > 0.10:
                    reasoning.append(f"Analyst upgrades accelerating ({_ard_delta*100:+.0f}pp) — positive revision cycle.")
        except Exception:
            pass

        # ── New: Sentiment Momentum (3-Day News Velocity) ─────────────────
        try:
            import time as _t_sm
            _news_sm  = yf.Ticker(ticker).news or []
            _now_sm   = _t_sm.time()
            _count_r  = sum(1 for n in _news_sm if n.get("providerPublishTime", 0) > _now_sm - 3 * 86400)
            _count_p  = sum(1 for n in _news_sm if _now_sm - 6 * 86400 < n.get("providerPublishTime", 0) <= _now_sm - 3 * 86400)
            _sm_delta = _count_r - _count_p
            _sm_score = (70.0 if _sm_delta > 2 else 55.0 if _sm_delta > 0 else 45.0 if _sm_delta == 0 else 30.0)
            factor_scores["sentiment_momentum"] = FactorScore(
                name="Sentiment Momentum (3-Day Trend)",
                value=float(_sm_delta),
                score=_sm_score,
                interpretation=f"News velocity: {_count_r} articles (0–3d) vs {_count_p} (3–6d) → Δ{_sm_delta:+d} ({'accelerating' if _sm_delta > 2 else 'decelerating' if _sm_delta < -2 else 'stable'})",
            )
            if _sm_delta < -3:
                reasoning.append(f"News flow decelerating ({_sm_delta}) — sentiment fading.")
            elif _sm_delta > 3:
                reasoning.append(f"News velocity surging ({_sm_delta:+d}) — catalysts emerging.")
        except Exception:
            pass

        # ── New: Source Credibility Weight ────────────────────────────────
        try:
            _CREDIBILITY_TIERS = {
                "wsj": 1.0, "bloomberg": 1.0, "reuters": 1.0, "ft.com": 1.0,
                "barrons": 0.95, "cnbc": 0.85, "marketwatch": 0.80,
                "seekingalpha": 0.70, "motleyfool": 0.65, "forbes": 0.65,
                "businessinsider": 0.60, "benzinga": 0.55, "thestreet": 0.55,
                "zacks": 0.60, "investopedia": 0.50,
            }
            _news_sc = yf.Ticker(ticker).news or []
            if _news_sc:
                _cred_scores = []
                for _art in _news_sc[:20]:
                    _pub = (_art.get("publisher", "") or "").lower()
                    _cred = 0.5
                    for _domain, _w in _CREDIBILITY_TIERS.items():
                        if _domain in _pub:
                            _cred = _w
                            break
                    _cred_scores.append(_cred)
                _avg_cred = sum(_cred_scores) / len(_cred_scores)
                _sc_score = max(20.0, min(80.0, _avg_cred * 100))
                factor_scores["source_credibility"] = FactorScore(
                    name="Source Credibility Weight",
                    value=round(_avg_cred, 3),
                    score=_sc_score,
                    interpretation=f"Avg source credibility: {_avg_cred:.2f}/1.0 ({len(_news_sc[:20])} articles) — {'premium financial media' if _avg_cred > 0.75 else 'mixed credibility' if _avg_cred > 0.5 else 'low-tier sources'}",
                )
        except Exception:
            pass

        # ── New: Headline vs Body Alignment (LLM Dual-Pass) ──────────────
        try:
            if _GENAI_AVAILABLE and os.getenv("GEMINI_API_KEY") and retrieved_articles:
                import google.genai as _genai_hb
                _hb_client = _genai_hb.Client(api_key=os.getenv("GEMINI_API_KEY"))
                _hb_context = "\n".join(retrieved_articles[:3])
                _hb_prompt = (
                    f"You analyze financial news for {ticker}. Given this content:\n\n"
                    f"{_hb_context}\n\n"
                    "Rate on 0-100 where 0=bearish, 100=bullish. Reply ONLY in this format:\n"
                    "HEADLINE_SCORE: [number]\n"
                    "BODY_SCORE: [number]\n"
                    "ALIGNMENT: [ALIGNED or DIVERGED]\n"
                )
                _hb_response = _hb_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=_hb_prompt,
                    config=_genai_hb.types.GenerateContentConfig(temperature=0.0),
                )
                _hb_lines = {}
                for _hb_line in _hb_response.text.split("\n"):
                    if ":" in _hb_line:
                        _hk, _hv = _hb_line.split(":", 1)
                        _hb_lines[_hk.strip()] = _hv.strip()
                _hl_score   = float(_hb_lines.get("HEADLINE_SCORE", "50"))
                _body_score = float(_hb_lines.get("BODY_SCORE", "50"))
                _alignment  = _hb_lines.get("ALIGNMENT", "ALIGNED").upper()
                _hba_score  = (
                    70.0 if _alignment == "ALIGNED" and _body_score > 50
                    else 50.0 if _alignment == "ALIGNED"
                    else 30.0 if _body_score < _hl_score - 20
                    else 45.0
                )
                factor_scores["headline_body_alignment"] = FactorScore(
                    name="Headline/Body Alignment",
                    value=round(_hl_score - _body_score, 1),
                    score=_hba_score,
                    interpretation=f"Headline {_hl_score:.0f}/100 vs Body {_body_score:.0f}/100 → {_alignment} — {'consistent coverage' if _alignment == 'ALIGNED' else 'misleading headline masks bearish content' if _body_score < _hl_score - 20 else 'minor divergence'}",
                )
                if _alignment == "DIVERGED" and _body_score < _hl_score - 20:
                    reasoning.append(f"Headline/body divergence: bullish headline masks bearish body ({_hl_score:.0f} vs {_body_score:.0f}) — credibility concern.")
        except Exception:
            pass

        # ── New: FinBERT-Style NLP (Gemini Structured Financial Analysis) ─────
        try:
            if _GENAI_AVAILABLE and os.getenv("GEMINI_API_KEY") and retrieved_articles:
                import google.genai as _genai_fb
                _fb_client = _genai_fb.Client(api_key=os.getenv("GEMINI_API_KEY"))
                _fb_context = "\n\n".join(retrieved_articles[:5])
                _fb_prompt = (
                    f"You are a financial NLP model (like FinBERT) analyzing news about {ticker}. "
                    f"Articles:\n\n{_fb_context}\n\n"
                    "Respond ONLY in this exact format:\n"
                    "OVERALL_SENTIMENT: [BULLISH/BEARISH/NEUTRAL]\n"
                    "REVENUE_GUIDANCE: [POSITIVE/NEGATIVE/NEUTRAL/NOT_MENTIONED]\n"
                    "MANAGEMENT_TONE: [CONFIDENT/CAUTIOUS/ALARMED/NEUTRAL]\n"
                    "RISK_FACTORS: [HIGH/MEDIUM/LOW]\n"
                    "CATALYST_PRESENT: [YES/NO]\n"
                    "CONFIDENCE_SCORE: [0-100]\n"
                )
                _fb_resp = _fb_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=_fb_prompt,
                    config=_genai_fb.types.GenerateContentConfig(temperature=0.0),
                )
                _fb_parsed = {}
                for _line in _fb_resp.text.split("\n"):
                    if ":" in _line:
                        _fk, _fv = _line.split(":", 1)
                        _fb_parsed[_fk.strip()] = _fv.strip()
                _fb_sent  = _fb_parsed.get("OVERALL_SENTIMENT", "NEUTRAL")
                _fb_tone  = _fb_parsed.get("MANAGEMENT_TONE", "NEUTRAL")
                _fb_risk  = _fb_parsed.get("RISK_FACTORS", "MEDIUM")
                _fb_cat   = _fb_parsed.get("CATALYST_PRESENT", "NO")
                _fb_guid  = _fb_parsed.get("REVENUE_GUIDANCE", "NEUTRAL")
                _fb_conf  = min(100.0, max(0.0, float(_fb_parsed.get("CONFIDENCE_SCORE", "50"))))
                _fb_base  = (70.0 if _fb_sent == "BULLISH" else 30.0 if _fb_sent == "BEARISH" else 50.0)
                _fb_adj   = (5.0 if _fb_tone == "CONFIDENT" else -5.0 if _fb_tone in ("CAUTIOUS", "ALARMED") else 0.0)
                _fb_adj  += (-5.0 if _fb_risk == "HIGH" else 3.0 if _fb_risk == "LOW" else 0.0)
                _fb_adj  += (5.0 if _fb_cat == "YES" else 0.0)
                _fb_score = max(10.0, min(90.0, _fb_base + _fb_adj))
                factor_scores["finbert_nlp"] = FactorScore(
                    name="FinBERT-Style NLP (Gemini)",
                    value=round(_fb_conf / 100.0, 3),
                    score=_fb_score,
                    interpretation=f"Sentiment: {_fb_sent} | Guidance: {_fb_guid} | Tone: {_fb_tone} | Risk: {_fb_risk} | Catalyst: {_fb_cat} (conf {_fb_conf:.0f}/100)",
                )
                if _fb_sent == "BULLISH" and _fb_cat == "YES" and _fb_tone == "CONFIDENT":
                    prob_up += 0.03
                    reasoning.append(f"FinBERT NLP: bullish catalyst + confident management tone.")
                elif _fb_sent == "BEARISH" and _fb_risk == "HIGH":
                    prob_up -= 0.03
                    reasoning.append(f"FinBERT NLP: bearish + high risk factors detected.")
        except Exception:
            pass

        # ── New: Reddit Sentiment (Public JSON — WSB + Stocks + Investing) ───
        try:
            import urllib.request as _req_r
            import json as _json_r
            _bull_r = ["buy", "calls", "bullish", "moon", "long", "squeeze", "undervalued", "catalyst", "beat", "breakout"]
            _bear_r = ["puts", "bearish", "short", "overvalued", "crash", "dump", "sell", "avoid", "miss", "downgrade"]
            _r_bull, _r_bear, _r_total = 0, 0, 0
            _limit_r = settings.get("backtest.reddit_post_limit", 15)
            for _sub in ["wallstreetbets", "stocks", "investing"]:
                try:
                    _url_r = f"https://www.reddit.com/r/{_sub}/search.json?q={ticker}&sort=new&limit={_limit_r}&restrict_sr=1"
                    _req_obj = _req_r.Request(_url_r, headers={"User-Agent": "AlphaAgent/1.0"})
                    with _req_r.urlopen(_req_obj, timeout=5) as _resp_r:
                        _data_r = _json_r.loads(_resp_r.read())
                    for _post in _data_r.get("data", {}).get("children", []):
                        _txt = (_post["data"].get("title", "") + " " + _post["data"].get("selftext", "")).lower()
                        if ticker.lower() in _txt:
                            _r_total += 1
                            _bc = sum(1 for w in _bull_r if w in _txt)
                            _dc = sum(1 for w in _bear_r if w in _txt)
                            if _bc > _dc:
                                _r_bull += 1
                            elif _dc > _bc:
                                _r_bear += 1
                except Exception:
                    pass
            if _r_total > 0:
                _r_bull_pct = _r_bull / _r_total * 100
                _reddit_score = (72.0 if _r_bull_pct > 60 else 55.0 if _r_bull_pct > 45 else 35.0 if _r_bull_pct < 30 else 45.0)
                factor_scores["reddit_sentiment"] = FactorScore(
                    name="Reddit Sentiment (WSB+Stocks+Investing)",
                    value=round(_r_bull_pct, 1),
                    score=_reddit_score,
                    interpretation=f"{_r_total} posts | {_r_bull} bull / {_r_bear} bear ({_r_bull_pct:.0f}% bullish) — r/wallstreetbets, r/stocks, r/investing",
                )
                if _r_bull_pct > 70 and _r_total >= 5:
                    reasoning.append(f"Reddit strongly bullish ({_r_bull_pct:.0f}% of {_r_total} posts).")
                elif _r_bull_pct < 25 and _r_total >= 5:
                    reasoning.append(f"Reddit bearish ({_r_bull_pct:.0f}% bullish in {_r_total} posts).")
        except Exception:
            pass

        # ── New: News Decay Model (Exponential Age-Weighted Sentiment) ───────
        try:
            import time as _t_nd
            import math as _math_nd
            # Dynamic half-life: shorter near earnings (news moves faster pre-event)
            _nd_hl_base = settings.get("backtest.news_decay_halflife_days", 4.6)
            # VIX-adaptive halflife: news decays faster in high-vol regimes
            try:
                import yfinance as _yf_hl
                _vix_hl = _yf_hl.download("^VIX", period="5d", interval="1d",
                                          auto_adjust=True, progress=False)
                if not _vix_hl.empty:
                    _vix_now_hl = float(_vix_hl["Close"].squeeze().dropna().iloc[-1])
                    if _vix_now_hl > 30:
                        _nd_hl_base = min(_nd_hl_base, 2.0)   # fast decay in crisis
                    elif _vix_now_hl > 20:
                        _nd_hl_base = min(_nd_hl_base, 3.5)   # moderate decay elevated vol
                    elif _vix_now_hl < 14:
                        _nd_hl_base = max(_nd_hl_base, 6.0)   # slow decay in calm market
            except Exception:
                pass
            try:
                import pandas as _pd_nd_hl
                _cal_nd = yf.Ticker(ticker).calendar
                if _cal_nd is not None and not _cal_nd.empty:
                    _ec = [c for c in _cal_nd.columns if "Earnings" in str(c)]
                    if _ec:
                        _ed = _pd_nd_hl.to_datetime(_cal_nd[_ec[0]].iloc[0])
                        _de = int((_ed - _pd_nd_hl.Timestamp.now(tz=_ed.tzinfo)).days)
                        if 0 <= _de <= 7:
                            _nd_hl_base = 1.5   # very fast decay around earnings
                        elif 7 < _de <= 21:
                            _nd_hl_base = 2.8   # faster decay pre-earnings
            except Exception:
                pass
            _nd_hl    = _nd_hl_base
            _nd_decay = _math_nd.log(2) / _nd_hl
            _nd_news  = yf.Ticker(ticker).news or []
            if _nd_news:
                _now_nd = _t_nd.time()
                _bull_nd = ["beat", "surge", "up", "growth", "record", "strong", "buy", "upgrade", "profit", "expand"]
                _bear_nd = ["miss", "drop", "down", "loss", "warning", "sell", "downgrade", "cut", "decline", "weak"]
                _nd_ws, _nd_ss = [], []
                for _art in _nd_news[:20]:
                    _age_d  = max(0.0, (_now_nd - _art.get("providerPublishTime", _now_nd)) / 86400.0)
                    _weight = _math_nd.exp(-_nd_decay * _age_d)
                    _title  = (_art.get("title", "") or "").lower()
                    _bc = sum(1 for w in _bull_nd if w in _title)
                    _dc = sum(1 for w in _bear_nd if w in _title)
                    _polarity = 0.65 if _bc > _dc else 0.35 if _dc > _bc else 0.5
                    _nd_ss.append(_polarity)
                    _nd_ws.append(_weight)
                _nd_sum = sum(_nd_ws)
                if _nd_sum > 0:
                    _nd_avg  = sum(s * w for s, w in zip(_nd_ss, _nd_ws)) / _nd_sum
                    _nd_score = max(10.0, min(90.0, _nd_avg * 100))
                    factor_scores["news_decay"] = FactorScore(
                        name="News Decay Model (Weighted)",
                        value=round(_nd_avg, 3),
                        score=_nd_score,
                        interpretation=f"Decay-weighted sentiment: {_nd_avg:.2f}/1.0 | {len(_nd_news[:20])} articles | half-life: {_nd_hl:.1f}d | newest weight: {_nd_ws[0]:.2f}, oldest: {_nd_ws[-1]:.2f}",
                    )
        except Exception:
            pass

        # ── New: Unusual Options Activity (Vol/OI > 2x) ──────────────────────
        try:
            _tkr_uoa = yf.Ticker(ticker)
            _vc_exps = _tkr_uoa.options
            if _vc_exps:
                _chain_uoa = _tkr_uoa.option_chain(_vc_exps[0])
                _unusual_calls, _unusual_puts = 0, 0
                _max_c_ratio, _max_p_ratio = 0.0, 0.0
                for _df_uoa, _is_call in [(_chain_uoa.calls, True), (_chain_uoa.puts, False)]:
                    if _df_uoa is not None and not _df_uoa.empty:
                        for _, _row_uoa in _df_uoa.iterrows():
                            _vol_uoa = float(_row_uoa.get("volume", 0) or 0)
                            _oi_uoa  = float(_row_uoa.get("openInterest", 0) or 0)
                            if _oi_uoa > 100 and _vol_uoa > 0:
                                _ratio_uoa = _vol_uoa / _oi_uoa
                                if _is_call:
                                    if _ratio_uoa > 2.0:
                                        _unusual_calls += 1
                                    _max_c_ratio = max(_max_c_ratio, _ratio_uoa)
                                else:
                                    if _ratio_uoa > 2.0:
                                        _unusual_puts += 1
                                    _max_p_ratio = max(_max_p_ratio, _ratio_uoa)
                _net_uoa = _unusual_calls - _unusual_puts
                _uoa_score = (78.0 if _net_uoa >= 3 else 62.0 if _net_uoa >= 1
                              else 38.0 if _net_uoa <= -1 else 50.0)
                _uoa_label = ("UNUSUAL CALL SWEEP — institutional demand" if _net_uoa >= 3
                              else "mild call bias" if _net_uoa >= 1
                              else "UNUSUAL PUT SWEEP — hedging/bearish bet" if _net_uoa <= -3
                              else "mild put bias" if _net_uoa <= -1
                              else "normal options activity")
                factor_scores["unusual_options"] = FactorScore(
                    name="Unusual Options Activity",
                    value=float(_net_uoa),
                    score=_uoa_score,
                    interpretation=f"{_unusual_calls} unusual call / {_unusual_puts} unusual put strikes (vol/OI>2x) — {_uoa_label}",
                )
                if _unusual_calls >= 3:
                    reasoning.append(f"Unusual call activity ({_unusual_calls} strikes, max vol/OI {_max_c_ratio:.1f}x) — potential institutional accumulation.")
                    prob_up += 0.04
                elif _unusual_puts >= 3:
                    reasoning.append(f"Unusual put sweep ({_unusual_puts} strikes, max vol/OI {_max_p_ratio:.1f}x) — hedging or directional bearish bet.")
                    prob_up -= 0.04
        except Exception:
            pass

        # ── New: Short-Squeeze Score ─────────────────────────────────────
        try:
            import yfinance as _yf_sq
            _sq_info = _yf_sq.Ticker(ticker).info or {}
            _short_float = _sq_info.get("shortPercentOfFloat", None)
            _short_ratio = _sq_info.get("shortRatio", None)
            if _short_float is not None:
                _sf_pct = float(_short_float) * 100
                _days_cover = float(_short_ratio) if _short_ratio else 0.0
                _sq_score = (80.0 if _sf_pct > 20 and _days_cover > 5 else
                             68.0 if _sf_pct > 15 else
                             50.0 if _sf_pct > 8 else 35.0)
                _squeeze_risk = _sf_pct > 20 and _days_cover > 5
                factor_scores["short_squeeze"] = FactorScore(
                    name="Short-Squeeze Score",
                    value=round(_sf_pct, 1),
                    score=_sq_score,
                    interpretation=(
                        f"Short float: {_sf_pct:.1f}% | Days to cover: {_days_cover:.1f}d — "
                        f"{'HIGH SQUEEZE RISK — crowded short' if _squeeze_risk else 'elevated short interest' if _sf_pct > 15 else 'moderate short interest' if _sf_pct > 8 else 'low short interest'}"
                    ),
                )
                if _squeeze_risk:
                    reasoning.append(f"Short-squeeze candidate: {_sf_pct:.0f}% float short, {_days_cover:.1f}d to cover.")
                    prob_up += 0.03
        except Exception:
            pass

        # ── New: Earnings Whisper Proxy (Implied Move vs Historical Move) ───
        # Options straddle price / stock price = implied expected move.
        # Compare to historical post-earnings moves to gauge whether the
        # market is pricing in a bigger-than-usual catalyst.
        try:
            import numpy as _np_ew
            _tkr_ew  = yf.Ticker(ticker)
            _exps_ew = _tkr_ew.options
            _info_ew = _tkr_ew.info or {}
            _curr_p  = (_info_ew.get("currentPrice") or
                        _info_ew.get("regularMarketPrice"))
            if _exps_ew and _curr_p and float(_curr_p) > 0:
                _cp = float(_curr_p)
                _chain_ew = _tkr_ew.option_chain(_exps_ew[0])
                # ATM = within 2% of current price, min 10 OI
                _atm_calls_ew = _chain_ew.calls[
                    (abs(_chain_ew.calls["strike"] - _cp) / _cp < 0.02) &
                    (_chain_ew.calls["openInterest"].fillna(0) > 10)
                ]
                _atm_puts_ew = _chain_ew.puts[
                    (abs(_chain_ew.puts["strike"] - _cp) / _cp < 0.02) &
                    (_chain_ew.puts["openInterest"].fillna(0) > 10)
                ]
                if not _atm_calls_ew.empty and not _atm_puts_ew.empty:
                    _c_bid = float(_atm_calls_ew["bid"].fillna(0).iloc[0])
                    _c_ask = float(_atm_calls_ew["ask"].fillna(0).iloc[0])
                    _p_bid = float(_atm_puts_ew["bid"].fillna(0).iloc[0])
                    _p_ask = float(_atm_puts_ew["ask"].fillna(0).iloc[0])
                    _straddle = ((_c_bid + _c_ask) / 2) + ((_p_bid + _p_ask) / 2)
                    _impl_move_pct = (_straddle / _cp) * 100

                    # Historical large-move days (>4%) as earnings-move proxy
                    _hist_ew = _tkr_ew.history(period="2y", interval="1d")
                    _hist_moves = []
                    if not _hist_ew.empty and len(_hist_ew) > 30:
                        _rets_ew = _hist_ew["Close"].pct_change().dropna().abs()
                        _hist_moves = list(_rets_ew[_rets_ew > 0.04].values * 100)

                    if _hist_moves and _impl_move_pct > 0:
                        _hist_avg_ew  = float(_np_ew.mean(
                            _hist_moves[-8:] if len(_hist_moves) >= 4 else _hist_moves
                        ))
                        _whisper_ratio = _impl_move_pct / _hist_avg_ew if _hist_avg_ew > 0 else 1.0
                        if _whisper_ratio > 1.5:
                            _ew_score, _ew_label = 70.0, "ELEVATED EVENT RISK"
                        elif _whisper_ratio > 1.2:
                            _ew_score, _ew_label = 62.0, "above-avg premium"
                        elif _whisper_ratio < 0.7:
                            _ew_score, _ew_label = 55.0, "options cheap — complacency?"
                        else:
                            _ew_score, _ew_label = 50.0, "normal premium"
                        factor_scores["earnings_whisper"] = FactorScore(
                            name="Earnings Whisper (Implied vs Historical Move)",
                            value=round(_whisper_ratio, 3),
                            score=_ew_score,
                            interpretation=(
                                f"Straddle implies {_impl_move_pct:.1f}% move vs "
                                f"hist avg {_hist_avg_ew:.1f}% — {_ew_label}"
                            ),
                        )
                        if _whisper_ratio > 1.5:
                            reasoning.append(
                                f"Earnings whisper: options pricing {_impl_move_pct:.1f}% move "
                                f"vs {_hist_avg_ew:.1f}% hist avg — elevated event risk."
                            )
        except Exception:
            pass

        # ── Clamp & Vote ──────────────────────────────────────────────────
        prob_up = max(0.01, min(0.99, prob_up))
        confidence = max(0.0, min(1.0, confidence))

        vote = (Direction.LONG if prob_up > self.long_threshold
                else Direction.SHORT if prob_up < self.short_threshold
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
