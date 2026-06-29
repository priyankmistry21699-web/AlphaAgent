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

            # Dynamic cluster threshold: scale by market cap
            # Small-cap (<$10B): 3 filings = cluster; mega-cap (>$100B): 8 filings
            try:
                import yfinance as _yf_mcap_f4
                _mcap_f4 = _yf_mcap_f4.Ticker(ticker).info.get("marketCap", 0) or 0
                _f4_thresh = 3 if _mcap_f4 < 10e9 else 5 if _mcap_f4 < 100e9 else 8
            except Exception:
                _f4_thresh = 5

            # Form 4 cluster signal: many filings in past 60d = insider activity
            if form4_count >= _f4_thresh:
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
            # Cluster threshold scales with market cap (small-cap more signal-rich)
            try:
                import yfinance as _yf_mcap_c
                _mcap_c = _yf_mcap_c.Ticker(ticker).info.get("marketCap", 0) or 0
                _cluster_thresh = 2 if _mcap_c < 10e9 else 3 if _mcap_c < 100e9 else 5
            except Exception:
                _cluster_thresh = 3
            if form4_filings:
                from collections import Counter
                import datetime
                today = __import__("datetime").date.today()
                recent_30d = [f for f in form4_filings
                              if hasattr(f, 'filed_at') and
                              (today - __import__("datetime").date.fromisoformat(str(f.filed_at)[:10])).days <= 30]
                if len(recent_30d) >= _cluster_thresh:
                    factor_scores["insider_cluster_30d"] = FactorScore(
                        name="Insider Cluster (30-Day)",
                        value=float(len(recent_30d)),
                        score=75.0,
                        interpretation=f"{len(recent_30d)} insider filings in 30 days — cluster buying signal",
                    )
        except Exception:
            pass

        # ── New: Congressional Trading Signal ────────────────────────────
        try:
            import requests as _req_cong
            _cong_url = "https://house-stock-watcher-data.s3-us-gov-west-1.amazonaws.com/data/all_transactions.json"
            _cong_resp = _req_cong.get(_cong_url, timeout=8)
            if _cong_resp.ok:
                import datetime as _dt_cong
                _cong_trades = _cong_resp.json()
                _today_cong  = _dt_cong.date.today()
                _ticker_up   = ticker.upper()
                _recent_trades = [
                    t for t in _cong_trades
                    if t.get("ticker", "").upper() == _ticker_up
                    and ((_today_cong - _dt_cong.date.fromisoformat(
                        t.get("transaction_date", "2000-01-01")[:10]
                    )).days <= 180)
                ]
                if _recent_trades:
                    _buy_c  = sum(1 for t in _recent_trades if "purchase" in t.get("type", "").lower())
                    _sell_c = sum(1 for t in _recent_trades if "sale" in t.get("type", "").lower())
                    _net_c  = _buy_c - _sell_c
                    _cong_score = (75.0 if _net_c > 0 else 45.0 if _net_c == 0 else 25.0)
                    factor_scores["congressional_trading"] = FactorScore(
                        name="Congressional Trading Signal",
                        value=float(_net_c),
                        score=_cong_score,
                        interpretation=f"Congress trades (180d): {_buy_c} buys, {_sell_c} sells → net {_net_c:+d} ({'insider buying signal' if _net_c > 0 else 'net selling — caution' if _net_c < 0 else 'neutral activity'})",
                    )
                    if _buy_c > 2:
                        edgar_reasoning.append(f"Congressional buying: {_buy_c} members purchased {ticker} recently.")
        except Exception:
            pass

        # ── New: 13F Institutional Ownership Change (QoQ) ────────────────
        try:
            import yfinance as _yf_13f
            _inst_h = _yf_13f.Ticker(ticker).institutional_holders
            if _inst_h is not None and not _inst_h.empty and "% Out" in _inst_h.columns:
                if "Date Reported" in _inst_h.columns:
                    _inst_dates = _inst_h["Date Reported"].dropna().unique()
                    if len(_inst_dates) >= 2:
                        _d_recent = sorted(_inst_dates)[-1]
                        _d_prior  = sorted(_inst_dates)[-2]
                        _pct_now  = float(_inst_h[_inst_h["Date Reported"] == _d_recent]["% Out"].sum()) * 100
                        _pct_prev = float(_inst_h[_inst_h["Date Reported"] == _d_prior]["% Out"].sum()) * 100
                        _13f_delta = _pct_now - _pct_prev
                        _13f_score = (75.0 if _13f_delta > 2 else 55.0 if _13f_delta > 0 else 40.0 if _13f_delta > -2 else 20.0)
                        factor_scores["inst_13f_delta"] = FactorScore(
                            name="13F Institutional Change (QoQ)",
                            value=round(_13f_delta, 2),
                            score=_13f_score,
                            interpretation=f"Institutional holdings: {_pct_now:.1f}% vs {_pct_prev:.1f}% prior → Δ{_13f_delta:+.1f}pp ({'accumulation' if _13f_delta > 1 else 'distribution' if _13f_delta < -1 else 'stable'})",
                        )
                    else:
                        _inst_pct_single = float(_inst_h["% Out"].sum()) * 100
                        factor_scores["inst_13f_delta"] = FactorScore(
                            name="13F Institutional Ownership",
                            value=round(_inst_pct_single, 2),
                            score=(70.0 if _inst_pct_single > 70 else 55.0 if _inst_pct_single > 50 else 40.0),
                            interpretation=f"Institutional ownership: {_inst_pct_single:.1f}% ({'high conviction' if _inst_pct_single > 70 else 'moderate' if _inst_pct_single > 50 else 'low institutional interest'})",
                        )
        except Exception:
            pass

        # ── New: FINRA Short Sale Volume (Dark Pool Proxy) ────────────────
        try:
            import requests as _req_dp
            import datetime as _dt_dp
            _today_dp = _dt_dp.date.today()
            _days_to_fri = (_today_dp.weekday() - 4) % 7 or 7
            _last_fri_dp = _today_dp - _dt_dp.timedelta(days=_days_to_fri)
            _dp_found = False
            for _wk in range(4):
                _fri_dt = _last_fri_dp - _dt_dp.timedelta(weeks=_wk)
                _dp_url = f"https://cdn.finra.org/equity/regsho/weekly/CNMSweekly{_fri_dt.strftime('%Y%m%d')}.txt"
                try:
                    _dp_resp = _req_dp.get(_dp_url, timeout=8)
                    if _dp_resp.ok and len(_dp_resp.text) > 200:
                        for _dp_line in _dp_resp.text.strip().split("\n")[1:]:
                            _dp_cols = _dp_line.split("|")
                            if len(_dp_cols) >= 4 and _dp_cols[0].upper() == ticker.upper():
                                _short_v = float(_dp_cols[1])
                                _total_v = float(_dp_cols[3])
                                if _total_v > 0:
                                    _short_pct = _short_v / _total_v * 100
                                    _dp_score = (70.0 if _short_pct < 45 else 50.0 if _short_pct < 55 else 25.0)
                                    factor_scores["finra_short_volume"] = FactorScore(
                                        name="FINRA Short Sale Volume %",
                                        value=round(_short_pct, 1),
                                        score=_dp_score,
                                        interpretation=f"RegSHO: short vol {_short_pct:.1f}% of weekly total — {'bearish institutional pressure' if _short_pct > 55 else 'low short activity' if _short_pct < 45 else 'normal range'}",
                                    )
                                    if _short_pct > 60:
                                        edgar_reasoning.append(f"FINRA short volume elevated: {_short_pct:.1f}% — institutional bearish pressure.")
                                    _dp_found = True
                                break
                        if _dp_found:
                            break
                except Exception:
                    pass
        except Exception:
            pass

        # ── New: ETF Flow Impact (Sector ETF Volume-Price Proxy) ─────────
        try:
            import yfinance as _yf_ef
            import numpy as _np_ef
            _sector_etf_map_ef = {
                "Technology": "XLK", "Communication Services": "XLC",
                "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
                "Health Care": "XLV", "Industrials": "XLI", "Materials": "XLB",
                "Energy": "XLE", "Financials": "XLF", "Real Estate": "XLRE", "Utilities": "XLU",
            }
            _tkr_info_ef = _yf_ef.Ticker(ticker).info
            _sector_ef = _tkr_info_ef.get("sector", "") if isinstance(_tkr_info_ef, dict) else ""
            _sec_etf_ef = _sector_etf_map_ef.get(_sector_ef)
            if _sec_etf_ef:
                _etf_hist = _yf_ef.download(_sec_etf_ef, period="3mo", interval="1d",
                                            auto_adjust=True, progress=False)
                if not _etf_hist.empty and len(_etf_hist) >= 22:
                    _etf_close = _etf_hist["Close"].squeeze().dropna()
                    _etf_vol   = _etf_hist["Volume"].squeeze().dropna()
                    _vol_now   = float(_etf_vol.iloc[-5:].mean())
                    _vol_hist  = float(_etf_vol.iloc[-22:-5].mean())
                    _vol_ratio = _vol_now / _vol_hist if _vol_hist > 0 else 1.0
                    _price_mom = float(_etf_close.iloc[-1] / _etf_close.iloc[-22] - 1) * 100
                    _flow_signal = _price_mom * float(_np_ef.log1p(_vol_ratio))
                    _ef_score = max(10.0, min(90.0, 50.0 + _flow_signal * 1.5))
                    factor_scores["etf_flow_impact"] = FactorScore(
                        name=f"ETF Flow Impact ({_sec_etf_ef})",
                        value=round(_flow_signal, 2),
                        score=_ef_score,
                        interpretation=f"{_sec_etf_ef} flow proxy: {_price_mom:+.1f}% × vol ratio {_vol_ratio:.2f} → {_flow_signal:+.1f} — {'sector inflow (bullish)' if _flow_signal > 3 else 'sector outflow (bearish)' if _flow_signal < -3 else 'neutral flow'}",
                    )
        except Exception:
            pass

        # ── New: Activist 13D/G Filing (EDGAR EFTS) ──────────────────────
        try:
            import requests as _req_act
            import datetime as _dt_act
            _act_start = (_dt_act.date.today() - _dt_act.timedelta(days=90)).isoformat()
            _act_url = (
                f"https://efts.sec.gov/LATEST/search-index?"
                f"q=%22{ticker}%22&forms=SC+13D,SC+13G"
                f"&dateRange=custom&startdt={_act_start}"
            )
            _act_resp = _req_act.get(
                _act_url, timeout=5,
                headers={"User-Agent": "AlphaAgent alphaagent@research.example.com"}
            )
            if _act_resp.ok:
                _act_data = _act_resp.json()
                _act_hits = _act_data.get("hits", {}).get("total", {})
                _act_count = _act_hits.get("value", 0) if isinstance(_act_hits, dict) else int(_act_hits)
                _act_score = (80.0 if _act_count > 0 else 50.0)
                factor_scores["activist_13d"] = FactorScore(
                    name="Activist 13D/G Filing",
                    value=float(_act_count),
                    score=_act_score,
                    interpretation=f"SC 13D/G filings (90d): {_act_count} — {'ACTIVIST INVESTOR — takeover/change catalyst' if _act_count > 0 else 'no activist filing'}",
                )
                if _act_count > 0:
                    warnings.append(f"ACTIVIST SIGNAL: {_act_count} SC 13D/G filing(s) in 90 days — activist investor pressure.")
                    edgar_reasoning.append(f"Activist 13D/G detected — hedge fund may be pushing for strategic change.")
        except Exception:
            pass

        # ── New: Top-10 Holder Concentration (Herfindahl Index) ──────────
        try:
            import yfinance as _yf_hhi
            import numpy as _np_hhi
            _inst_hhi = _yf_hhi.Ticker(ticker).institutional_holders
            if _inst_hhi is not None and not _inst_hhi.empty and "% Out" in _inst_hhi.columns:
                _pcts_raw = _inst_hhi["% Out"].dropna().values[:10]
                _pcts = _np_hhi.array([
                    float(p) if float(p) < 1.0 else float(p) / 100.0
                    for p in _pcts_raw
                ], dtype=float)
                if len(_pcts) > 0 and _pcts.sum() > 0:
                    _hhi_raw = float(_np_hhi.sum(_pcts ** 2))
                    _hhi_score = (60.0 if 0.05 < _hhi_raw < 0.25 else 40.0 if _hhi_raw > 0.25 else 55.0)
                    factor_scores["top10_concentration"] = FactorScore(
                        name="Top-10 Holder Concentration (HHI)",
                        value=round(_hhi_raw, 4),
                        score=_hhi_score,
                        interpretation=f"Herfindahl Index: {_hhi_raw:.4f} — {'high concentration — activist/block risk' if _hhi_raw > 0.25 else 'dispersed base — stable ownership' if _hhi_raw < 0.05 else 'moderate concentration'}",
                    )
        except Exception:
            pass

        # ── New: Dark Pool Print Ratio (FINRA ADF Volume) ───────────────
        try:
            import urllib.request as _req_dp
            import json as _json_dp
            _dp_url = f"https://regsho.finra.org/regsho-Index.html"
            _dp_found = False
            _dp_url2 = f"https://api.finra.org/data/group/otcmarket/name/weeklySummary?limit=5&offset=0&compareFilters=issueSymbolIdentifier%3Aeq%3A{ticker.upper()}"
            try:
                _req_dp2 = _req_dp.Request(_dp_url2, headers={"User-Agent": "AlphaAgent/1.0", "Accept": "application/json"})
                with _req_dp.urlopen(_req_dp2, timeout=4) as _resp_dp:
                    _data_dp = _json_dp.loads(_resp_dp.read().decode())
                if _data_dp and isinstance(_data_dp, list) and len(_data_dp) > 0:
                    _row_dp = _data_dp[0]
                    _adf_vol = float(_row_dp.get("totalWeeklyShareQuantity", 0) or 0)
                    _tot_vol = float(_row_dp.get("totalReportedWeeklyShareQuantity", 0) or _adf_vol)
                    if _tot_vol > 0 and _adf_vol > 0:
                        _dp_ratio = _adf_vol / _tot_vol * 100
                        _dp_score = (72.0 if _dp_ratio > 50 else 55.0 if _dp_ratio > 35 else 45.0)
                        factor_scores["dark_pool_ratio"] = FactorScore(
                            name="Dark Pool Print Ratio (FINRA ADF)",
                            value=round(_dp_ratio, 1),
                            score=_dp_score,
                            interpretation=(
                                f"ADF/OTC vol: {_dp_ratio:.1f}% of reported weekly volume — "
                                f"{'high dark pool activity (institutional accumulation likely)' if _dp_ratio > 50 else 'moderate off-exchange activity' if _dp_ratio > 35 else 'low dark pool presence'}"
                            ),
                        )
                        if _dp_ratio > 55:
                            edgar_reasoning.append(f"High dark pool volume ({_dp_ratio:.0f}%) — institutional off-exchange activity elevated.")
                        _dp_found = True
            except Exception:
                pass
            if not _dp_found:
                import yfinance as _yf_dp
                _hist_dp = _yf_dp.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
                if not _hist_dp.empty and "Volume" in _hist_dp.columns:
                    _vol_5d = float(_hist_dp["Volume"].squeeze().dropna().mean())
                    _vol_hist_dp = _yf_dp.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
                    _vol_avg_dp = float(_vol_hist_dp["Volume"].squeeze().dropna().mean()) if not _vol_hist_dp.empty else _vol_5d
                    _vol_ratio_dp = _vol_5d / _vol_avg_dp if _vol_avg_dp > 0 else 1.0
                    _dp_proxy = min(90.0, max(10.0, 40.0 + (_vol_ratio_dp - 1.0) * 20.0))
                    factor_scores["dark_pool_ratio"] = FactorScore(
                        name="Dark Pool Proxy (Volume vs Avg)",
                        value=round(_vol_ratio_dp, 2),
                        score=_dp_proxy,
                        interpretation=f"5d avg vol vs 3M avg: {_vol_ratio_dp:.2f}x — {'elevated institutional activity' if _vol_ratio_dp > 1.5 else 'normal volume' if _vol_ratio_dp > 0.7 else 'below-avg volume'} (FINRA ADF unavailable)",
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
        try:
            _price = data.get_current_price()
            _insider_dollar_value = abs(result.net_insider_shares) * (_price or 1.0)
            if _insider_dollar_value > 1_000_000:
                confidence += 0.2
        except Exception:
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
