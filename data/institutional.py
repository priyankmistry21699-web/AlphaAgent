"""
AlphaAgent — Institutional Data Layer (SEC EDGAR)

Fetches institutional ownership changes from SEC EDGAR.
Uses the free EDGAR full-text search and company facts APIs.
No API key required.

Endpoints:
  Company facts: https://data.sec.gov/api/xbrl/companyfacts/{CIK}.json
  Submissions:   https://data.sec.gov/submissions/CIK{CIK}.json
  Full-text:     https://efts.sec.gov/LATEST/search-index?q=...
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List

import requests
import yfinance as yf

from data.cache import DataCache

logger = logging.getLogger(__name__)

EDGAR_HEADERS = {
    "User-Agent": "AlphaAgent/1.0 contact@alphaagent.ai",
    "Accept-Encoding": "gzip, deflate",
}


@dataclass
class FilingRecord:
    """A single SEC filing record."""
    form_type: str
    filed_date: str
    description: str
    accession_number: str


@dataclass
class InstitutionalSnapshot:
    """Aggregated institutional data for a ticker."""
    ticker: str
    cik: str = ""
    recent_13f_changes: List[dict] = field(default_factory=list)  # hedge fund position changes
    recent_form4: List[dict] = field(default_factory=list)         # insider Form 4 trades
    recent_8k_count: int = 0                                       # material events in last 30d
    institutional_ownership_pct: float = 0.0                       # from yfinance
    warnings: List[str] = field(default_factory=list)


class InstitutionalData:
    """
    Fetches SEC EDGAR filings and institutional ownership data.
    Falls back gracefully when EDGAR is slow or unavailable.
    """

    EDGAR_TTL = 3600 * 6  # 6 hours

    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()
        self._session = requests.Session()
        self._session.headers.update(EDGAR_HEADERS)

    # ── CIK Lookup ────────────────────────────────────────────────────────
    def get_cik(self, ticker: str) -> str:
        """Resolve a ticker to its SEC CIK number."""
        cache_key = f"cik_{ticker}"
        cached = self.cache.get("edgar", "CIK", cache_key)
        if cached is not None:
            return cached.get("cik", "")

        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            r = self._session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    self.cache.set("edgar", "CIK", cache_key, {"cik": cik},
                                   ttl_seconds=86400 * 7)
                    return cik
        except Exception as e:
            logger.warning(f"[EDGAR] CIK lookup failed for {ticker}: {e}")

        return ""

    # ── Recent Filings ────────────────────────────────────────────────────
    def get_recent_filings(self, ticker: str, form_types: List[str] = None) -> List[FilingRecord]:
        """
        Fetch recent SEC filings for a company.
        form_types: e.g. ["4", "13F-HR", "8-K"]
        """
        if form_types is None:
            form_types = ["4", "8-K", "13F-HR"]

        cik = self.get_cik(ticker)
        if not cik:
            return []

        cache_key = f"filings_{cik}_{'_'.join(form_types)}"
        cached = self.cache.get("edgar", "FILINGS", cache_key)
        if cached is not None:
            return [FilingRecord(**r) for r in cached]

        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            r = self._session.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()

            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            descriptions = filings.get("primaryDocument", [])
            accessions = filings.get("accessionNumber", [])

            records = []
            for form, date, desc, acc in zip(forms, dates, descriptions, accessions):
                if form in form_types:
                    records.append(FilingRecord(
                        form_type=form,
                        filed_date=date,
                        description=desc,
                        accession_number=acc,
                    ))
                if len(records) >= 50:
                    break

            self.cache.set("edgar", "FILINGS", cache_key,
                           [r.__dict__ for r in records],
                           ttl_seconds=self.EDGAR_TTL)
            return records

        except Exception as e:
            logger.warning(f"[EDGAR] Filing fetch failed for {ticker}: {e}")
            return []

    # ── Form 4 Analysis ───────────────────────────────────────────────────
    def analyze_form4(self, ticker: str) -> List[dict]:
        """
        Parse recent Form 4 (insider transaction) filings.
        Returns simplified dicts with date, type, and estimated share count.
        """
        filings = self.get_recent_filings(ticker, form_types=["4"])
        result = []
        for f in filings[:20]:
            result.append({
                "date": f.filed_date,
                "form": f.form_type,
                "description": f.description,
                "accession": f.accession_number,
            })
        return result

    # ── Institutional Ownership ───────────────────────────────────────────
    def get_institutional_ownership(self, ticker: str) -> float:
        """
        Returns institutional ownership percentage via yfinance info.
        Falls back to 0.0 if unavailable.
        """
        try:
            info = yf.Ticker(ticker).info
            held_pct = info.get("heldPercentInstitutions", 0.0)
            return float(held_pct) * 100.0
        except Exception:
            return 0.0

    # ── 8-K Count (Material Events) ───────────────────────────────────────
    def count_recent_8k(self, ticker: str, days: int = 30) -> int:
        """Count 8-K filings in the last N days."""
        from datetime import datetime, timedelta
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        filings = self.get_recent_filings(ticker, form_types=["8-K"])
        return sum(1 for f in filings if f.filed_date >= cutoff)

    # ── Full Snapshot ─────────────────────────────────────────────────────
    def get_snapshot(self, ticker: str) -> InstitutionalSnapshot:
        snap = InstitutionalSnapshot(ticker=ticker)
        snap.cik = self.get_cik(ticker)
        snap.recent_form4 = self.analyze_form4(ticker)
        snap.recent_8k_count = self.count_recent_8k(ticker)
        snap.institutional_ownership_pct = self.get_institutional_ownership(ticker)
        return snap
