"""
AlphaAgent — SEC 10-K NLP Language Shift

Theory (Bernard & Thomas 1989 / Cohen et al. 2020):
  Year-over-year changes in SEC 10-K risk-factor language predict negative
  abnormal returns.  More negative language → worse subsequent performance.
  Measured via cosine distance between consecutive 10-K risk sections.

Current status: stub returning None (graceful degradation).
  Full implementation requires SEC EDGAR XBRL API + transformer embeddings.
  When implemented, inject into agents/fundamental.py via compute_10k_shift().

Planned implementation:
  1. Fetch current-year and prior-year 10-K from SEC EDGAR EDGAR full-text API
  2. Extract "Risk Factors" section via regex
  3. Embed both sections with sentence-transformers (all-MiniLM-L6-v2)
  4. Cosine similarity < 0.85 → language shifted significantly
  5. Sentiment delta (VADER/FinBERT) on changed sentences
  6. Score 0–100: high negative shift → bearish (score 20-35), stable/positive → 60-75
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SEC10KResult:
    score: float               # 0-100 factor score
    prob_adjustment: float     # -0.06 to +0.04
    cosine_similarity: float   # 1.0 = unchanged, 0.0 = completely different
    sentiment_delta: float     # negative = more bearish language YoY
    sections_compared: int     # number of risk-factor paragraphs compared
    filing_year: int
    regime: str                # STABLE / MODERATE_SHIFT / HIGH_SHIFT / EXTREME_SHIFT


def compute_10k_shift(ticker: str) -> Optional[SEC10KResult]:
    """
    Compute year-over-year SEC 10-K risk-factor language shift.
    Returns None until SEC EDGAR NLP pipeline is implemented.
    """
    logger.debug(f"[SEC NLP] {ticker}: stub — returning None")
    return None
