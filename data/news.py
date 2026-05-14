"""
AlphaAgent — News & Vector Database (ChromaDB)

Fetches news, creates embeddings using sentence-transformers,
and stores them in a local ChromaDB instance to enable
Retrieval-Augmented Generation (RAG) for the Sentiment Agent.
"""

import logging
from pathlib import Path
from datetime import datetime, timezone

import chromadb
from chromadb.utils import embedding_functions
import yfinance as yf

logger = logging.getLogger(__name__)

# Always relative to the project root, not CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_PATH = str(_PROJECT_ROOT / ".chroma_db")

# Articles older than this are pruned on each fetch to keep the collection bounded
_MAX_ARTICLE_AGE_DAYS = 7


class NewsDatabase:
    """Manages the local vector database for news RAG."""

    def __init__(self, db_path: str = _CHROMA_PATH):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="market_news",
            embedding_function=self.embedding_fn,
        )

    def _cleanup_old_articles(self) -> None:
        """Removes articles older than _MAX_ARTICLE_AGE_DAYS to bound collection size."""
        try:
            count = self.collection.count()
            if count == 0:
                return
            cutoff_ts = int(
                (datetime.now(timezone.utc).timestamp()) - _MAX_ARTICLE_AGE_DAYS * 86400
            )
            result = self.collection.get(where={"timestamp": {"$lt": cutoff_ts}})
            stale_ids = result.get("ids", [])
            if stale_ids:
                self.collection.delete(ids=stale_ids)
                logger.info(f"[News] Pruned {len(stale_ids)} stale articles (>{_MAX_ARTICLE_AGE_DAYS}d old).")
        except Exception as e:
            logger.warning(f"[News] TTL cleanup failed (non-fatal): {e}")

    def fetch_and_store_news(self, ticker: str) -> None:
        """Fetches latest news from yfinance, prunes stale articles, stores new ones."""
        self._cleanup_old_articles()

        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news

        if not news:
            return

        documents = []
        metadatas = []
        ids = []

        for article in news:
            title = article.get("title", "")
            publisher = article.get("publisher", "")
            url = article.get("link", "")
            timestamp = article.get("providerPublishTime", 0)

            if not title:
                continue

            content = f"[{publisher}] {title}"
            doc_id = f"{ticker}_{timestamp}_{hash(title) & 0xFFFFFFFF}"

            existing = self.collection.get(ids=[doc_id])
            if existing and len(existing["ids"]) > 0:
                continue

            documents.append(content)
            metadatas.append({"ticker": ticker, "url": url, "timestamp": timestamp})
            ids.append(doc_id)

        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"[News] Stored {len(documents)} new articles for {ticker}.")

    def search_news(
        self,
        query: str,
        ticker: str = None,
        n_results: int = 5,
        where: dict = None,
    ) -> list:
        """Retrieves the most semantically relevant news for a query."""
        if ticker and where:
            where_clause = {"$and": [{"ticker": {"$eq": ticker}}, where]}
        elif ticker:
            where_clause = {"ticker": {"$eq": ticker}}
        elif where:
            where_clause = where
        else:
            where_clause = None

        if self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where=where_clause,
            )
        except Exception:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self.collection.count()),
                )
            except Exception:
                return []

        if not results["documents"] or not results["documents"][0]:
            return []

        return [doc for doc in results["documents"][0]]
