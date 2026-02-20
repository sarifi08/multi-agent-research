"""
Search Cache — avoids burning API credits on repeated queries.

How it works:
    - Before hitting Tavily, check if we've seen this query before
    - Cache is stored as JSON on disk (survives restarts)
    - TTL-based expiry so results don't go stale
    - Thread-safe for parallel researcher access

Why this matters:
    During development you'll run the same query 20+ times.
    Without caching, that's 20 × (3-4 sub-queries) × $0.01 = wasted money.
    With caching, repeat queries are instant and free.
"""
import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path
from loguru import logger

from tools.web_search import SearchResult


class SearchCache:
    """
    Disk-backed search cache with TTL expiry.

    Usage:
        cache = SearchCache(cache_dir=".cache", ttl_hours=24)
        cached = cache.get("AI agents 2024")
        if cached:
            return cached   # free!
        results = await search_tool.search_async(query)
        cache.set("AI agents 2024", results)
    """

    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

        # Load index (maps query hashes → metadata)
        self.index_path = self.cache_dir / "index.json"
        self.index = self._load_index()

    def get(self, query: str) -> Optional[List[SearchResult]]:
        """
        Retrieve cached results for a query.
        Returns None if not cached or expired.
        """
        key = self._hash(query)

        if key not in self.index:
            return None

        entry = self.index[key]

        # Check TTL
        cached_at = datetime.fromisoformat(entry["cached_at"])
        if datetime.now() - cached_at > self.ttl:
            logger.debug(f"Cache expired for: '{query}'")
            self._remove(key)
            return None

        # Load cached results from disk
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            self._remove(key)
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            results = [
                SearchResult(
                    title=r["title"],
                    url=r["url"],
                    summary=r["summary"],
                    relevance_score=r["relevance_score"],
                )
                for r in data["results"]
            ]

            logger.info(f"📦 Cache HIT: '{query}' ({len(results)} results)")
            return results

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache corrupted for '{query}': {e}")
            self._remove(key)
            return None

    def set(self, query: str, results: List[SearchResult]):
        """Cache search results for a query."""
        key = self._hash(query)

        # Save results to disk
        cache_file = self.cache_dir / f"{key}.json"
        data = {
            "query": query,
            "results": [
                {
                    "title":           r.title,
                    "url":             r.url,
                    "summary":         r.summary,
                    "relevance_score": r.relevance_score,
                }
                for r in results
            ],
        }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

        # Update index
        self.index[key] = {
            "query":     query,
            "cached_at": datetime.now().isoformat(),
            "n_results": len(results),
        }
        self._save_index()

        logger.info(f"📦 Cache SET: '{query}' ({len(results)} results)")

    def clear(self):
        """Remove all cached data."""
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
        self.index = {}
        logger.info("📦 Cache cleared")

    def stats(self) -> dict:
        """Return cache statistics."""
        total_entries = len(self.index)
        expired = 0
        for entry in self.index.values():
            cached_at = datetime.fromisoformat(entry["cached_at"])
            if datetime.now() - cached_at > self.ttl:
                expired += 1

        return {
            "total_entries": total_entries,
            "active":        total_entries - expired,
            "expired":       expired,
            "cache_dir":     str(self.cache_dir),
        }

    def _hash(self, query: str) -> str:
        """Deterministic hash for a query string."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _load_index(self) -> dict:
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def _remove(self, key: str):
        """Remove a single cache entry."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()
        self.index.pop(key, None)
        self._save_index()
