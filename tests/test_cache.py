"""
Tests for SearchCache.

What we test:
    1. Cache miss returns None
    2. Cache hit returns stored results
    3. Expired entries return None
    4. Cache clear removes all entries
    5. Cache stats are accurate
    6. Corrupted cache files are handled gracefully
"""
import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta

from tools.cache import SearchCache
from tools.web_search import SearchResult


def make_results():
    """Create sample search results."""
    return [
        SearchResult("AI News", "https://example.com/1", "AI is changing the world", 0.9),
        SearchResult("ML Update", "https://example.com/2", "New ML techniques", 0.85),
    ]


class TestSearchCache:

    def setup_method(self):
        """Create a temp directory for each test."""
        self.tmp_dir = tempfile.mkdtemp()
        self.cache = SearchCache(cache_dir=self.tmp_dir, ttl_hours=24)

    def teardown_method(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cache_miss_returns_none(self):
        """Uncached query should return None."""
        result = self.cache.get("never seen this query")
        assert result is None

    def test_cache_hit_returns_results(self):
        """Cached query should return stored results."""
        results = make_results()
        self.cache.set("AI agents 2024", results)

        cached = self.cache.get("AI agents 2024")

        assert cached is not None
        assert len(cached) == 2
        assert cached[0].title == "AI News"
        assert cached[0].relevance_score == 0.9

    def test_case_insensitive(self):
        """Cache should be case-insensitive."""
        results = make_results()
        self.cache.set("AI Agents 2024", results)

        cached = self.cache.get("ai agents 2024")
        assert cached is not None
        assert len(cached) == 2

    def test_expired_entries_return_none(self):
        """Expired entries should return None."""
        # Create cache with 0-hour TTL (immediately expires)
        cache = SearchCache(cache_dir=self.tmp_dir, ttl_hours=0)
        cache.set("test query", make_results())

        # Should be expired
        result = cache.get("test query")
        assert result is None

    def test_cache_clear(self):
        """clear() should remove all cached data."""
        self.cache.set("query 1", make_results())
        self.cache.set("query 2", make_results())

        self.cache.clear()

        assert self.cache.get("query 1") is None
        assert self.cache.get("query 2") is None

    def test_cache_stats(self):
        """stats() should return accurate counts."""
        self.cache.set("query 1", make_results())
        self.cache.set("query 2", make_results())

        stats = self.cache.stats()

        assert stats["total_entries"] == 2
        assert stats["active"] == 2
        assert stats["expired"] == 0

    def test_handles_corrupted_cache_file(self):
        """Corrupted cache files should be handled gracefully."""
        results = make_results()
        self.cache.set("test query", results)

        # Corrupt the cache file
        key = self.cache._hash("test query")
        cache_file = os.path.join(self.tmp_dir, f"{key}.json")
        with open(cache_file, "w") as f:
            f.write("not valid json{{{")

        # Should return None without crashing
        result = self.cache.get("test query")
        assert result is None

    def test_handles_missing_cache_file(self):
        """If cache file is deleted but index exists, handle gracefully."""
        self.cache.set("test query", make_results())

        # Delete the cache file but keep the index
        key = self.cache._hash("test query")
        cache_file = os.path.join(self.tmp_dir, f"{key}.json")
        os.remove(cache_file)

        result = self.cache.get("test query")
        assert result is None

    def test_persistence_across_instances(self):
        """Cache should survive creating a new SearchCache instance."""
        self.cache.set("persistent query", make_results())

        # Create new instance pointing at same directory
        new_cache = SearchCache(cache_dir=self.tmp_dir, ttl_hours=24)

        cached = new_cache.get("persistent query")
        assert cached is not None
        assert len(cached) == 2
