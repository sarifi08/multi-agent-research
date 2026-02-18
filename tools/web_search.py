"""
Web Search Tool — truly async implementation using aiohttp.

Why rewritten:
    Previous version used asyncio.to_thread() which is a workaround —
    it runs synchronous code in a thread pool, not true async.

    This version uses aiohttp for genuinely non-blocking HTTP requests.
    Multiple searches truly run in parallel without blocking each other.
"""
import aiohttp
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class SearchResult:
    """One search result returned to the Researcher agent."""
    title: str
    url: str
    summary: str            # max 500 chars — Analyst will process further
    relevance_score: float  # 0-1, scored by Tavily


class WebSearchTool:

    TAVILY_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, max_results: int = 5):
        self.api_key = api_key
        self.max_results = max_results

    async def search_async(self, query: str) -> List[SearchResult]:
        """
        Truly async web search — doesn't block other searches from running.
        Called by Researcher when running in parallel.
        """
        logger.info(f"🔍 Searching: '{query}'")

        payload = {
            "api_key":      self.api_key,
            "query":        query,
            "max_results":  self.max_results,
            "search_depth": "advanced",
            "include_answer": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.TAVILY_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Tavily returned status {response.status}")
                        return []

                    data = await response.json()
                    return self._parse_results(data)

        except asyncio.TimeoutError:
            logger.error(f"Search timed out for '{query}'")
            return []
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []

    def _parse_results(self, data: dict) -> List[SearchResult]:
        """Parse Tavily response into clean SearchResult objects."""
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", "No title"),
                url=item.get("url", ""),
                summary=item.get("content", "")[:500],
                relevance_score=item.get("score", 0.0)
            ))
        return results

    def is_useful(
        self,
        results: List[SearchResult],
        threshold: float = 0.5
    ) -> bool:
        """
        Are these results good enough, or should Researcher retry?
        Returns False if empty or average relevance is below threshold.
        """
        if not results:
            return False
        avg_score = sum(r.relevance_score for r in results) / len(results)
        return avg_score >= threshold
