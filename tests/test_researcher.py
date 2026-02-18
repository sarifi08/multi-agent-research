"""
Tests for ResearcherAgent.

What we test:
    1. Returns results on first successful search
    2. Retries when results are poor quality
    3. Returns empty dict after all retries exhausted
    4. Writes to session correctly
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from agents.researcher import ResearcherAgent
from tools.web_search import SearchResult
from core.session import ResearchSession
from monitoring.tracker import MonitoringTracker


def make_session():
    return ResearchSession(query="AI trends in healthcare")


def make_tracker(session):
    return MonitoringTracker(session, model="gpt-4o")


def make_good_results():
    """Simulate useful search results (high relevance score)."""
    return [
        SearchResult("AI in radiology", "https://example.com/1", "AI improves diagnosis", 0.85),
        SearchResult("ML healthcare", "https://example.com/2", "ML transforms care", 0.80),
    ]


def make_poor_results():
    """Simulate useless search results (low relevance score)."""
    return [
        SearchResult("Random article", "https://example.com/3", "Unrelated content", 0.2),
    ]


class TestResearcherAgent:

    def test_returns_results_on_first_success(self):
        """When first search is good, no retry needed."""
        search_tool = MagicMock()
        search_tool.search_async = AsyncMock(return_value=make_good_results())
        search_tool.is_useful.return_value = True

        session = make_session()
        tracker = make_tracker(session)

        researcher = ResearcherAgent(
            api_key="fake-key",
            search_tool=search_tool,
            max_retries=2
        )

        result = asyncio.run(
            researcher.research_async("AI diagnostics 2024", session, tracker)
        )

        assert result["success"] is True
        assert result["attempts"] == 1
        assert len(result["results"]) == 2

    def test_retries_on_poor_results(self):
        """When first search is poor, should retry with rephrased query."""
        search_tool = MagicMock()

        # First search poor, second search good
        search_tool.search_async = AsyncMock(
            side_effect=[make_poor_results(), make_good_results()]
        )
        search_tool.is_useful.side_effect = [False, True]

        session = make_session()
        tracker = make_tracker(session)

        researcher = ResearcherAgent(
            api_key="fake-key",
            search_tool=search_tool,
            max_retries=2
        )

        # Mock the rephrase call
        with patch.object(researcher, "_ask_planner_to_rephrase",
                         return_value=("better query", {"input": 50, "output": 20})):
            result = asyncio.run(
                researcher.research_async("bad query", session, tracker)
            )

        assert result["success"] is True
        assert result["attempts"] == 2  # took 2 tries

    def test_returns_empty_after_all_retries_fail(self):
        """When all retries fail, returns empty result (triggers C logic)."""
        search_tool = MagicMock()
        search_tool.search_async = AsyncMock(return_value=make_poor_results())
        search_tool.is_useful.return_value = False  # always poor

        session = make_session()
        tracker = make_tracker(session)

        researcher = ResearcherAgent(
            api_key="fake-key",
            search_tool=search_tool,
            max_retries=2
        )

        with patch.object(researcher, "_ask_planner_to_rephrase",
                         return_value=("rephrased", {"input": 50, "output": 20})):
            result = asyncio.run(
                researcher.research_async("impossible query", session, tracker)
            )

        assert result["success"] is False
        assert result["results"] == []
        assert result["attempts"] == 3   # tried max_retries + 1 times

    def test_writes_to_session(self):
        """Researcher should append its output to session.raw_results."""
        search_tool = MagicMock()
        search_tool.search_async = AsyncMock(return_value=make_good_results())
        search_tool.is_useful.return_value = True

        session = make_session()
        tracker = make_tracker(session)

        researcher = ResearcherAgent(
            api_key="fake-key",
            search_tool=search_tool
        )

        asyncio.run(researcher.research_async("test query", session, tracker))

        # Session should have raw results
        assert len(session.raw_results) == 1
        assert session.raw_results[0]["success"] is True
