"""
Tests for AnalystAgent.

What we test:
    1. Filters out low-score results before calling LLM
    2. Handles failed Researcher outputs gracefully
    3. Returns empty list when nothing passes filter
    4. Writes findings to session correctly
"""
import pytest
from unittest.mock import MagicMock, patch
from agents.analyst import AnalystAgent
from tools.web_search import SearchResult
from core.session import ResearchSession, AgentStatus
from monitoring.tracker import MonitoringTracker


def make_session(raw_results=None):
    session = ResearchSession(query="AI trends in healthcare")
    if raw_results:
        session.raw_results = raw_results
    return session


def make_tracker(session):
    return MonitoringTracker(session, model="gpt-4o")


def make_raw_result(success=True, score=0.8, n_results=3):
    """Helper to create a fake Researcher output."""
    results = [
        SearchResult(f"Title {i}", f"https://example.com/{i}", f"Summary {i}", score)
        for i in range(n_results)
    ]
    return {
        "query":    "test query",
        "results":  results if success else [],
        "success":  success,
        "attempts": 1
    }


def mock_judgment_response(relevant: bool, score: float = 0.8):
    """Helper: fake the LLM judgment response."""
    response = MagicMock()
    response.choices[0].message.content = (
        f"RELEVANT: {'yes' if relevant else 'no'}\n"
        f"SCORE: {score}\n"
        f"REASON: This is relevant because it covers the topic"
    )
    response.usage.prompt_tokens     = 100
    response.usage.completion_tokens = 30
    return response


class TestAnalystAgent:

    @patch("agents.analyst.OpenAI")
    def test_filters_low_score_results(self, mock_openai_class):
        """Results with score below 0.5 should not reach LLM judgment."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_judgment_response(True)

        # Mix of high and low score results
        high_score = make_raw_result(score=0.8, n_results=2)
        low_score  = make_raw_result(score=0.2, n_results=3)

        session = make_session([high_score, low_score])
        tracker = make_tracker(session)
        analyst = AnalystAgent(api_key="fake-key")

        analyst.analyze(session, tracker)

        # LLM should only be called for the 2 high-score results, not 3 low-score ones
        assert mock_client.chat.completions.create.call_count == 2

    @patch("agents.analyst.OpenAI")
    def test_handles_failed_researcher_output(self, mock_openai_class):
        """Failed Researcher outputs should be skipped gracefully."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_judgment_response(True)

        good_result   = make_raw_result(success=True,  score=0.8)
        failed_result = make_raw_result(success=False)  # Researcher gave up

        session = make_session([good_result, failed_result])
        tracker = make_tracker(session)
        analyst = AnalystAgent(api_key="fake-key")

        # Should not crash
        findings = analyst.analyze(session, tracker)
        assert isinstance(findings, list)

    @patch("agents.analyst.OpenAI")
    def test_returns_empty_when_nothing_passes(self, mock_openai_class):
        """When LLM rejects everything, return empty list."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        # LLM rejects all results
        mock_client.chat.completions.create.return_value = mock_judgment_response(
            relevant=False
        )

        session = make_session([make_raw_result(score=0.8)])
        tracker = make_tracker(session)
        analyst = AnalystAgent(api_key="fake-key")

        findings = analyst.analyze(session, tracker)
        assert findings == []

    @patch("agents.analyst.OpenAI")
    def test_writes_to_session(self, mock_openai_class):
        """Analyst should write approved findings to session.findings."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_judgment_response(
            relevant=True, score=0.9
        )

        session = make_session([make_raw_result(score=0.8, n_results=1)])
        tracker = make_tracker(session)
        analyst = AnalystAgent(api_key="fake-key")

        analyst.analyze(session, tracker)

        assert len(session.findings) == 1
        assert len(session.sources)  == 1
        assert session.agent_statuses["analyst"] == AgentStatus.DONE
