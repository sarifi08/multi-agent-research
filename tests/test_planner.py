"""
Tests for PlannerAgent.

What we test:
    1. Planner produces the right number of sub-queries
    2. Planner writes to session correctly
    3. Planner handles bad LLM output gracefully
"""
import pytest
from unittest.mock import MagicMock, patch
from agents.planner import PlannerAgent
from core.session import ResearchSession, AgentStatus
from monitoring.tracker import MonitoringTracker


def make_session(query="Research AI trends in healthcare"):
    return ResearchSession(query=query)


def make_tracker(session):
    return MonitoringTracker(session, model="gpt-4o")


def mock_openai_response(content: str, input_tokens=100, output_tokens=50):
    """Helper: fake an OpenAI API response."""
    response = MagicMock()
    response.choices[0].message.content = content
    response.usage.prompt_tokens     = input_tokens
    response.usage.completion_tokens = output_tokens
    return response


class TestPlannerAgent:

    @patch("agents.planner.OpenAI")
    def test_produces_sub_queries(self, mock_openai_class):
        """Planner should return a list of sub-queries."""
        # Arrange
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response(
            '["AI diagnostics 2024", "ML in medical imaging", "AI drug discovery"]'
        )

        session = make_session()
        tracker = make_tracker(session)
        planner = PlannerAgent(api_key="fake-key")

        # Act
        queries = planner.plan(session, tracker)

        # Assert
        assert len(queries) == 3
        assert all(isinstance(q, str) for q in queries)
        assert "AI diagnostics 2024" in queries

    @patch("agents.planner.OpenAI")
    def test_writes_to_session(self, mock_openai_class):
        """Planner should write sub_queries to the shared session."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response(
            '["query 1", "query 2"]'
        )

        session = make_session()
        tracker = make_tracker(session)
        planner = PlannerAgent(api_key="fake-key")

        planner.plan(session, tracker)

        # Session should have sub_queries populated
        assert session.sub_queries == ["query 1", "query 2"]
        # Status should be DONE
        assert session.agent_statuses["planner"] == AgentStatus.DONE

    @patch("agents.planner.OpenAI")
    def test_handles_bad_llm_output(self, mock_openai_class):
        """Planner should handle messy LLM output without crashing."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        # LLM returns messy format — newline separated instead of Python list
        mock_client.chat.completions.create.return_value = mock_openai_response(
            "- AI diagnostics\n- ML imaging\n- drug discovery"
        )

        session = make_session()
        tracker = make_tracker(session)
        planner = PlannerAgent(api_key="fake-key")

        # Should not crash
        queries = planner.plan(session, tracker)
        assert isinstance(queries, list)
        assert len(queries) > 0

    @patch("agents.planner.OpenAI")
    def test_session_logs_are_populated(self, mock_openai_class):
        """Session audit log should have entries after Planner runs."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response(
            '["query 1", "query 2"]'
        )

        session = make_session("What is quantum computing?")
        tracker = make_tracker(session)
        planner = PlannerAgent(api_key="fake-key")

        planner.plan(session, tracker)

        # Should have log entries
        assert len(session.logs) > 0
        # At least one log should mention the query
        assert any("quantum computing" in log for log in session.logs)
