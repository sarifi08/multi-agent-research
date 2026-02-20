"""
Tests for Orchestrator.

What we test:
    1. Orchestrator initializes all agents correctly
    2. Full pipeline runs end-to-end with mocked agents
    3. Pipeline handles agent failure gracefully
    4. Session summary contains expected fields
    5. model_override parameter works
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from core.orchestrator import Orchestrator
from core.session import AgentStatus


def mock_settings(**overrides):
    """Create mock settings."""
    defaults = {
        "openai_api_key": "fake-key",
        "tavily_api_key": "fake-tavily",
        "llm_model": "gpt-4o",
        "max_output_tokens": 2000,
        "max_retries": 2,
        "max_parallel_searches": 3,
        "max_search_results": 5,
        "enable_tracking": True,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


class TestOrchestrator:

    @patch("core.orchestrator.SearchCache")
    @patch("core.orchestrator.get_settings")
    def test_initializes_all_agents(self, mock_get_settings, mock_cache):
        """Orchestrator should create Planner, Analyst, Writer, and SearchTool."""
        mock_get_settings.return_value = mock_settings()

        orch = Orchestrator()

        assert orch.planner is not None
        assert orch.analyst is not None
        assert orch.writer is not None
        assert orch.search_tool is not None

    @patch("core.orchestrator.SearchCache")
    @patch("core.orchestrator.get_settings")
    def test_model_override(self, mock_get_settings, mock_cache):
        """CLI model override should replace the settings model."""
        settings = mock_settings()
        mock_get_settings.return_value = settings

        Orchestrator(model_override="gpt-3.5-turbo")

        assert settings.llm_model == "gpt-3.5-turbo"

    @patch("core.orchestrator.SearchCache")
    @patch("core.orchestrator.get_settings")
    def test_run_returns_summary_dict(self, mock_get_settings, mock_cache):
        """run() should return a dict with expected keys."""
        mock_get_settings.return_value = mock_settings()

        orch = Orchestrator()

        # Mock all agents to do nothing
        orch.planner.plan = MagicMock(return_value=["query 1", "query 2"])
        orch.analyst.analyze = MagicMock(return_value=[])
        orch.writer.write = MagicMock(return_value="Test report")

        # Mock the session's sub_queries to be set by planner
        with patch.object(orch, "_run_researchers", new_callable=AsyncMock):
            result = orch.run("test query", stream=False)

        assert isinstance(result, dict)
        assert "query" in result
        assert "success" in result
        assert "total_cost" in result
        assert "duration" in result
        assert "sub_queries" in result
        assert "num_sources" in result

    @patch("core.orchestrator.SearchCache")
    @patch("core.orchestrator.get_settings")
    def test_handles_planner_failure(self, mock_get_settings, mock_cache):
        """Pipeline should handle Planner failure gracefully."""
        mock_get_settings.return_value = mock_settings()

        orch = Orchestrator()
        orch.planner.plan = MagicMock(side_effect=Exception("API quota exceeded"))

        result = orch.run("test query", stream=False)

        assert result["success"] is False
        assert "query" in result

    @patch("core.orchestrator.SearchCache")
    @patch("core.orchestrator.get_settings")
    def test_run_populates_logs(self, mock_get_settings, mock_cache):
        """run() should populate the session audit log."""
        mock_get_settings.return_value = mock_settings()

        orch = Orchestrator()
        orch.planner.plan = MagicMock(side_effect=Exception("test error"))

        result = orch.run("test query", stream=False)

        assert len(result["logs"]) > 0
        log_text = " ".join(result["logs"])
        assert "Session started" in log_text
