"""
Tests for WriterAgent.

What we test:
    1. Writer produces a report from findings
    2. Writer writes to session correctly
    3. Writer handles empty findings gracefully
    4. Writer streaming yields tokens
    5. Session logs are populated after writing
"""
import pytest
from unittest.mock import MagicMock, patch
from agents.writer import WriterAgent
from core.session import ResearchSession, AgentStatus
from monitoring.tracker import MonitoringTracker


def make_session(findings=None):
    session = ResearchSession(query="AI trends in healthcare")
    if findings:
        session.findings = findings
    return session


def make_tracker(session):
    return MonitoringTracker(session, model="gpt-4o")


def make_findings():
    """Realistic findings that Analyst would produce."""
    return [
        {
            "title": "AI Diagnostics Revolution",
            "url": "https://example.com/1",
            "summary": "AI is transforming medical diagnostics with 95% accuracy.",
            "relevance_score": 0.9,
            "why_relevant": "Directly addresses AI in healthcare diagnostics.",
        },
        {
            "title": "ML Drug Discovery Breakthroughs",
            "url": "https://example.com/2",
            "summary": "Machine learning cuts drug discovery time by 60%.",
            "relevance_score": 0.85,
            "why_relevant": "Covers ML applications in pharmaceutical research.",
        },
    ]


def mock_openai_response(content: str, input_tokens=200, output_tokens=500):
    """Helper: fake an OpenAI API response."""
    response = MagicMock()
    response.choices[0].message.content = content
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens
    return response


def mock_streaming_response(tokens: list):
    """Helper: fake a streaming OpenAI API response."""
    chunks = []
    for token in tokens:
        chunk = MagicMock()
        chunk.choices[0].delta.content = token
        chunks.append(chunk)
    # Final chunk with None content
    final = MagicMock()
    final.choices[0].delta.content = None
    chunks.append(final)
    return chunks


class TestWriterAgent:

    @patch("agents.writer.OpenAI")
    def test_produces_report(self, mock_openai_class):
        """Writer should return a non-empty report string."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response(
            "This is a comprehensive research report about AI in healthcare."
        )

        session = make_session(make_findings())
        tracker = make_tracker(session)
        writer = WriterAgent(api_key="fake-key")

        report = writer.write(session, tracker)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "research report" in report.lower()

    @patch("agents.writer.OpenAI")
    def test_writes_to_session(self, mock_openai_class):
        """Writer should write the report to the shared session."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response(
            "Final research report content."
        )

        session = make_session(make_findings())
        tracker = make_tracker(session)
        writer = WriterAgent(api_key="fake-key")

        writer.write(session, tracker)

        assert session.report == "Final research report content."
        assert session.success is True
        assert session.agent_statuses["writer"] == AgentStatus.DONE

    @patch("agents.writer.OpenAI")
    def test_handles_empty_findings(self, mock_openai_class):
        """Writer should return graceful message when no findings exist."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        session = make_session(findings=[])
        tracker = make_tracker(session)
        writer = WriterAgent(api_key="fake-key")

        report = writer.write(session, tracker)

        assert "no relevant information" in report.lower()
        assert session.success is False
        # LLM should NOT be called when there's nothing to write about
        mock_client.chat.completions.create.assert_not_called()

    @patch("agents.writer.OpenAI")
    def test_streaming_yields_tokens(self, mock_openai_class):
        """Writer streaming should yield individual tokens."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        tokens = ["This ", "is ", "a ", "streamed ", "report."]
        mock_client.chat.completions.create.return_value = mock_streaming_response(tokens)

        session = make_session(make_findings())
        tracker = make_tracker(session)
        writer = WriterAgent(api_key="fake-key")

        received = list(writer.write_stream(session, tracker))

        assert len(received) == len(tokens)
        assert "".join(received) == "This is a streamed report."

    @patch("agents.writer.OpenAI")
    def test_session_logs_populated(self, mock_openai_class):
        """Session audit log should have writer entries."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response(
            "Report content."
        )

        session = make_session(make_findings())
        tracker = make_tracker(session)
        writer = WriterAgent(api_key="fake-key")

        writer.write(session, tracker)

        log_text = " ".join(session.logs)
        assert "WRITER" in log_text
        assert "report complete" in log_text.lower() or "done" in log_text.lower()

    @patch("agents.writer.OpenAI")
    def test_tracks_cost(self, mock_openai_class):
        """Writer should track token usage and cost."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response(
            "Report.", input_tokens=300, output_tokens=800
        )

        session = make_session(make_findings())
        tracker = make_tracker(session)
        writer = WriterAgent(api_key="fake-key")

        writer.write(session, tracker)

        metrics = session.metrics["writer"]
        assert metrics.input_tokens == 300
        assert metrics.output_tokens == 800
        assert metrics.cost_usd > 0
