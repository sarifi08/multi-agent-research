"""
Writer Agent — composes the final research report with streaming.

Two modes:
    write()        → returns full report at once (for APIs)
    write_stream() → streams tokens as they're generated (for CLI/UI)

Temperature: 0.7 — creative enough for natural language, not random.
"""
from typing import List, Generator
from loguru import logger
from openai import OpenAI

from core.session import ResearchSession, AgentStatus
from monitoring.tracker import MonitoringTracker


class WriterAgent:

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 2000
    ):
        self.client    = OpenAI(api_key=api_key)
        self.model     = model
        self.max_tokens = max_tokens
        self.temperature = 0.7  # creative but not random — we agreed on this!

        self.system_prompt = """You are an expert research writer.

Transform research findings into a clear, engaging report.

Rules:
- Write in natural, flowing prose (no bullet point dumps)
- Organize by themes, not by source
- Cite sources inline like this: (Source: url)
- Open with a 2-3 sentence summary of the key finding
- End with a "Key Takeaways" section (3-4 points max)
- Be concise — quality over quantity
- If given no findings, clearly tell the user nothing was found and why
"""

    def write(
        self,
        session: ResearchSession,
        tracker: MonitoringTracker
    ) -> str:
        """
        Write the full report at once.
        Reads session.findings, writes to session.report.
        """
        session.set_agent_status("writer", AgentStatus.RUNNING)
        tracker.start("writer")

        if not session.findings:
            report = self._empty_report(session.query)
            session.report   = report
            session.success  = False
            session.set_agent_status("writer", AgentStatus.DONE)
            tracker.end("writer")
            return report

        session.log(f"Writer: composing report from {len(session.findings)} findings")

        findings_text = self._format_findings(session.findings)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Research question: {session.query}\n\n"
                        f"Findings:\n{findings_text}\n\n"
                        f"Write the research report."
                    )
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        report = response.choices[0].message.content.strip()

        # Write to shared session
        session.report  = report
        session.success = True
        session.set_agent_status("writer", AgentStatus.DONE)
        session.log(f"Writer: report complete ({len(report)} chars)")

        tracker.end(
            "writer",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            llm_calls=1
        )

        return report

    def write_stream(
        self,
        session: ResearchSession,
        tracker: MonitoringTracker
    ) -> Generator[str, None, None]:
        """
        Stream the report token by token — user sees output as it's written.

        Usage:
            for token in writer.write_stream(session, tracker):
                print(token, end="", flush=True)

        Why this matters:
            Without streaming, user stares at blank screen for 20+ seconds.
            With streaming, they see words appearing immediately.
            Same total time, but perceived experience is dramatically better.
        """
        session.set_agent_status("writer", AgentStatus.RUNNING)
        tracker.start("writer")

        if not session.findings:
            report = self._empty_report(session.query)
            session.report  = report
            session.success = False
            session.set_agent_status("writer", AgentStatus.DONE)
            tracker.end("writer")
            yield report
            return

        findings_text = self._format_findings(session.findings)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Research question: {session.query}\n\n"
                        f"Findings:\n{findings_text}\n\n"
                        f"Write the research report."
                    )
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True   # ← this is the key difference
        )

        full_report = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_report += token
                yield token   # send each token to caller immediately

        # Write complete report to session after streaming finishes
        session.report  = full_report
        session.success = True
        session.set_agent_status("writer", AgentStatus.DONE)
        session.log(f"Writer: streamed report ({len(full_report)} chars)")

        # Note: streaming doesn't return token counts, so we estimate
        estimated_output = len(full_report.split()) * 1.3
        tracker.end(
            "writer",
            input_tokens=500,                    # rough estimate for input
            output_tokens=int(estimated_output),
            llm_calls=1
        )

    def _format_findings(self, findings: List[dict]) -> str:
        """Format findings dict list into readable text block for LLM."""
        parts = []
        for i, f in enumerate(findings, 1):
            parts.append(
                f"Finding {i}:\n"
                f"Title: {f['title']}\n"
                f"URL: {f['url']}\n"
                f"Summary: {f['summary']}\n"
                f"Why relevant: {f['why_relevant']}\n"
            )
        return "\n---\n".join(parts)

    def _empty_report(self, query: str) -> str:
        """Honest message when nothing was found — never hallucinate."""
        return (
            f"No relevant information was found for: '{query}'.\n\n"
            f"Possible reasons:\n"
            f"- The topic is very recent or niche\n"
            f"- Search terms may need rephrasing\n"
            f"- Information may be behind paywalls\n\n"
            f"Try rephrasing your question with more specific terms."
        )
