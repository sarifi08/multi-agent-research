"""
Analyst Agent — filters and ranks all research results.
Reads from session.raw_results, writes to session.findings.
"""
from typing import List
from dataclasses import dataclass
from loguru import logger
from openai import OpenAI

from core.session import ResearchSession, AgentStatus
from monitoring.tracker import MonitoringTracker
from tools.web_search import SearchResult


@dataclass
class AnalystFinding:
    title: str
    url: str
    summary: str
    relevance_score: float
    why_relevant: str   # Analyst explains its reasoning — useful for debugging


class AnalystAgent:

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = 0.2  # strict and consistent

    def analyze(
        self,
        session: ResearchSession,
        tracker: MonitoringTracker
    ) -> List[AnalystFinding]:
        """
        Filter all raw results by relevance to the original query.
        Reads session.raw_results and session.query.
        Writes approved findings to session.findings.
        """
        session.set_agent_status("analyst", AgentStatus.RUNNING)
        tracker.start("analyst")

        # Step 1: Collect all results, log failed searches
        all_results = self._collect_results(session)

        session.log(f"Analyst: reviewing {len(all_results)} total results")

        if not all_results:
            session.log("Analyst: zero results to analyze — pipeline ending early")
            session.set_agent_status("analyst", AgentStatus.FAILED)
            tracker.end("analyst")
            return []

        # Step 2: Quick score filter (no LLM needed — saves cost)
        pre_filtered = [r for r in all_results if r.relevance_score >= 0.5]

        if not pre_filtered:
            session.log("Analyst: lowering score threshold to 0.3")
            pre_filtered = [r for r in all_results if r.relevance_score >= 0.3]

        session.log(f"Analyst: {len(pre_filtered)} passed score filter")

        # Step 3: LLM judges relevance to ORIGINAL query (catches topic drift)
        findings, total_input, total_output = self._judge_relevance(
            query=session.query,    # always the ORIGINAL user query
            results=pre_filtered
        )

        # Step 4: Sort by relevance
        findings.sort(key=lambda x: x.relevance_score, reverse=True)

        # Write to shared session
        session.findings = [
            {
                "title":           f.title,
                "url":             f.url,
                "summary":         f.summary,
                "relevance_score": f.relevance_score,
                "why_relevant":    f.why_relevant
            }
            for f in findings
        ]
        session.sources = [f.url for f in findings]

        session.set_agent_status("analyst", AgentStatus.DONE)
        session.log(f"Analyst: approved {len(findings)} findings")

        tracker.end(
            "analyst",
            input_tokens=total_input,
            output_tokens=total_output,
            llm_calls=len(pre_filtered)
        )

        return findings

    def _collect_results(self, session: ResearchSession) -> List[SearchResult]:
        """Pull results from all Researcher outputs in session."""
        all_results = []
        for output in session.raw_results:
            if not output["success"]:
                session.log(
                    f"Analyst: skipping failed search '{output['query']}' "
                    f"({output['attempts']} attempts)"
                )
                continue
            all_results.extend(output["results"])
        return all_results

    def _judge_relevance(
        self,
        query: str,
        results: List[SearchResult]
    ) -> tuple[List[AnalystFinding], int, int]:
        """LLM judges each result against the original query."""
        findings = []
        total_input  = 0
        total_output = 0

        for result in results:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict research analyst. "
                            "Only approve results truly relevant to the research question."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f'Research question: "{query}"\n\n'
                            f"Result title: {result.title}\n"
                            f"Result summary: {result.summary}\n\n"
                            f"Answer in this exact format:\n"
                            f"RELEVANT: yes/no\n"
                            f"SCORE: 0.0-1.0\n"
                            f"REASON: one sentence"
                        )
                    }
                ],
                temperature=self.temperature,
                max_tokens=80
            )

            total_input  += response.usage.prompt_tokens
            total_output += response.usage.completion_tokens

            judgment = response.choices[0].message.content.strip()
            parsed   = self._parse_judgment(judgment)

            if parsed["relevant"]:
                findings.append(AnalystFinding(
                    title=result.title,
                    url=result.url,
                    summary=result.summary,
                    relevance_score=parsed["score"],
                    why_relevant=parsed["reason"]
                ))

        return findings, total_input, total_output

    def _parse_judgment(self, judgment: str) -> dict:
        """Safely parse structured LLM output."""
        lines  = judgment.lower().split("\n")
        result = {"relevant": False, "score": 0.0, "reason": ""}

        for line in lines:
            if line.startswith("relevant:"):
                result["relevant"] = "yes" in line
            elif line.startswith("score:"):
                try:
                    result["score"] = float(line.split(":")[1].strip())
                except ValueError:
                    result["score"] = 0.5
            elif line.startswith("reason:"):
                result["reason"] = line.split(":", 1)[1].strip()

        return result
