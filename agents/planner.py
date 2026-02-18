"""
Planner Agent — breaks user query into specific sub-searches.
Now writes results to shared ResearchSession.
"""
from typing import List
from loguru import logger
from openai import OpenAI

from core.session import ResearchSession, AgentStatus
from monitoring.tracker import MonitoringTracker


class PlannerAgent:

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = """You are a research planning expert.

Break the user's research question into 3-4 specific, non-overlapping search queries.

Rules:
- Each query must be specific and searchable (like a real Google search)
- Queries must not overlap — each covers a different angle
- Together they should give a complete picture of the topic
- Return ONLY a Python list of strings, nothing else

Example:
User: "Research AI trends in healthcare"
You: ["AI diagnostic tools in healthcare 2024",
      "machine learning medical imaging breakthroughs",
      "AI drug discovery latest research",
      "ethical challenges AI healthcare industry"]
"""

    def plan(
        self,
        session: ResearchSession,
        tracker: MonitoringTracker
    ) -> List[str]:
        """
        Break user query into sub-searches.
        Reads from session.query, writes to session.sub_queries.
        """
        session.set_agent_status("planner", AgentStatus.RUNNING)
        tracker.start("planner")
        session.log(f"Planner: breaking down '{session.query}'")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": session.query}
                ],
                temperature=0.3,
                max_tokens=300
            )

            # Track token usage for cost monitoring
            usage = response.usage
            tracker.end(
                "planner",
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                llm_calls=1
            )

            raw_output = response.choices[0].message.content.strip()
            sub_queries = self._parse_queries(raw_output)

            # Write to shared session
            session.sub_queries = sub_queries
            session.set_agent_status("planner", AgentStatus.DONE)

            for i, q in enumerate(sub_queries, 1):
                session.log(f"  Sub-query {i}: {q}")

            logger.info(f"Planner created {len(sub_queries)} sub-queries")
            return sub_queries

        except Exception as e:
            session.set_agent_status("planner", AgentStatus.FAILED)
            session.error = str(e)
            logger.error(f"Planner failed: {e}")
            raise

    def _parse_queries(self, raw_output: str) -> List[str]:
        """Safely parse LLM output into a list — never trust raw LLM output."""
        try:
            queries = eval(raw_output)
            if isinstance(queries, list):
                return [str(q) for q in queries]
        except Exception:
            pass

        # Fallback: split by newlines
        lines = [
            line.strip().strip('"-,[]')
            for line in raw_output.split("\n")
            if line.strip() and line.strip() not in ['[', ']']
        ]
        return [l for l in lines if len(l) > 5]
