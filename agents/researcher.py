"""
Researcher Agent — truly async web search with retry logic.
Now uses proper async/await instead of thread pool workaround.
"""
from typing import List
from loguru import logger
from openai import OpenAI

from core.session import ResearchSession, AgentStatus
from monitoring.tracker import MonitoringTracker
from tools.web_search import WebSearchTool, SearchResult


class ResearcherAgent:

    def __init__(
        self,
        api_key: str,
        search_tool: WebSearchTool,
        model: str = "gpt-4o",
        max_retries: int = 2
    ):
        self.client = OpenAI(api_key=api_key)
        self.search_tool = search_tool
        self.model = model
        self.max_retries = max_retries

    async def research_async(
        self,
        query: str,
        session: ResearchSession,
        tracker: MonitoringTracker
    ) -> dict:
        """
        Async search with retry logic.
        Writes raw results to session.raw_results.

        Flow:
            Search → useful? → yes: return results
                             → no:  ask Planner to rephrase → retry
                                    still no: return empty (Analyst handles it)
        """
        current_query = query
        attempts = 0
        total_input_tokens = 0
        total_output_tokens = 0

        while attempts <= self.max_retries:
            attempts += 1
            session.log(f"Researcher attempt {attempts}: '{current_query}'")

            # True async search — doesn't block other researchers
            results = await self.search_tool.search_async(current_query)

            if self.search_tool.is_useful(results):
                session.log(f"Researcher: good results found for '{current_query}'")

                output = {
                    "query":    current_query,
                    "results":  results,
                    "success":  True,
                    "attempts": attempts
                }

                # Append to shared session (thread-safe append)
                session.raw_results.append(output)

                tracker.end(
                    "researcher",
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    llm_calls=attempts - 1  # minus 1 because first attempt needs no LLM
                )
                return output

            # Results not useful — should we retry?
            if attempts <= self.max_retries:
                session.log(f"Researcher: poor results, asking Planner to rephrase")
                current_query, tokens = self._ask_planner_to_rephrase(
                    failed_query=current_query,
                    original_query=session.query  # always anchored to user's original ask
                )
                total_input_tokens  += tokens["input"]
                total_output_tokens += tokens["output"]

        # All retries exhausted
        session.log(f"Researcher: all retries exhausted for '{query}'")
        output = {
            "query":    query,
            "results":  [],
            "success":  False,
            "attempts": attempts
        }
        session.raw_results.append(output)
        return output

    def _ask_planner_to_rephrase(
        self,
        failed_query: str,
        original_query: str
    ) -> tuple[str, dict]:
        """
        Ask LLM (acting as Planner) to rephrase a failed query.
        Returns (new_query, token_usage).
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a research planning expert.
A search query returned poor results. Rephrase it to try a different angle.
Return ONLY the new query string, nothing else."""
                },
                {
                    "role": "user",
                    "content": (
                        f"Original research goal: {original_query}\n"
                        f"Failed query: {failed_query}\n"
                        f"Suggest a better search query."
                    )
                }
            ],
            temperature=0.4,
            max_tokens=50
        )

        new_query = response.choices[0].message.content.strip().strip('"')
        tokens = {
            "input":  response.usage.prompt_tokens,
            "output": response.usage.completion_tokens
        }

        logger.info(f"Rephrased to: '{new_query}'")
        return new_query, tokens
