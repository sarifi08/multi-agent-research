"""
Orchestrator — coordinates all agents around the shared ResearchSession.

What changed from v1:
    - All agents now share a ResearchSession (the whiteboard)
    - MonitoringTracker records cost + time per agent
    - Researchers are truly async (not thread pool workaround)
    - Writer supports streaming
    - Full audit log of everything that happened
"""
import asyncio
from datetime import datetime
from typing import Generator, Optional
from loguru import logger

from config.settings import get_settings
from core.session import ResearchSession, AgentStatus
from monitoring.tracker import MonitoringTracker
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.analyst import AnalystAgent
from agents.writer import WriterAgent
from tools.web_search import WebSearchTool
from tools.cache import SearchCache


class Orchestrator:

    def __init__(self, model_override: str = None):
        self.settings = get_settings()

        # CLI can override the model from .env
        if model_override:
            self.settings.llm_model = model_override

        # Search cache — avoids paying for repeated queries
        self.cache = SearchCache(cache_dir=".cache", ttl_hours=24)

        # Shared tool — one instance, used by all Researcher agents
        self.search_tool = WebSearchTool(
            api_key=self.settings.tavily_api_key,
            max_results=self.settings.max_search_results,
            cache=self.cache
        )

        # Initialize all agents
        self.planner    = PlannerAgent(self.settings.openai_api_key, self.settings.llm_model)
        self.analyst    = AnalystAgent(self.settings.openai_api_key, self.settings.llm_model)
        self.writer     = WriterAgent(
            self.settings.openai_api_key,
            self.settings.llm_model,
            self.settings.max_output_tokens
        )

        logger.info("✅ Orchestrator initialized")

    def run(self, user_query: str, stream: bool = False) -> dict:
        """
        Main entry point.

        Args:
            user_query: what the user wants researched
            stream:     if True, prints report tokens as they arrive

        Returns:
            Full session summary dict
        """
        # Create the shared whiteboard for this research session
        session = ResearchSession(query=user_query)
        tracker = MonitoringTracker(session, model=self.settings.llm_model)

        session.log(f"Session started: '{user_query}'")
        logger.info(f"\n{'='*55}\nResearching: '{user_query}'\n{'='*55}")

        try:
            # ── 1. PLANNER ────────────────────────────────────────
            logger.info("\n[1/4] 🧠 Planner running...")
            self.planner.plan(session, tracker)

            # ── 2. RESEARCHERS (PARALLEL) ─────────────────────────
            logger.info(f"\n[2/4] 🔍 Researchers running ({len(session.sub_queries)} parallel)...")
            session.set_agent_status("researcher", AgentStatus.RUNNING)
            tracker.start("researcher")

            asyncio.run(self._run_researchers(session, tracker))

            session.set_agent_status("researcher", AgentStatus.DONE)

            # ── 3. ANALYST ────────────────────────────────────────
            logger.info("\n[3/4] 📊 Analyst running...")
            self.analyst.analyze(session, tracker)

            # ── 4. WRITER ─────────────────────────────────────────
            logger.info("\n[4/4] ✍️  Writer running...")

            if stream:
                print("\n── RESEARCH REPORT " + "─" * 36 + "\n")
                for token in self.writer.write_stream(session, tracker):
                    print(token, end="", flush=True)
                print("\n" + "─" * 55)
            else:
                self.writer.write(session, tracker)

        except Exception as e:
            session.error   = str(e)
            session.success = False
            session.log(f"Pipeline error: {e}")
            logger.error(f"Pipeline failed: {e}")

        finally:
            session.completed_at = datetime.now()

        # Print monitoring summary
        if self.settings.enable_tracking:
            tracker.print_summary()

        return session.summary()

    async def _run_researchers(
        self,
        session: ResearchSession,
        tracker: MonitoringTracker
    ):
        """
        Run all Researcher agents truly in parallel.
        Respects max_parallel_searches limit to avoid rate limiting.
        """
        max_parallel = self.settings.max_parallel_searches
        queries      = session.sub_queries

        for i in range(0, len(queries), max_parallel):
            batch = queries[i:i + max_parallel]
            session.log(f"Researcher: starting batch of {len(batch)} parallel searches")

            # One Researcher instance per query — they all run at the same time
            tasks = [
                ResearcherAgent(
                    api_key=self.settings.openai_api_key,
                    search_tool=self.search_tool,
                    model=self.settings.llm_model,
                    max_retries=self.settings.max_retries
                ).research_async(query, session, tracker)
                for query in batch
            ]

            # All tasks in this batch run simultaneously
            await asyncio.gather(*tasks)
