"""
MonitoringTracker — records what happened during a research session.

Why separate from ResearchSession?
    Session = the data being passed between agents (the whiteboard)
    Tracker = the observer watching the session and recording metrics

    Keeping them separate means monitoring logic doesn't pollute
    the core data flow. You can disable monitoring without
    touching any agent code.
"""
from datetime import datetime
from loguru import logger
from core.session import ResearchSession, AgentMetrics


# Token costs per 1K — update these as OpenAI changes pricing
MODEL_PRICING = {
    "gpt-4o":        {"input": 0.005,  "output": 0.015},
    "gpt-4-turbo":   {"input": 0.01,   "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


class MonitoringTracker:
    """
    Wraps a ResearchSession and records timing + cost
    for every agent that runs.

    Usage:
        tracker = MonitoringTracker(session, model="gpt-4o")
        tracker.start("planner")
        # ... planner runs ...
        tracker.end("planner", input_tokens=150, output_tokens=80, llm_calls=1)
    """

    def __init__(self, session: ResearchSession, model: str = "gpt-4o"):
        self.session = session
        self.model = model
        self.pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])

    def start(self, agent_name: str):
        """Mark an agent as started — records start time."""
        metrics = self.session.metrics[agent_name]
        metrics.start_time = datetime.now()
        logger.info(f"⏱  {agent_name} started")

    def end(
        self,
        agent_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        llm_calls: int = 1
    ):
        """
        Mark an agent as finished.
        Records end time, tokens used, and calculates cost.
        """
        metrics = self.session.metrics[agent_name]
        metrics.end_time = datetime.now()
        metrics.input_tokens  += input_tokens
        metrics.output_tokens += output_tokens
        metrics.llm_calls     += llm_calls

        # Calculate cost for this agent
        cost = (
            (input_tokens  / 1000) * self.pricing["input"] +
            (output_tokens / 1000) * self.pricing["output"]
        )
        metrics.cost_usd += cost

        logger.info(
            f"✅ {agent_name} done | "
            f"{metrics.duration_seconds():.1f}s | "
            f"{input_tokens + output_tokens} tokens | "
            f"${cost:.4f}"
        )

    def print_summary(self):
        """Print a full cost + timing breakdown to the console."""
        summary = self.session.summary()

        print("\n" + "=" * 55)
        print("📊 RESEARCH SESSION SUMMARY")
        print("=" * 55)
        print(f"Query:    {summary['query']}")
        print(f"Success:  {'✅' if summary['success'] else '❌'}")
        print(f"Duration: {summary['duration']}")
        print(f"Cost:     {summary['total_cost']}")
        print(f"Sources:  {summary['num_sources']}")

        print("\n── Agent Breakdown ──────────────────────────────────")
        print(f"{'Agent':<15} {'Time':>8} {'Cost':>10} {'Tokens':>10}")
        print("-" * 50)

        for name, m in self.session.metrics.items():
            tokens = m.input_tokens + m.output_tokens
            print(
                f"{name:<15} "
                f"{m.duration_seconds():>7.1f}s "
                f"${m.cost_usd:>9.4f} "
                f"{tokens:>10,}"
            )

        print("-" * 50)
        print(f"{'TOTAL':<15} {self.session.total_duration():>7.1f}s {self.session.total_cost():>10.4f}")

        print("\n── Audit Log ────────────────────────────────────────")
        for entry in summary["logs"]:
            print(f"  {entry}")

        print("=" * 55 + "\n")
