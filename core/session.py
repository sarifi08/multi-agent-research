"""
ResearchSession — the shared whiteboard all agents read and write to.

Why this exists:
    Without shared state, agents pass data like a relay race.
    One agent hands results to the next, but earlier agents'
    context is lost. The Writer can't see what the Planner decided.

    With shared state, every agent sees the full picture at all times.

Design decision:
    We use a dataclass with explicit fields (not a dict) so that:
    - Every field is typed and documented
    - Agents can't accidentally write to wrong fields
    - IDE autocomplete works properly
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum


class AgentStatus(Enum):
    """Tracks what each agent is currently doing."""
    PENDING  = "pending"    # not started yet
    RUNNING  = "running"    # currently working
    DONE     = "done"       # finished successfully
    FAILED   = "failed"     # failed, even after retries


@dataclass
class AgentMetrics:
    """Cost + performance tracking for one agent run."""
    agent_name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0

    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def calculate_cost(self, model: str):
        """
        Calculate cost based on token usage.
        Prices per 1K tokens (GPT-4o as of 2024).
        """
        pricing = {
            "gpt-4o":         {"input": 0.005,  "output": 0.015},
            "gpt-4-turbo":    {"input": 0.01,   "output": 0.03},
            "gpt-3.5-turbo":  {"input": 0.0005, "output": 0.0015},
        }
        rates = pricing.get(model, {"input": 0.005, "output": 0.015})
        self.cost_usd = (
            (self.input_tokens  / 1000) * rates["input"] +
            (self.output_tokens / 1000) * rates["output"]
        )


@dataclass
class ResearchSession:
    """
    The shared whiteboard — every agent reads and writes here.

    Lifecycle:
        Created by Orchestrator → passed to each agent → returned to user
    """

    # ── What the user asked ──────────────────────────────────
    query: str = ""

    # ── Planner output ───────────────────────────────────────
    sub_queries: List[str] = field(default_factory=list)

    # ── Researcher output ────────────────────────────────────
    # Raw results per sub-query before Analyst filters them
    raw_results: List[dict] = field(default_factory=list)

    # ── Analyst output ───────────────────────────────────────
    findings: List[dict] = field(default_factory=list)

    # ── Writer output ────────────────────────────────────────
    report: str = ""
    sources: List[str] = field(default_factory=list)

    # ── Agent statuses (the "who is doing what" view) ────────
    agent_statuses: dict = field(default_factory=lambda: {
        "planner":    AgentStatus.PENDING,
        "researcher": AgentStatus.PENDING,
        "analyst":    AgentStatus.PENDING,
        "writer":     AgentStatus.PENDING,
    })

    # ── Step by step audit log ───────────────────────────────
    logs: List[str] = field(default_factory=list)

    # ── Monitoring metrics per agent ─────────────────────────
    metrics: dict = field(default_factory=lambda: {
        "planner":    AgentMetrics("planner"),
        "researcher": AgentMetrics("researcher"),
        "analyst":    AgentMetrics("analyst"),
        "writer":     AgentMetrics("writer"),
    })

    # ── Final status ─────────────────────────────────────────
    success: bool = False
    error: Optional[str] = None

    # ── Timestamps ───────────────────────────────────────────
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def log(self, message: str):
        """Add a timestamped entry to the audit log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)

    def set_agent_status(self, agent: str, status: AgentStatus):
        """Update an agent's status + log it."""
        self.agent_statuses[agent] = status
        self.log(f"{agent.upper()} → {status.value}")

    def total_cost(self) -> float:
        """Sum cost across all agents."""
        return sum(m.cost_usd for m in self.metrics.values())

    def total_duration(self) -> float:
        """Total wall-clock time for the full pipeline."""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return 0.0

    def summary(self) -> dict:
        """Clean summary for displaying to the user."""
        return {
            "query":        self.query,
            "success":      self.success,
            "sub_queries":  self.sub_queries,
            "num_sources":  len(self.sources),
            "total_cost":   f"${self.total_cost():.4f}",
            "duration":     f"{self.total_duration():.1f}s",
            "agent_costs":  {
                name: f"${m.cost_usd:.4f}"
                for name, m in self.metrics.items()
            },
            "agent_times":  {
                name: f"{m.duration_seconds():.1f}s"
                for name, m in self.metrics.items()
            },
            "logs":         self.logs
        }
