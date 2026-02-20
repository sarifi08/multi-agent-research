# Multi-Agent Research System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991.svg)](https://openai.com/)
[![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-orange.svg)](https://tavily.com/)

> A multi-agent AI system that breaks down complex research queries into parallel sub-searches, filters results by relevance, and produces a polished research report — all orchestrated through a shared-state architecture with full cost & timing monitoring.

```
User Query
    │
    ▼
┌─────────┐     ┌──────────────┐     ┌──────────┐     ┌────────┐
│ Planner │ ──▶ │ Researchers  │ ──▶ │ Analyst  │ ──▶ │ Writer │
│         │     │ (parallel)   │     │          │     │        │
│ Breaks  │     │ Async web    │     │ Filters  │     │ Writes │
│ query   │     │ search +     │     │ & ranks  │     │ report │
│ into    │     │ retry logic  │     │ results  │     │ with   │
│ sub-    │     │              │     │          │     │ stream │
│ queries │     │              │     │          │     │        │
└─────────┘     └──────────────┘     └──────────┘     └────────┘
                                                          │
                                                          ▼
                                                   Research Report
```

## ✨ Features

- **Multi-agent orchestration** — four specialized AI agents (Planner → Researcher → Analyst → Writer) coordinated through a shared-state "whiteboard" pattern
- **True async parallelism** — researchers run genuinely concurrent web searches using `aiohttp`, not thread-pool workarounds
- **Smart retry logic** — failed searches are automatically rephrased by the Planner and retried, with configurable retry limits
- **Cost & performance monitoring** — every agent's token usage, cost, and timing is tracked and reported in a detailed breakdown
- **Streaming output** — the Writer streams the final report token-by-token for a responsive CLI experience
- **Interactive CLI** — full `argparse` interface with `--query`, `--model`, `--export`, and `--no-stream` flags
- **Streamlit Web UI** — visual interface to run research, watch agents work, and download reports
- **Search caching** — disk-backed cache with TTL expiry avoids burning API credits on repeated queries
- **Markdown export** — save reports as `.md` files from CLI or download from the web UI
- **Clean separation of concerns** — session (data), tracker (observability), and agents (logic) are fully decoupled

## 📁 Project Structure

```
multi-agent-research/
├── agents/
│   ├── planner.py          # Breaks user query into sub-searches
│   ├── researcher.py       # Async web search with retry logic
│   ├── analyst.py          # Filters & ranks results by relevance
│   └── writer.py           # Composes final report (streaming)
├── core/
│   ├── orchestrator.py     # Coordinates the 4-agent pipeline
│   └── session.py          # Shared state (the "whiteboard")
├── tools/
│   ├── web_search.py       # Async Tavily search wrapper (aiohttp)
│   └── cache.py            # Disk-backed search cache with TTL
├── config/
│   └── settings.py         # Pydantic settings from .env
├── monitoring/
│   └── tracker.py          # Cost & timing tracker per agent
├── tests/
│   ├── test_planner.py     # Unit tests with mocked OpenAI
│   ├── test_researcher.py  # Async tests with mocked search
│   ├── test_analyst.py     # Filter & ranking tests
│   ├── test_writer.py      # Report generation tests
│   ├── test_orchestrator.py # Pipeline integration tests
│   └── test_cache.py       # Cache TTL + persistence tests
├── .github/
│   ├── workflows/tests.yml # CI: pytest on Python 3.10-3.12
│   ├── ISSUE_TEMPLATE/     # Bug report & feature request templates
│   └── pull_request_template.md
├── app.py                  # Streamlit web UI
├── example.py              # CLI entry point
├── pyproject.toml          # Modern Python packaging
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
├── .env.example
└── README.md
```

## 🏗️ Architecture

### Shared-State "Whiteboard" Pattern

All agents read from and write to a single `ResearchSession` object instead of passing data like a relay race. This means every agent has full context at all times:

```python
@dataclass
class ResearchSession:
    query: str                          # User's original question
    sub_queries: List[str]              # Planner output
    raw_results: List[dict]             # Researcher output
    findings: List[dict]                # Analyst output
    report: str                         # Writer output
    agent_statuses: dict                # Who's doing what
    logs: List[str]                     # Full audit trail
```

### Agent Pipeline

| Step | Agent | Input | Output | LLM Calls |
|------|-------|-------|--------|-----------|
| 1 | **Planner** | User query | 3–4 targeted sub-queries | 1 |
| 2 | **Researchers** | Sub-queries (parallel) | Raw search results per query | 0–2 per retry |
| 3 | **Analyst** | All raw results | Filtered & ranked findings | 1 per result |
| 4 | **Writer** | Approved findings | Polished research report | 1 (streamed) |

### Monitoring

Every agent run is tracked with:
- ⏱ **Timing** — start/end timestamps and duration
- 🪙 **Token usage** — input and output tokens per agent
- 💰 **Cost** — calculated from model pricing (GPT-4o, GPT-4-turbo, GPT-3.5-turbo)
- 📋 **Audit log** — timestamped log of every action in the pipeline

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Tavily API key](https://tavily.com/) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/sarifi08/multi-agent-research.git
cd multi-agent-research

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys
```

### Configuration

Create a `.env` file (or copy from `.env.example`):

```env
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here
```

Optional settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o` | OpenAI model (`gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`) |
| `MAX_OUTPUT_TOKENS` | `2000` | Max tokens for Writer output |
| `MAX_RETRIES` | `2` | Researcher retry attempts before giving up |
| `MAX_PARALLEL_SEARCHES` | `3` | Concurrent search limit (rate limit protection) |
| `MAX_SEARCH_RESULTS` | `5` | Results per search query |
| `ENABLE_TRACKING` | `true` | Cost & timing monitoring |

### Usage

#### CLI (Interactive)

```bash
# Interactive prompt — just run it
python example.py

# Direct query
python example.py "What are the latest breakthroughs in AI agents?"

# With options
python example.py --query "AI in healthcare" --model gpt-3.5-turbo
python example.py "quantum computing 2024" --no-stream --export report.md
```

**CLI Options:**

| Flag | Description |
|------|-------------|
| `query` | Research query (positional or `--query`) |
| `--model`, `-m` | Override LLM model (default: from `.env`) |
| `--no-stream` | Get full report at once (no streaming) |
| `--export`, `-e` | Export report to Markdown file |

#### Web UI (Streamlit)

```bash
streamlit run app.py
```

The web UI provides:
- 🔍 Query input with real-time agent status
- 📊 Cost, timing, and source metrics
- 📥 One-click Markdown report download
- 📜 Search history in sidebar

**Example CLI output:**

```
🔍 Query: What are the latest breakthroughs in AI agents in 2024?

[1/4] 🧠 Planner running...
[2/4] 🔍 Researchers running (4 parallel)...
[3/4] 📊 Analyst running...
[4/4] ✍️  Writer running...

── RESEARCH REPORT ────────────────────────────

[Streamed report appears here token by token...]

═══════════════════════════════════════════════
📊 RESEARCH SESSION SUMMARY
═══════════════════════════════════════════════
Query:    What are the latest breakthroughs in AI agents in 2024?
Success:  ✅
Duration: 32.1s
Cost:     $0.0342
Sources:  8

── Agent Breakdown ─────────────────────────────
Agent               Time       Cost     Tokens
--------------------------------------------------
planner             2.1s $   0.0012        280
researcher          8.4s $   0.0000          0
analyst            12.3s $   0.0180      3,200
writer              9.3s $   0.0150      2,800
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_planner.py -v

# Run with coverage report
pytest tests/ -v --cov=agents --cov=core --cov=tools --cov=monitoring --cov-report=term-missing
```

**Test coverage includes:**

| Module | Test File | What's Tested |
|--------|-----------|---------------|
| Planner | `test_planner.py` | Query decomposition, bad LLM output handling |
| Researcher | `test_researcher.py` | Async search, retry logic, failure recovery |
| Analyst | `test_analyst.py` | Score filtering, relevance judgment, empty results |
| Writer | `test_writer.py` | Report generation, streaming, empty findings |
| Orchestrator | `test_orchestrator.py` | Full pipeline, model override, failure handling |
| Cache | `test_cache.py` | Hit/miss, TTL expiry, corruption recovery, persistence |

All tests use `unittest.mock` — **no API keys needed** to run tests.

## 🔧 How It Works (Detailed)

### 1. Planner Agent
Takes the user's broad research question and breaks it into 3–4 specific, non-overlapping search queries. Uses a low temperature (0.3) for consistency.

### 2. Researcher Agents (Parallel)
Each sub-query spawns an async researcher that:
- Searches the web via Tavily API using `aiohttp` (true async, not threaded)
- Checks if results are useful (average relevance ≥ 0.5)
- If poor results: asks the Planner LLM to rephrase and retries
- Respects `MAX_PARALLEL_SEARCHES` to avoid API rate limits

### 3. Analyst Agent
Reviews all raw results against the **original** user query (catches topic drift):
- Pre-filters by Tavily's relevance score (≥ 0.5, falls back to ≥ 0.3)
- LLM judges each result's relevance with reasoning
- Sorts findings by relevance score

### 4. Writer Agent
Composes the final report from approved findings:
- Writes in natural prose organized by themes (not bullet dumps)
- Cites sources inline
- Supports token-by-token streaming for responsive output
- Temperature 0.7 for creative but grounded writing

## 📊 Cost Estimates

| Model | Typical Query Cost | Speed |
|-------|-------------------|-------|
| `gpt-4o` | ~$0.03–0.05 | ~30s |
| `gpt-4-turbo` | ~$0.05–0.08 | ~40s |
| `gpt-3.5-turbo` | ~$0.002–0.005 | ~20s |

## 🛠️ Tech Stack

- **[OpenAI API](https://openai.com/)** — powers all four agents (planning, rephrasing, analysis, writing)
- **[Tavily API](https://tavily.com/)** — advanced web search with relevance scoring
- **[aiohttp](https://docs.aiohttp.org/)** — true async HTTP for parallel searches
- **[Streamlit](https://streamlit.io/)** — web UI for visual research interface
- **[Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)** — type-safe configuration from `.env`
- **[Loguru](https://github.com/Delgan/loguru)** — structured logging with emoji
- **[pytest](https://docs.pytest.org/)** — unit testing with async support and coverage

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
