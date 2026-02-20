# Contributing to Multi-Agent Research System

Thank you for considering contributing! This project benefits from community input — whether it's a bug fix, new feature, or documentation improvement.

## 🚀 Quick Start for Contributors

```bash
# 1. Fork the repo on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/multi-agent-research.git
cd multi-agent-research

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install with dev dependencies
pip install -r requirements.txt
pip install pytest-cov

# 4. Set up environment
cp .env.example .env
# Add your API keys to .env

# 5. Run tests to make sure everything works
pytest tests/ -v
```

## 📋 Development Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** — keep commits focused and descriptive

3. **Run tests** before pushing:
   ```bash
   pytest tests/ -v --cov=agents --cov=core --cov=tools --cov=monitoring
   ```

4. **Push and open a PR** against `main`

## 🧪 Testing Guidelines

- All tests use **mocked API calls** — no real API keys needed to run tests
- Add tests for any new functionality
- Tests live in `tests/` and follow the pattern `test_<module>.py`
- Use `pytest` fixtures and `unittest.mock` for mocking

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_writer.py -v

# Run with coverage report
pytest tests/ -v --cov=agents --cov=core --cov=tools --cov=monitoring --cov-report=term-missing
```

## 📁 Project Structure

```
agents/     → AI agents (Planner, Researcher, Analyst, Writer)
core/       → Orchestrator + shared session state
tools/      → Web search + caching
config/     → Settings management
monitoring/ → Cost & timing tracking
tests/      → Unit tests (mocked APIs)
```

## 🎯 Areas for Contribution

- **New agents** — add specialist agents (fact-checker, summarizer, etc.)
- **Multi-provider support** — add Anthropic/Claude alongside OpenAI
- **Export formats** — PDF, HTML, DOCX export options
- **Better caching** — Redis-backed cache for production use
- **Rate limiting** — smarter rate limit handling for APIs
- **Documentation** — improve docstrings, add architecture diagrams

## 📝 Code Style

- **Docstrings** on all public methods (explain *why*, not just *what*)
- **Type hints** on all function signatures
- **Descriptive variable names** — no single-letter variables
- Keep functions focused — if it's doing two things, split it

## ⚠️ Important Notes

- **Never commit API keys** — they go in `.env` which is in `.gitignore`
- **Tests should not call real APIs** — always mock external calls
- **Keep `requirements.txt` updated** if you add dependencies

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
