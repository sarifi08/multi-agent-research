"""
Demo — run this to see the full multi-agent system in action.

Usage:
    python example.py "What are the latest breakthroughs in AI agents?"
    python example.py --query "AI trends in healthcare" --model gpt-3.5-turbo
    python example.py --no-stream --export report.md

Setup:
    1. pip install -r requirements.txt
    2. cp .env.example .env  →  add your API keys
    3. python example.py
"""
import argparse
import sys

from core.orchestrator import Orchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🔍 Multi-Agent Research System — AI-powered deep research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example.py "What are the latest AI agent breakthroughs?"
  python example.py --query "quantum computing 2024" --model gpt-3.5-turbo
  python example.py "AI in healthcare" --no-stream --export report.md
        """,
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Research query (can also use --query)",
    )
    parser.add_argument(
        "--query", "-q",
        dest="query_flag",
        default=None,
        help="Research query (alternative to positional argument)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="OpenAI model to use (default: from .env or gpt-4o)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (get full report at once)",
    )
    parser.add_argument(
        "--export", "-e",
        default=None,
        metavar="FILE",
        help="Export report to a Markdown file (e.g. report.md)",
    )

    args = parser.parse_args()

    # Resolve query: positional > --query flag > interactive prompt
    if args.query:
        pass
    elif args.query_flag:
        args.query = args.query_flag
    else:
        args.query = input("\n🔍 Enter your research query: ").strip()
        if not args.query:
            print("❌ No query provided. Exiting.")
            sys.exit(1)

    return args


def export_report(result: dict, filepath: str):
    """Export the research report to a Markdown file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"**Query:** {result['query']}\n\n")
        f.write(f"---\n\n")

        if result.get("report"):
            f.write(result["report"])
        else:
            f.write("*No report was generated.*\n")

        f.write(f"\n\n---\n\n")
        f.write(f"## Metadata\n\n")
        f.write(f"- **Sources:** {result['num_sources']}\n")
        f.write(f"- **Cost:** {result['total_cost']}\n")
        f.write(f"- **Duration:** {result['duration']}\n")
        f.write(f"- **Sub-queries:**\n")
        for q in result.get("sub_queries", []):
            f.write(f"  - {q}\n")


def main():
    args = parse_args()

    # Override model if specified via CLI
    orchestrator = Orchestrator(model_override=args.model)

    stream = not args.no_stream

    print(f"\n🔍 Query: {args.query}\n")

    result = orchestrator.run(args.query, stream=stream)

    print(f"\n📋 Sub-queries used:")
    for q in result["sub_queries"]:
        print(f"  • {q}")

    print(f"\n🔗 Sources: {result['num_sources']}")
    print(f"💰 Total cost: {result['total_cost']}")
    print(f"⏱  Duration: {result['duration']}")

    # Export report if requested
    if args.export:
        export_report(result, args.export)
        print(f"\n📄 Report exported to: {args.export}")


if __name__ == "__main__":
    main()
