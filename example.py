"""
Demo — run this to see the full multi-agent system in action.

Setup:
    1. pip install -r requirements.txt
    2. cp .env.example .env  →  add your API keys
    3. python example.py
"""
from core.orchestrator import Orchestrator


def main():
    orchestrator = Orchestrator()

    query = "What are the latest breakthroughs in AI agents in 2024?"

    print(f"\n🔍 Query: {query}\n")

    # stream=True means you see the report being written word by word
    result = orchestrator.run(query, stream=True)

    print(f"\n📋 Sub-queries used:")
    for q in result["sub_queries"]:
        print(f"  • {q}")

    print(f"\n🔗 Sources: {result['num_sources']}")
    print(f"💰 Total cost: {result['total_cost']}")
    print(f"⏱  Duration: {result['duration']}")


if __name__ == "__main__":
    main()
