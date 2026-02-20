"""
Streamlit Web UI — visual interface for the Multi-Agent Research System.

Run:
    streamlit run app.py

Features:
    - Type a query and watch agents work in real-time
    - See each agent's status, timing, and cost
    - Streaming report output
    - Export report as Markdown
    - Model selection from sidebar
"""
import streamlit as st
import time
from datetime import datetime

from core.orchestrator import Orchestrator
from core.session import AgentStatus


# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research",
    page_icon="🔍",
    layout="wide",
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "result": None,
        "running": False,
        "history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar():
    """Sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Settings")

        model = st.selectbox(
            "LLM Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4o = best quality, gpt-3.5-turbo = cheapest",
        )

        stream = st.checkbox("Stream output", value=True, help="Show report token by token")

        st.divider()
        st.header("📊 Cost Reference")
        st.markdown("""
        | Model | ~Cost/Query |
        |-------|-----------|
        | gpt-4o | $0.03–0.05 |
        | gpt-4-turbo | $0.05–0.08 |
        | gpt-3.5-turbo | $0.002–0.005 |
        """)

        st.divider()

        if st.session_state.history:
            st.header("📜 History")
            for i, h in enumerate(reversed(st.session_state.history[-5:])):
                with st.expander(f"🔍 {h['query'][:40]}...", expanded=False):
                    st.write(f"**Status:** {'✅' if h['success'] else '❌'}")
                    st.write(f"**Cost:** {h['total_cost']}")
                    st.write(f"**Duration:** {h['duration']}")
                    st.write(f"**Sources:** {h['num_sources']}")

    return model, stream


def render_agent_status(placeholder, agent_name: str, status: str, elapsed: float = 0):
    """Render a single agent status indicator."""
    icons = {
        "pending":  "⏳",
        "running":  "🔄",
        "done":     "✅",
        "failed":   "❌",
    }
    icon = icons.get(status, "⏳")
    return f"{icon} **{agent_name.title()}** — {status} ({elapsed:.1f}s)"


def run_research(query: str, model: str, stream: bool):
    """Execute the research pipeline with live UI updates."""
    st.session_state.running = True

    # Agent status display
    status_container = st.container()
    with status_container:
        st.subheader("🤖 Agent Pipeline")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            planner_status = st.empty()
            planner_status.info("⏳ **Planner** — waiting")
        with col2:
            researcher_status = st.empty()
            researcher_status.info("⏳ **Researcher** — waiting")
        with col3:
            analyst_status = st.empty()
            analyst_status.info("⏳ **Analyst** — waiting")
        with col4:
            writer_status = st.empty()
            writer_status.info("⏳ **Writer** — waiting")

    # Report area
    st.subheader("📄 Research Report")
    report_area = st.empty()

    progress_bar = st.progress(0, text="Starting research pipeline...")

    # Run orchestrator
    try:
        orchestrator = Orchestrator(model_override=model)

        # Update: Planner running
        planner_status.warning("🔄 **Planner** — running...")
        progress_bar.progress(10, text="🧠 Planner breaking down query...")

        start = time.time()
        result = orchestrator.run(query, stream=False)

        # Update all statuses to done
        elapsed = time.time() - start
        planner_status.success("✅ **Planner** — done")
        researcher_status.success("✅ **Researcher** — done")
        analyst_status.success("✅ **Analyst** — done")
        writer_status.success("✅ **Writer** — done")
        progress_bar.progress(100, text="✅ Research complete!")

        # Display report
        if result.get("report"):
            if stream:
                # Simulate streaming for visual effect
                full_report = result["report"]
                displayed = ""
                for i in range(0, len(full_report), 3):
                    displayed = full_report[: i + 3]
                    report_area.markdown(displayed + "▌")
                    time.sleep(0.01)
                report_area.markdown(full_report)
            else:
                report_area.markdown(result["report"])
        else:
            report_area.warning("No report was generated. Check your API keys and try again.")

        # Store result
        st.session_state.result = result
        st.session_state.history.append(result)

    except Exception as e:
        progress_bar.progress(100, text="❌ Pipeline failed")
        st.error(f"Pipeline error: {str(e)}")
        st.session_state.result = None

    finally:
        st.session_state.running = False


def render_results(result: dict):
    """Display research results in a nice layout."""
    if not result:
        return

    st.divider()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sources", result["num_sources"])
    col2.metric("Cost", result["total_cost"])
    col3.metric("Duration", result["duration"])
    col4.metric("Status", "✅ Success" if result["success"] else "❌ Failed")

    # Sub-queries
    with st.expander("📋 Sub-queries used", expanded=False):
        for q in result.get("sub_queries", []):
            st.write(f"• {q}")

    # Agent breakdown
    with st.expander("📊 Agent Breakdown", expanded=False):
        cols = st.columns(4)
        for i, (name, cost) in enumerate(result.get("agent_costs", {}).items()):
            time_taken = result.get("agent_times", {}).get(name, "0.0s")
            cols[i].metric(
                name.title(),
                cost,
                delta=time_taken,
            )

    # Audit log
    with st.expander("📜 Audit Log", expanded=False):
        for entry in result.get("logs", []):
            st.text(entry)

    # Export button
    if result.get("report"):
        report_md = (
            f"# Research Report\n\n"
            f"**Query:** {result['query']}\n\n"
            f"---\n\n"
            f"{result['report']}\n\n"
            f"---\n\n"
            f"*Sources: {result['num_sources']} | "
            f"Cost: {result['total_cost']} | "
            f"Duration: {result['duration']}*\n"
        )
        st.download_button(
            label="📥 Download Report as Markdown",
            data=report_md,
            file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )


def main():
    init_session_state()

    # Header
    st.title("🔍 Multi-Agent Research System")
    st.caption(
        "AI-powered deep research — Planner → Researcher → Analyst → Writer"
    )

    # Sidebar
    model, stream = render_sidebar()

    # Query input
    with st.form("research_form"):
        query = st.text_input(
            "What do you want to research?",
            placeholder="e.g., What are the latest breakthroughs in AI agents?",
        )
        submitted = st.form_submit_button(
            "🚀 Start Research",
            disabled=st.session_state.running,
            use_container_width=True,
        )

    if submitted and query:
        run_research(query, model, stream)

    # Show results if we have them
    if st.session_state.result:
        render_results(st.session_state.result)


if __name__ == "__main__":
    main()
