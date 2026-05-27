"""
page_audit.py
-------------
Audit Log & Governance Dashboard
Demonstrates accountability, traceability, and human-in-the-loop governance.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys, os

# =========================================================================
# 【核心修复】：用原生 Session State 数据存取函数替代隔离的外部 audit_log 模块
# =========================================================================
def get_recent_evaluations(limit=50):
    return st.session_state.audit_entries[-limit:]

def get_override_statistics():
    entries = st.session_state.audit_entries
    total = len(entries)
    if total == 0:
        return {
            "total_evaluations": 0, "total_overrides": 0, "override_rate_pct": 0,
            "approved_overrides": 0, "rejected_overrides": 0
        }
    
    overrides = [e for e in entries if e["recruiter_override"]]
    total_overrides = len(overrides)
    
    # 统计审核人员的最终干预倾向
    approved = sum(1 for e in overrides if "Approve" in e["recruiter_decision"])
    rejected = sum(1 for e in overrides if "Reject" in e["recruiter_decision"])
    
    return {
        "total_evaluations": total,
        "total_overrides": total_overrides,
        "override_rate_pct": round((total_overrides / total) * 100, 1),
        "approved_overrides": approved,
        "rejected_overrides": rejected
    }

def export_log_as_dataframe():
    if not st.session_state.audit_entries:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.audit_entries)

def clear_log():
    st.session_state.audit_entries = []


def render():
    st.title("📋 Audit Log & Governance")

    st.markdown("""
    This page demonstrates the **accountability and traceability** principles of responsible AI governance.
    Every AI recommendation, recruiter override, and model decision is logged here —
    creating a transparent record that can be reviewed, audited, and challenged.

    > *"An AI system that cannot explain its past decisions cannot be held accountable for them."*
    """)

    # ----------------------------------------------------------------
    # SECTION 1: Why Audit Logs Matter
    # ----------------------------------------------------------------
    with st.expander("📖 Why Does This Matter Ethically?", expanded=False):
        st.markdown("""
        **The problem with opaque AI systems:**

        When a candidate is rejected by an AI-assisted system and receives no explanation,
        they cannot challenge that decision. When a company is audited and cannot produce
        records of how its AI made decisions, it cannot demonstrate compliance or good faith.

        Audit logs solve this by creating:
        - A permanent, timestamped record of every AI recommendation
        - Documentation of recruiter overrides — showing humans are genuinely in the loop
        - Evidence for fairness auditing over time
        - A paper trail for legal compliance (EU AI Act requires this for high-risk AI)

        **The Amazon case and transparency:**
        Amazon's recruitment AI existed for years without any public transparency.
        When Reuters exposed it in 2018, Amazon could not produce meaningful documentation
        of how the system had been operating or how many candidates had been affected.
        This audit system is designed to prevent exactly that — every decision is traceable.
        """)

    st.divider()

    # ----------------------------------------------------------------
    # SECTION 2: Governance Statistics
    # ----------------------------------------------------------------
    st.markdown("### 📊 Governance Overview")

    stats = get_override_statistics()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total AI Evaluations", stats["total_evaluations"])
    with col2:
        st.metric("Recruiter Overrides", stats["total_overrides"])
    with col3:
        st.metric("Override Rate", f"{stats['override_rate_pct']}%",
                 help="How often recruiters chose to override the AI recommendation")
    with col4:
        st.metric("Approved Overrides", stats["approved_overrides"],
                 help="AI said reject, recruiter approved")
    with col5:
        st.metric("Rejected Overrides", stats["rejected_overrides"],
                 help="AI said recommend, recruiter rejected")

    if stats["total_evaluations"] > 0:
        override_rate = stats["override_rate_pct"]
        if override_rate > 30:
            st.warning(f"""
            ⚠️ **High Override Rate ({override_rate}%):** Recruiters are frequently
            disagreeing with the AI. This may indicate the model is not well-calibrated
            to recruiter judgment and should be re-evaluated.
            """)
        elif override_rate > 10:
            st.info(f"""
            ℹ️ **Moderate Override Rate ({override_rate}%):** Recruiters are occasionally
            overriding the AI. This is healthy — it shows humans are exercising oversight
            rather than rubber-stamping AI recommendations.
            """)
        else:
            st.success(f"""
            ✅ **Low Override Rate ({override_rate}%):** Recruiters are largely accepting
            AI recommendations. Monitor this over time — very low override rates may
            indicate "automation bias" (humans deferring to AI without critical review).
            """)

    st.divider()

    # ----------------------------------------------------------------
    # SECTION 3: Recent Evaluation Log
    # ----------------------------------------------------------------
    st.markdown("### 📜 Recent Evaluation Log")

    recent = get_recent_evaluations(50)

    if not recent:
        st.info("""
        No evaluations logged yet. Evaluate candidates on the
        **Candidate Evaluation** page to populate the audit log.
        """)
    else:
        log_df = export_log_as_dataframe()
        if not log_df.empty:
            # Format for display
            display_df = log_df[[
                "evaluation_id", "timestamp", "candidate_name",
                "biased_score", "fair_score",
                "biased_recommendation", "fair_recommendation",
                "recruiter_override", "recruiter_decision", "recruiter_notes"
            ]].copy()

            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            display_df["recruiter_override"] = display_df["recruiter_override"].fillna(False).map(
                {True: "✅ Yes", False: "—"}
            )
            display_df["recruiter_decision"] = display_df["recruiter_decision"].fillna("—")

            display_df.columns = [
                "Evaluation ID", "Timestamp", "Candidate",
                "Biased Score", "Fair Score",
                "Biased Rec.", "Fair Rec.",
                "Override", "Decision", "Notes"
            ]

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Export
            csv = log_df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Audit Log (CSV)",
                data=csv,
                file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    st.divider()

    # ----------------------------------------------------------------
    # SECTION 4: Human-in-the-Loop Design Principles
    # ----------------------------------------------------------------
    st.markdown("### 🧑‍💼 Human-in-the-Loop Design: Why It Matters")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What human-in-the-loop means here:**

        - The AI provides a recommendation **score**, not a hiring decision
        - Every output includes a mandatory "Human review required" flag
        - Recruiters can **approve, reject, or defer** any candidate regardless of AI score
        - All overrides are **logged with timestamps and notes**
        - Override patterns become data for improving future model versions

        **The governance loop:**
        1. AI evaluates candidate → logs recommendation
        2. Recruiter reviews → may override
        3. Override is logged → becomes training signal
        4. Model is periodically re-evaluated using override history
        5. Persistent bias patterns in overrides trigger mandatory re-audit
        """)

    with col2:
        st.markdown("""
        **Why full automation would be dangerous:**

        Automation bias is the documented tendency of humans to defer to
        automated systems even when those systems are wrong or biased.
        If recruiters cannot meaningfully override AI recommendations —
        because overrides are discouraged, logged in ways that create
        accountability for the human but not the AI, or simply never reviewed —
        then "human oversight" becomes theater rather than governance.

        **Responsible AI governance requires:**
        - Override options that are genuinely easy to use
        - No penalty for overriding the AI's recommendation
        - Regular review of override patterns for systemic bias
        - Periodic third-party audits of the AI system itself
        - Clear escalation paths when the AI produces concerning outputs
        """)

    # ----------------------------------------------------------------
    # SECTION 5: Responsible AI Procurement Checklist
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### ✅ Responsible AI Procurement Checklist for HR Departments")
    st.markdown("""*What should organizations ask before deploying any AI hiring tool?*""")

    checklist_items = [
        ("📊", "Bias Audit", "Has the system been independently audited for gender, race, age, and disability bias before deployment? (Required under NYC Local Law 144)"),
        ("🔍", "Explainability", "Can the system explain each recommendation in plain language? Can candidates understand why they received a particular score?"),
        ("👤", "Human Oversight", "Is there a genuine, easy-to-use mechanism for recruiters to override AI recommendations without penalty?"),
        ("📋", "Audit Trail", "Does the system maintain tamper-evident logs of all recommendations, overrides, and model decisions?"),
        ("⚖️", "Fairness Metrics", "Which fairness metrics does the system optimize for? What tradeoffs were made, and were they made transparently?"),
        ("🔄", "Re-auditing", "How often is the system re-audited? Who conducts the audit — internal teams or independent third parties?"),
        ("📢", "Candidate Disclosure", "Are candidates informed that AI is involved in screening their application? Do they have the right to request human review?"),
        ("🛑", "Data Governance", "What historical data was the system trained on? Was that data itself audited for bias before training?"),
        ("🌐", "Scope Limitation", "Is the AI used only for initial screening, or does it influence final hiring decisions? The narrower the scope, the lower the risk."),
        ("📉", "Sunset Criteria", "Under what conditions will the organization stop using this system? What metrics trigger mandatory review or discontinuation?"),
    ]

    for icon, title, description in checklist_items:
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown(f"<div style='font-size:1.5rem; padding-top:4px;'>{icon}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{title}:** {description}")

    st.markdown("""
    <div style='background-color: #f1f8e9; border-left: 6px solid #558b2f; padding: 12px; border-radius: 4px; margin-top: 10px;'>
        <strong>📌 Note:</strong> Amazon's AI recruitment system failed on nearly every item
        in this checklist. It was not independently audited, did not explain its decisions,
        had no meaningful override mechanism for affected candidates, and was eventually
        discontinued without public disclosure. This checklist is designed to operationalize
        the lessons of that case.
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # SECTION 6: Clear Log (Demo Reset)
    # ----------------------------------------------------------------
    st.divider()
    with st.expander("⚠️ Admin: Reset Audit Log (Demo Only)", expanded=False):
        st.warning("This will permanently delete all logged evaluations. For demo resets only.")
        if st.button("🗑️ Clear All Log Entries", type="secondary"):
            clear_log()
            st.success("Audit log cleared.")
            st.rerun()
