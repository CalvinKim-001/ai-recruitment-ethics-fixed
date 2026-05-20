"""
main.py
------
Main entry point for the AI Recruitment Ethics Dashboard.
"""

import streamlit as st

# ----------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AI Recruitment Ethics Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------
# Initialize session state
# ----------------------------------------------------------------
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "biased_model" not in st.session_state:
    st.session_state.biased_model = None
if "fair_model" not in st.session_state:
    st.session_state.fair_model = None
if "fair_scaler" not in st.session_state:
    st.session_state.fair_scaler = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "audit_entries" not in st.session_state:
    st.session_state.audit_entries = []
if "last_evaluation" not in st.session_state:
    st.session_state.last_evaluation = None

# ----------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------
st.sidebar.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <div style='font-size: 2.5rem;'>⚖️</div>
    <div style='font-size: 1.1rem; font-weight: 700; letter-spacing: 0.05em;'>
        AI RECRUITMENT<br>ETHICS DASHBOARD
    </div>
    <div style='font-size: 0.7rem; opacity: 0.6; margin-top: 4px;'>
        Business Ethics & CSR Project
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

pages = {
    " Home & Ethical Background": "home",
    " Candidate Evaluation": "evaluation",
    " Fairness Audit Dashboard": "fairness",
    " Model Comparison": "comparison",
    " Audit Log & Governance": "audit",
}

selected_page = st.sidebar.radio(
    "Navigate to:",
    list(pages.keys()),
    label_visibility="collapsed",
    key="main_nav"
)

st.sidebar.markdown("---")
if st.session_state.models_trained:
    st.sidebar.success("✅ Models trained and ready")
else:
    st.sidebar.warning("⚙️ Models not yet trained")
    st.sidebar.caption("Visit the Fairness Audit page to initialize models.")

st.sidebar.markdown("""
---
<div style='font-size: 0.72rem; opacity: 0.55; text-align: center; padding-top: 8px;'>
    Inspired by Amazon AI Recruitment<br>Bias Case (Reuters, 2018)<br><br>
    ⚠️ For educational purposes only.<br>
    All data is synthetic.
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# Page routing
# ----------------------------------------------------------------
page_key = pages[selected_page]

if page_key == "home":
    from pages import page_home
    page_home.render()

elif page_key == "evaluation":
    from pages import page_evaluation
    page_evaluation.render()

elif page_key == "fairness":
    from pages import page_fairness
    page_fairness.render()

elif page_key == "comparison":
    from pages import page_comparison
    page_comparison.render()

elif page_key == "audit":
    from pages import page_audit
    page_audit.render()
