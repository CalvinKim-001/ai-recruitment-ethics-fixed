"""
page_home.py
------------
Home page: Ethical Background & Project Overview

This page sets the ethical context BEFORE the user interacts with any
candidate data. On purpose — the ethics frame should come first.
"""

import streamlit as st


def render():
    # Hero header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
                padding: 48px 40px; border-radius: 16px; margin-bottom: 32px; color: white;'>
        <div style='font-size: 0.85rem; letter-spacing: 0.15em; opacity: 0.7; margin-bottom: 8px;'>
            BUSINESS ETHICS & CSR PROJECT
        </div>
        <h1 style='font-size: 2.4rem; font-weight: 700; margin: 0 0 12px 0; color: white;'>
            When Algorithms Discriminate
        </h1>
        <p style='font-size: 1.1rem; opacity: 0.85; max-width: 700px; line-height: 1.7; margin: 0;'>
            A fairness-aware AI recruitment demonstration inspired by the Amazon AI
            Recruitment Bias Case — exploring how historical data encodes discrimination
            and how responsible AI governance can reduce measurable harm.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # The Amazon Case
    st.markdown("## 📰 The Case That Started It All")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        In **2014**, Amazon began developing an AI-powered recruitment tool intended to
        automate the screening of resumes for technical positions. The system was trained on
        approximately **10 years of historical hiring data** — a decade of real hiring decisions
        made by real human recruiters.

        The problem: Amazon's technology workforce was — and the broader tech industry
        remains — **predominantly male**. The historical hiring data reflected that imbalance.

        The AI did exactly what it was designed to do: it found patterns in the data that
        predicted hiring success. But the patterns it found were **male patterns** — because
        historically, the candidates who had been hired were mostly men.

        By **2015**, engineers discovered the system was actively penalizing resumes that
        contained indicators associated with women:
        """)

        st.error("""
        🚨 **What the AI penalized:**
        - Resumes containing the word **"women's"** — e.g., *"captain of women's chess club"*
        - Graduates of **all-women's colleges** (e.g., Smith, Wellesley, Mount Holyoke)
        - Phrases associated with female-dominated professional networks
        """)

        st.markdown("""
        Amazon attempted to correct the bias, but engineers could not guarantee the model would
        stop identifying **proxy variables** — indirect signals correlated with gender.

        In **2018**, Amazon quietly shut down the project. This became public when Reuters
        reported the story, making it one of the most widely cited examples of **algorithmic
        discrimination** in employment history.

        Amazon confirmed the tool was never used for actual hiring decisions — but the
        system existed, grew more biased over time, and was abandoned rather than corrected.
        """)

    with col2:
        st.markdown("""
        <div style='background: #fff8e1; border: 1px solid #f9a825; border-radius: 12px;
                    padding: 20px; text-align: center;'>
            <div style='font-size: 2rem;'>📅</div>
            <div style='font-weight: 700; font-size: 1.1rem; color: #1a237e;'>Timeline</div>
            <hr style='margin: 8px 0; border-color: #f9a825;'>
            <div style='font-size: 0.85rem; line-height: 2;'>
                <b>2014</b><br>System built<br><br>
                <b>2015</b><br>Bias discovered<br><br>
                <b>2017</b><br>Project halted<br><br>
                <b>2018</b><br>Reuters exposes it<br><br>
                <b>Today</b><br>Still widely cited
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Core Concepts
    st.markdown("## 🧠 Three Concepts That Explain Everything")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='background: white; border-radius: 12px; padding: 24px;
                    border: 1px solid #e8eaf6; height: 280px;'>
            <div style='font-size: 2rem; margin-bottom: 8px;'>📊</div>
            <div style='font-weight: 700; font-size: 1.05rem; color: #1a237e; margin-bottom: 8px;'>
                Historical Bias in Data
            </div>
            <div style='font-size: 0.88rem; line-height: 1.7; color: #424242;'>
                If historical hiring data reflects discrimination — even discrimination
                made by humans, unconsciously — the AI learns those discriminatory
                patterns as if they were fact. The bias is <em>not programmed in</em>;
                it is <em>learned from history</em>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: white; border-radius: 12px; padding: 24px;
                    border: 1px solid #e8eaf6; height: 280px;'>
            <div style='font-size: 2rem; margin-bottom: 8px;'>🎭</div>
            <div style='font-weight: 700; font-size: 1.05rem; color: #1a237e; margin-bottom: 8px;'>
                Proxy Discrimination
            </div>
            <div style='font-size: 0.88rem; line-height: 1.7; color: #424242;'>
                Removing the word "gender" from the data is <strong>not enough</strong>.
                AI systems learn to infer gender through indirect signals: names,
                university names, club activities, employment gaps, even writing style.
                These are called <em>proxy variables</em>. This is exactly what
                Amazon's system exploited.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background: white; border-radius: 12px; padding: 24px;
                    border: 1px solid #e8eaf6; height: 280px;'>
            <div style='font-size: 2rem; margin-bottom: 8px;'>⚖️</div>
            <div style='font-weight: 700; font-size: 1.05rem; color: #1a237e; margin-bottom: 8px;'>
                Fairness Tradeoffs
            </div>
            <div style='font-size: 0.88rem; line-height: 1.7; color: #424242;'>
                There is no single definition of fairness. Different metrics reflect
                different ethical theories — and they can <em>mathematically conflict</em>
                with each other. Reducing bias sometimes means accepting lower prediction
                accuracy. These are real tradeoffs, not technical failures.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # This Project
    st.markdown("## 🎯 What This Dashboard Demonstrates")

    st.markdown("""
    This platform is built for a **Business Ethics & CSR** university course. It is not
    a production hiring system. It is an educational demonstration of how AI systems
    can inherit discrimination — and how responsible design can reduce it.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What we build:**
        - A **biased baseline model** trained on historically imbalanced synthetic data
        - A **fairness-aware model** trained with bias mitigation techniques
        - **10 matched resume pairs** — identical qualifications, only gendered signals differ
        - A complete **fairness audit** with multiple metrics and ethical interpretations
        - **SHAP explainability** so every recommendation can be questioned
        - A **recruiter override system** demonstrating human-in-the-loop governance
        """)

    with col2:
        st.markdown("""
        **What we claim:**
        - The fairness-aware model is **fairness-improved**, not perfectly fair
        - Fairness tradeoffs exist and we acknowledge them honestly
        - No AI system should make final hiring decisions without human review
        - Auditing must be **continuous**, not a one-time checkbox
        - This system covers gender; race, age, and other dimensions deserve equal attention
        """)

    # Ethical Framework
    st.divider()
    st.markdown("## 🏛️ The Ethical Framework")

    st.markdown("""
    Different fairness metrics in this dashboard reflect different **ethical theories**:
    """)

    ethics_data = {
        "Metric": [
            "Demographic Parity",
            "Equal Opportunity",
            "Equalized Odds",
            "Disparate Impact Ratio",
        ],
        "Ethical Theory": [
            "Egalitarianism",
            "Meritocracy",
            "Procedural Justice",
            "Anti-Discrimination Law",
        ],
        "What It Asks": [
            "Are selection rates equal across groups?",
            "Are qualified candidates equally recognized?",
            "Are errors equally distributed across groups?",
            "Is one group disadvantaged relative to another?",
        ],
        "Legal Standard": [
            "No direct legal standard",
            "EEOC equal employment principle",
            "Title VII disparate treatment",
            "EEOC 80% / 4/5 rule",
        ],
    }

    import pandas as pd
    st.dataframe(pd.DataFrame(ethics_data), use_container_width=True, hide_index=True)

    # Important disclaimer
    st.markdown("""
    <div class='ethical-note'>
        <strong>⚠️ Important:</strong> All data in this dashboard is <strong>synthetic</strong>
        — generated algorithmically for educational purposes. No real candidates, real resumes,
        or real hiring decisions are involved. The Amazon case is described using publicly
        reported facts from the 2018 Reuters investigation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
    **Navigate using the sidebar** to explore candidate evaluation, fairness auditing,
    model comparison, and the governance audit log. We recommend starting with
    **Fairness Audit Dashboard** to initialize the models before using other pages.
    """)
