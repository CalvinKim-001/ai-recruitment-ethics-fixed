"""
page_evaluation.py
------------------
Interactive Workspace integrating Single Candidate Auditing & Resume Pairs Experimentation.
Fixes the ValueError string splitting bug using robust indexing lookup.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import models
import resume_pairs

def render():
    st.title("⚖️ Candidate Evaluation Workspace")
    st.markdown("Explore individual screening assessments or evaluate paired compliance experiments.")

    # Model training validation gate
    if not st.session_state.models_trained:
        st.warning("⚙️ AI Models are not initialized yet. Please navigate to the **Fairness Audit Dashboard** to trigger training routines first.")
        return

    # 使用 Streamlit Tabs 让原有单人评估功能与 Resume Pairs 实验完美并存
    tab1, tab2 = st.tabs(["🎯 Interactive Candidate Evaluator", "📊 Gender Signals Pair Experiment"])

    # =========================================================================
    # TAB 1: 实时单人求职者特征组合评估（找回的核心功能）
    # =========================================================================
    with tab1:
        st.subheader("Evaluate Custom Candidate Specifications")
        st.markdown("Simulate a custom profile to observe real-time risk disparity metrics between systems.")
        
        with st.form("interactive_evaluator_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                exp = st.slider("Years of Industry Experience", 0, 10, 3)
                edu = st.selectbox("Completed Education Level", ["High School", "Bachelor's", "Master's", "PhD"], index=1)
                tier = st.selectbox("Previous Company Profile Tier", ["Startup", "Mid-size", "Large Corp", "Top Tech (FAANG)"], index=1)
            with c2:
                prog = st.slider("Programming Mastery Score", 0, 100, 75)
                lead = st.slider("Leadership Aptitude Vector", 0, 100, 60)
                comm = st.slider("Communication Articulation Index", 0, 100, 65)
            with c3:
                proj = st.slider("Completed Open Source Projects", 0, 5, 2)
                inter = st.slider("Live Technical Interview Evaluation", 0, 100, 70)
                signal_type = st.radio("Resume Phrasing Signal Pattern", ["Standard Profile", "Features Gendered Phrases (e.g., Women's Club)"])

            # Map categories back into standard feature models values
            edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
            tier_map = {"Startup": 1, "Mid-size": 2, "Large Corp": 3, "Top Tech (FAANG)": 4}
            sig_val = 0.15 if signal_type == "Features Gendered Phrases (e.g., Women's Club)" else 0.85

            submitted = st.form_submit_button("Run Algorithmic Auditing", type="primary")

        if submitted:
            candidate_payload = {
                "years_experience": exp,
                "education_level": edu_map[edu],
                "programming_skill": prog,
                "leadership_score": lead,
                "communication_score": comm,
                "company_tier": tier_map[tier],
                "project_experience": proj,
                "interview_score": inter,
                "resume_gender_signal": sig_val
            }
            
            evaluation_res = models.score_candidate(
                candidate_payload,
                st.session_state.biased_model,
                st.session_state.fair_model,
                st.session_state.fair_scaler
            )
            
            st.markdown("### 🔍 Evaluation Metrics Analysis")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Biased System Recommendation Score", f"{evaluation_res['biased_score']*100:.1f}%")
                if evaluation_res['biased_score'] >= 0.5:
                    st.success(f"Outcome: {evaluation_res['biased_recommendation']}")
                else:
                    st.error(f"Outcome: {evaluation_res['biased_recommendation']}")
            with res_col2:
                st.metric("Fairness-Improved Recommendation Score", f"{evaluation_res['fair_score']*100:.1f}%")
                if evaluation_res['fair_score'] >= 0.5:
                    st.success(f"Outcome: {evaluation_res['fair_recommendation']}")
                else:
                    st.error(f"Outcome: {evaluation_res['fair_recommendation']}")

    # =========================================================================
    # TAB 2: 10组简历配对实验看板（彻底解决 Matplotlib 通天长线拉伸 Bug）
    # =========================================================================
    with tab2:
        st.subheader("Gender Signals Audit Dashboard (10 Matched Profiles)")
        st.markdown("Identical academic and work backgrounds. The only variance lies within proxy linguistic flags.")

        pairs_df = resume_pairs.get_all_pairs_dataframe()
        differentials = models.score_resume_pairs(
            pairs_df,
            st.session_state.biased_model,
            st.session_state.fair_model,
            st.session_state.fair_scaler
        )

        avg_b_gap = differentials["biased_score_gap"].mean() * 100
        avg_f_gap = differentials["fair_score_gap"].mean() * 100
        avg_red = differentials["gap_reduction"].mean() * 100
        imp_p = sum(differentials["gap_reduction"] > 0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Gap — Biased Model", f"{avg_b_gap:+.2f}pp")
        m2.metric("Avg Gap — Fair Model", f"{avg_f_gap:+.2f}pp")
        m3.metric("Average Gap Mitigation", f"{avg_red:.2f}pp")
        m4.metric("Mitigated Pairs", f"{imp_p}/10")

        st.markdown("#### Score Disparity Visualizer")
        
        # 建立画幅控制结构
        fig, ax = plt.subplots(figsize=(12, 5))
        indices = np.arange(len(differentials))
        bar_width = 0.35

        # 换算至百分点值 (pp)
        b_data = differentials["biased_score_gap"] * 100
        f_data = differentials["fair_score_gap"] * 100

        bar1 = ax.bar(indices - bar_width/2, b_data, bar_width, label="Biased Model Gap", color="#D9534F")
        bar2 = ax.bar(indices + bar_width/2, f_data, bar_width, label="Fairness-Aware Model Gap", color="#5CB85C")

        # 动态视窗控制，绝不产生过大硬编码越界
        v_max = max(b_data.max(), f_data.max(), 3.0)
        v_min = min(b_data.min(), f_data.min(), -3.0)
        ax.set_ylim(v_min * 1.4, v_max * 1.4)

        # 改用安全稳定的数字内嵌标签标注，替换不稳定的 annotation
        ax.bar_label(bar1, fmt='%.1f', padding=3, fontsize=8, color='#A33A37')
        ax.bar_label(bar2, fmt='%.1f', padding=3, fontsize=8, color='#3B823B')

        ax.set_ylabel("Score Disparity: Male - Female (pp)", fontsize=10)
        ax.set_xticks(indices)
        ax.set_xticklabels([f"Pair {i+1}" for i in range(10)], rotation=0)
        ax.axhline(0, color="#888888", linestyle="--", linewidth=1)
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        
        st.pyplot(fig)

        # =========================================================================
        # 【核心 Bug 修复位置】：用 options.index 替代脆弱的文本切割转换
        # =========================================================================
        st.markdown("---")
        options = [f"Resume Pair {i+1}: {resume_pairs.RESUME_PAIRS[i]['scenario']}" for i in range(10)]
        p_sel = st.selectbox("Inspect Configuration Profile Parameters:", options)
        sel_idx = options.index(p_sel)  # 100% 安全的索引定位法
