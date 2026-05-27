"""
page_evaluation.py
------------------
Interactive Workspace integrating Single Candidate Auditing & Resume Pairs Experimentation.
Restores missing Candidate Name, Human-in-the-Loop Overrides, and detailed textual explanations.
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
    # TAB 1: 实时单人求职者特征组合评估（完美找回丢失的姓名、Override及反馈功能）
    # =========================================================================
    with tab1:
        st.subheader("Evaluate Custom Candidate Specifications")
        st.markdown("Simulate a custom profile to observe real-time risk disparity metrics between systems.")
        
        with st.form("interactive_evaluator_form"):
            # 【功能找回】：候选人姓名输入框
            candidate_name = st.text_input("Candidate Name", "Alex Morgan")
            st.divider()
            
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
            
            st.divider()
            # 【功能找回】：人机协作（Human-in-the-Loop）治理改写项与合规日志反馈框
            st.markdown("##### 👥 Human-in-the-Loop Governance Override")
            hr_override = st.selectbox(
                "Recruiter Override AI Algorithmic Recommendation?", 
                ["No Override - Follow AI Advice", "Force Approve / Recommend for Interview", "Force Reject / Decline Profile"]
            )
            hr_feedback = st.text_area(
                "HR Governance Feedback & Auditing Notes", 
                placeholder="Provide qualitative justification for manual override, bias mitigation tracking or compliance record logs..."
            )

            submitted = st.form_submit_button("Run Algorithmic Auditing & Log Entry", type="primary")

        if submitted:
            # Map categories back into standard feature models values
            edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
            tier_map = {"Startup": 1, "Mid-size": 2, "Large Corp": 3, "Top Tech (FAANG)": 4}
            sig_val = 0.15 if signal_type == "Features Gendered Phrases (e.g., Women's Club)" else 0.85

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
            
            # 【核心修复】：将判定逻辑和命名与 page_audit.py 严密对齐
            is_override = hr_override != "No Override - Follow AI Advice"
            eval_id = f"EVAL-{len(st.session_state.audit_entries) + 1:03d}"
            
            audit_entry = {
                "evaluation_id": eval_id,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "candidate_name": candidate_name,
                "biased_score": float(evaluation_res['biased_score']),
                "fair_score": float(evaluation_res['fair_score']),
                "biased_recommendation": evaluation_res.get('biased_recommendation', "Recommend" if evaluation_res['biased_score'] >= 0.5 else "Reject"),
                "fair_recommendation": evaluation_res.get('fair_recommendation', "Recommend" if evaluation_res['fair_score'] >= 0.5 else "Reject"),
                "recruiter_override": is_override,
                "recruiter_decision": hr_override if is_override else "—",
                "recruiter_notes": hr_feedback if hr_feedback.strip() else "No comment supplied by auditor."
            }
            st.session_state.audit_entries.append(audit_entry)
            
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
                    
            st.success("📝 Candidate evaluated and governance audit log entry saved successfully! View the records on the Governance page.")

    # =========================================================================
    # TAB 2: 10组简历配对实验看板
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
        
        fig, ax = plt.subplots(figsize=(12, 5))
        indices = np.arange(len(differentials))
        bar_width = 0.35

        b_data = differentials["biased_score_gap"] * 100
        f_data = differentials["fair_score_gap"] * 100

        bar1 = ax.bar(indices - bar_width/2, b_data, bar_width, label="Biased Model Gap", color="#D9534F")
        bar2 = ax.bar(indices + bar_width/2, f_data, bar_width, label="Fairness-Aware Model Gap", color="#5CB85C")

        v_max = max(b_data.max(), f_data.max(), 3.0)
        v_min = min(b_data.min(), f_data.min(), -3.0)
        ax.set_ylim(v_min * 1.4, v_max * 1.4)

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

        st.markdown("---")
        options = [f"Resume Pair {i+1}: {resume_pairs.RESUME_PAIRS[i]['scenario']}" for i in range(10)]
        p_sel = st.selectbox("Inspect Configuration Profile Parameters:", options)
        sel_idx = options.index(p_sel)
        
        r_pair = resume_pairs.RESUME_PAIRS[sel_idx]
        d_row = differentials.iloc[sel_idx]
        
        st.markdown("### 📝 Experiment Pair Qualitative Details")
        st.info(f"**Historical Context & Scenario Explanation:**\n\n{r_pair['narrative']}")
        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"#### 👨 Male Candidate Resume Profile")
            st.markdown(f"* **Candidate Name:** {r_pair['male']['name']}")
            st.markdown(f"* **Graduated University:** {r_pair['male']['university']}")
            st.markdown(f"* **Activity Phrasing (Standard Vector):** `{r_pair['male']['activity']}`")
            st.markdown("---")
            st.markdown("**📊 Model Evaluation Results:**")
            st.metric("Biased Baseline Model Score", f"{d_row['male_biased_score']*100:.1f}%")
            st.metric("Fairness-Aware Model Score", f"{d_row['male_fair_score']*100:.1f}%")
        with sc2:
            st.markdown(f"#### 👩 Female Candidate Resume Profile")
            st.markdown(f"* **Candidate Name:** {r_pair['female']['name']}")
            st.markdown(f"* **Graduated University:** {r_pair['female']['university']}")
            st.markdown(f"* **Activity Phrasing (Gender Vector):** `{r_pair['female']['activity']}`")
            st.markdown("---")
            st.markdown("**📊 Model Evaluation Results:**")
            st.metric("Biased Baseline Model Score", f"{d_row['female_biased_score']*100:.1f}%",
                      delta=f"{(d_row['female_biased_score'] - d_row['male_biased_score'])*100:.1f}% Bias Penalty" if d_row['biased_score_gap'] != 0 else None,
                      delta_color="inverse")
            st.metric("Fairness-Aware Model Score", f"{d_row['female_fair_score']*100:.1f}%",
                      delta=f"{(d_row['female_fair_score'] - d_row['male_fair_score'])*100:.1f}% Disparity" if d_row['fair_score_gap'] != 0 else None,
                      delta_color="off")
