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
import random
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
    # TAB 1: 实时单人求职者特征组合评估（随机生成、即刻审计、四键无缝留痕）
    # =========================================================================
    with tab1:
        st.subheader("Evaluate Custom Candidate Specifications")
        st.markdown("Simulate a custom profile to observe real-time risk disparity metrics or instantly randomize context attributes.")
        
        # ---------------------------------------------------------------------
        # 1. 初始化或动态管理要求特征的 Session State
        # ---------------------------------------------------------------------
        if "eval_name" not in st.session_state:
            st.session_state.eval_name = "Alex Morgan"
            st.session_state.eval_exp = 3
            st.session_state.eval_edu = "Bachelor's"
            st.session_state.eval_tier = "Mid-size"
            st.session_state.eval_prog = 75
            st.session_state.eval_lead = 60
            st.session_state.eval_comm = 65
            st.session_state.eval_proj = 2
            st.session_state.eval_inter = 70
            st.session_state.eval_signal = "Standard Profile"
            st.session_state.eval_notes = ""

        # ---------------------------------------------------------------------
        # 2. 【核心新增】：一键随机化生成候选人核心特征按钮（置于顶部）
        # ---------------------------------------------------------------------
        if st.button("🎲 Generate Random Candidate Profile", type="secondary", use_container_width=True):
            first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Jamie", "Skyler", "Robin", "Cameron"]
            last_names = ["Lee", "Morgan", "Smith", "Davis", "Kim", "Chen", "Patel", "Walsh", "Reed", "Johnson"]
            
            st.session_state.eval_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            st.session_state.eval_exp = random.randint(0, 10)
            st.session_state.eval_edu = random.choice(["High School", "Bachelor's", "Master's", "PhD"])
            st.session_state.eval_tier = random.choice(["Startup", "Mid-size", "Large Corp", "Top Tech (FAANG)"])
            st.session_state.eval_prog = random.randint(30, 100)
            st.session_state.eval_lead = random.randint(30, 100)
            st.session_state.eval_comm = random.randint(30, 100)
            st.session_state.eval_proj = random.randint(0, 5)
            st.session_state.eval_inter = random.randint(30, 100)
            st.session_state.eval_signal = random.choice(["Standard Profile", "Features Gendered Phrases (e.g., Women's Club)"])
            st.session_state.eval_notes = "" # 重置备注信息
            st.rerun()

        st.divider()

        # ---------------------------------------------------------------------
        # 3. 渲染候选人属性输入组件（通过 key 实现状态绑定与数据双向联动）
        # ---------------------------------------------------------------------
        candidate_name = st.text_input("Candidate Name", key="eval_name")
        st.divider()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            exp = st.slider("Years of Industry Experience", 0, 10, key="eval_exp")
            edu = st.selectbox("Completed Education Level", ["High School", "Bachelor's", "Master's", "PhD"], key="eval_edu")
            tier = st.selectbox("Previous Company Profile Tier", ["Startup", "Mid-size", "Large Corp", "Top Tech (FAANG)"], key="eval_tier")
        with c2:
            prog = st.slider("Programming Mastery Score", 0, 100, key="eval_prog")
            lead = st.slider("Leadership Aptitude Vector", 0, 100, key="eval_lead")
            comm = st.slider("Communication Articulation Index", 0, 100, key="eval_comm")
        with c3:
            proj = st.slider("Completed Open Source Projects", 0, 5, key="eval_proj")
            inter = st.slider("Live Technical Interview Evaluation", 0, 100, key="eval_inter")
            signal_type = st.radio("Resume Phrasing Signal Pattern", ["Standard Profile", "Features Gendered Phrases (e.g., Women's Club)"], key="eval_signal")
        
        st.divider()

        # ---------------------------------------------------------------------
        # 4. 【直接运行算法审计】：去除表单包裹后，每次滑块变动均会自动、实时触发模型预测
        # ---------------------------------------------------------------------
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
        
        # 实时渲染当前的评分指标分析看板
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

        st.divider()

        # ---------------------------------------------------------------------
        # 5. 【治理改写四键客】：人机协作（Human-in-the-Loop）多选择一键留痕提交
        # ---------------------------------------------------------------------
        st.markdown("##### 👥 Human-in-the-Loop Governance Override")
        hr_feedback = st.text_area(
            "HR Governance Feedback & Auditing Notes", 
            placeholder="Provide qualitative justification for manual override, bias mitigation tracking or compliance record logs...",
            key="eval_notes"
        )

        st.caption("Select a governance action below to finalize this evaluation and write directly to the log:")
        
        # 定义核心的决策控制变量
        action_triggered = False
        is_override = False
        decision_text = "—"
        
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            if st.button("🟢 Ignore (Follow AI)", use_container_width=True, help="Accept fairness-improved recommendations entirely."):
                action_triggered = True
                is_override = False
                decision_text = "No Override - Follow AI Advice"
        with b_col2:
            if st.button("💙 Force Approve", type="primary", use_container_width=True, help="Override algorithm to manually move candidate to interview."):
                action_triggered = True
                is_override = True
                decision_text = "Force Approve / Recommend for Interview"
        with b_col3:
            if st.button("🔴 Force Reject", use_container_width=True, help="Override algorithm to manually decline candidate profile."):
                action_triggered = True
                is_override = True
                decision_text = "Force Reject / Decline Profile"
        with b_col4:
            if st.button("⚠️ Defer / Flag", use_container_width=True, help="Hold profiles for secondary cross-departmental auditing."):
                action_triggered = True
                is_override = True
                decision_text = "Defer / Flag for Review"

        # 执行日志封装追加与会话推送
        if action_triggered:
            eval_id = f"EVAL-{len(st.session_state.audit_entries) + 1:03d}"
            
            audit_entry = {
                "evaluation_id": eval_id,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "candidate_name": candidate_name if candidate_name.strip() else "Unnamed Candidate",
                "biased_score": float(evaluation_res['biased_score']),
                "fair_score": float(evaluation_res['fair_score']),
                "biased_recommendation": evaluation_res.get('biased_recommendation', "Recommend" if evaluation_res['biased_score'] >= 0.5 else "Reject"),
                "fair_recommendation": evaluation_res.get('fair_recommendation', "Recommend" if evaluation_res['fair_score'] >= 0.5 else "Reject"),
                "recruiter_override": is_override,
                "recruiter_decision": decision_text,
                "recruiter_notes": hr_feedback if hr_feedback.strip() else "No comment supplied by auditor."
            }
            
            st.session_state.audit_entries.append(audit_entry)
            st.toast(f"📝 {eval_id} logged with action: {decision_text}!", icon="💾")
            st.success(f"📝 System captured entry successfully! Target parameter logged under action: **{decision_text}**.")

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
