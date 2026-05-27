"""
page_evaluation.py
------------------
Streamlined Recruiter Workspace optimized for live presentations.
Eliminates text inputs and sliders. Provides instant candidate drop-downs 
and direct clicking action buttons for human-in-the-loop overrides.
Enhanced Matplotlib readability layer to ensure giant, ultra-clear numbers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import models
import resume_pairs

def render():
    st.title("🤝 Algorithmic Auditing & Human Governance Workspace")
    st.markdown("Select a candidate profile to view risk disparity metrics, then use the clicking tokens below to issue human decisions.")

    # Model training validation gate
    if not st.session_state.models_trained:
        st.warning("⚙️ AI Models are not initialized yet. Please navigate to the **Fairness Audit Dashboard** to trigger training routines first.")
        return

    # 1. 核心计算与总体差距柱状图渲染
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

    # 顶部四大核心指标看板
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Gap - Biased Model", f"{avg_b_gap:+.2f}pp")
    m2.metric("Avg Gap - Fair Model", f"{avg_f_gap:+.2f}pp")
    m3.metric("Average Gap Mitigation", f"{avg_red:.2f}pp")
    m4.metric("Mitigated Pairs", f"{imp_p}/10")

    st.markdown("#### Score Disparity Visualizer")
    
    # 建立宽敞的画幅控制结构
    fig, ax = plt.subplots(figsize=(12, 5.5))
    indices = np.arange(len(differentials))
    bar_width = 0.35

    # 换算至百分点值 (pp)
    b_data = differentials["biased_score_gap"] * 100
    f_data = differentials["fair_score_gap"] * 100

    bar1 = ax.bar(indices - bar_width/2, b_data, bar_width, label="Biased Model Gap", color="#D9534F")
    bar2 = ax.bar(indices + bar_width/2, f_data, bar_width, label="Fairness-Aware Model Gap", color="#5CB85C")

    # =========================================================================
    # 【视觉无敌优化】：硬性锁定纵轴范围，配合超大加粗数字标签，绝不重叠
    # =========================================================================
    # 强制让 Y 轴视窗至少保持在 -15 到 +15 之间，留出完美的垂直呼吸空间
    abs_max = max(abs(b_data).max(), abs(f_data).max(), 10.0)
    ax.set_ylim(-abs_max * 1.4, abs_max * 1.4)

    # 放大并加粗柱状图顶部的数字标签，使用高对比度深色
    ax.bar_label(bar1, fmt='%.1fpp', padding=5, fontsize=10, weight='bold', color='#B32420')
    ax.bar_label(bar2, fmt='%.1fpp', padding=5, fontsize=10, weight='bold', color='#2B7A2B')

    ax.set_ylabel("Score Disparity: Male - Female (pp)", fontsize=11, weight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels([f"Pair {i+1}" for i in range(10)], fontsize=10, weight='bold')
    
    # 强化0刻度基准线
    ax.axhline(0, color="#555555", linestyle="-", linewidth=1.2, alpha=0.7)
    
    # 开启轻量级网格线辅助视觉对齐
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # 2. 全新重构的「HR 智能审计控制台」
    st.subheader("🎯 Live Recruiter Audit Console (Human-in-the-Loop)")
    st.markdown("Select a candidate profile from the dropdown below to instantly invoke dual-model compliance tracking.")

    candidate_options = []
    candidate_lookup_table = {}

    for pair in resume_pairs.RESUME_PAIRS:
        p_id = pair["pair_id"]
        scenario_title = pair["scenario"]
        
        m_key = f"Pair {p_id} (Male) - {pair['male']['name']} [{scenario_title}]"
        candidate_options.append(m_key)
        candidate_lookup_table[m_key] = {"meta": pair["male"], "gender": "Male", "narrative": pair["narrative"], "pair_id": p_id}
        
        f_key = f"Pair {p_id} (Female) - {pair['female']['name']} [{scenario_title}]"
        candidate_options.append(f_key)
        candidate_lookup_table[f_key] = {"meta": pair["female"], "gender": "Female", "narrative": pair["narrative"], "pair_id": p_id}

    selected_candidate_key = st.selectbox("Choose Target Candidate Profile to Review:", candidate_options)
    
    current_target = candidate_lookup_table[selected_candidate_key]
    candidate_data = current_target["meta"]
    
    payload = candidate_data.copy()
    payload["resume_gender_signal"] = 0.85 if current_target["gender"] == "Male" else 0.15

    evaluation_res = models.score_candidate(
        payload,
        st.session_state.biased_model,
        st.session_state.fair_model,
        st.session_state.fair_scaler
    )

    st.info(f"📌 **Qualitative Background & Context:** {current_target['narrative']}")

    col_profile, col_scores = st.columns([4, 3])
    
    with col_profile:
        st.markdown(f"##### 📋 Candidate Dossier: {candidate_data['name']} ({current_target['gender']})")
        st.markdown(f"* **Graduated From:** `{candidate_data['university']}`")
        st.markdown(f"* **Resume Extracted Activity:** `{candidate_data['activity']}`")
        st.markdown(f"* **Experience & Interview:** {candidate_data['years_experience']} Years Exp | Interview Score: {candidate_data['interview_score']}/100")
        st.markdown(f"* **Hard Skills Matrix:** Tech: {candidate_data['programming_skill']} | Leadership: {candidate_data['leadership_score']} | Comm: {candidate_data['communication_score']}")

    with col_scores:
        st.markdown("##### 📊 Algorithmic Scoring")
        st.metric("Biased Baseline Model Score", f"{evaluation_res['biased_score']*100:.1f}%")
        st.metric("Fairness-Improved Model Score", f"{evaluation_res['fair_score']*100:.1f}%")

    # 3. 纯人工点击选项按钮
    st.markdown("##### 👥 Recruiter Executive Action (Click to issue final decree)")
    st.caption("As a mandatory CSR governance framework, AI only provides recommendation vectors; humans retain absolute executive decision authority.")
    
    act_col1, act_col2 = st.columns(2)
    
    if act_col1.button("👍 Human Override: Approve for Interview", type="primary", use_container_width=True):
        audit_entry = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "candidate_name": candidate_data['name'],
            "biased_score": evaluation_res['biased_score'],
            "fair_score": evaluation_res['fair_score'],
            "override_status": "Force Approve / Recommend for Interview",
            "feedback": f"Auditor executed manual compliance override. Mitigated baseline data bias for {candidate_data['name']}."
        }
        st.session_state.audit_entries.append(audit_entry)
        st.success(f"📝 Decision Logged: {candidate_data['name']} has been manually APPROVED. Record synced to Governance tab!")
        
    if act_col2.button("👎 Human Decision: Decline Profile", type="secondary", use_container_width=True):
        audit_entry = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "candidate_name": candidate_data['name'],
            "biased_score": evaluation_res['biased_score'],
            "fair_score": evaluation_res['fair_score'],
            "override_status": "Force Reject / Decline Profile",
            "feedback": f"Auditor declined profile entry based on combined contextual human review."
        }
        st.session_state.audit_entries.append(audit_entry)
        st.warning(f"📝 Decision Logged: {candidate_data['name']} has been manually DECLINED. Record synced to Governance tab!")
