"""
page_evaluation.py
------------------
Streamlined Recruiter Workspace optimized for live presentations.
Eliminates text inputs and sliders. Provides instant candidate drop-downs 
and direct clicking action buttons for human-in-the-loop overrides.
Clean coding standard applied to prevent invisible copy-paste syntax leaks.
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

    # 1. 核心计算与总体差距柱状图渲染（保持图表极简典雅，数字内嵌防拉伸）
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

    # 渲染防止拉伸的经典 Matplotlib 柱状图
    fig, ax = plt.subplots(figsize=(12, 4))
    indices = np.arange(len(differentials))
    bar_width = 0.35

    b_data = differentials["biased_score_gap"] * 100
    f_data = differentials["fair_score_gap"] * 100

    bar1 = ax.bar(indices - bar_width/2, b_data, bar_width, label="Biased Model Gap", color="#D9534F")
    bar2 = ax.bar(indices + bar_width/2, f_data, bar_width, label="Fairness-Aware Model Gap", color="#5CB85C")

    # 紧凑视窗控制，绝不产生高度拉伸 Bug
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

    st.divider()

    # 2. 全新重构的「HR 智能审计控制台」：全鼠标点击选择操作
    st.subheader("🎯 Live Recruiter Audit Console (Human-in-the-Loop)")
    st.markdown("Select a candidate profile from the dropdown below to instantly invoke dual-model compliance tracking.")

    # 扁平化构建 20 位候选人的秒级检索列表（10组简历配对，每组包含1男1女）
    candidate_options = []
    candidate_lookup_table = {}

    for pair in resume_pairs.RESUME_PAIRS:
        p_id = pair["pair_id"]
        scenario_title = pair["scenario"]
        
        # 绑定男性实体
        m_key = f"Pair {p_id} (Male) - {pair['male']['name']} [{scenario_title}]"
        candidate_options.append(m_key)
        candidate_lookup_table[m_key] = {"meta": pair["male"], "gender": "Male", "narrative": pair["narrative"], "pair_id": p_id}
        
        # 绑定女性实体
        f_key = f"Pair {p_id} (Female) - {pair['female']['name']} [{scenario_title}]"
        candidate_options.append(f_key)
        candidate_lookup_table[f_key] = {"meta": pair["female"], "gender": "Female", "narrative": pair["narrative"], "pair_id": p_id}

    # 100% 鼠标点击选择框
    selected_candidate_key = st.selectbox("Choose Target Candidate Profile to Review:", candidate_options)
    
    # 抓取当前选中的候选人底层能力指标特征
    current_target = candidate_lookup_table[selected_candidate_key]
    candidate_data = current_target["meta"]
    
    # 注入模型必需的性别信号量特征（男性 0.85，女性含有Women's词汇赋予 0.15）
    payload = candidate_data.copy()
    payload["resume_gender_signal"] = 0.85 if current_target["gender"] == "Male" else 0.15

    # 调度算法核心进行在线盲测评估
    evaluation_res = models.score_candidate(
        payload,
        st.session_state.biased_model,
        st.session_state.fair_model,
        st.session_state.fair_scaler
    )

    # 显示该求职者的学术背景叙事说明
    st.info(f"📌 **Qualitative Background & Context:** {current_target['narrative']}")

    # 左右并排展示候选人的核心特质与两套模型的运行评分百分比对比
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

    # 3. 核心功能：纯人工点击选项按钮（👍批准 / 👎淘汰），无需填写任何文字需求！
    st.markdown("##### 👥 Recruiter Executive Action (Click to issue final decree)")
    st.caption("As a mandatory CSR governance framework, AI only provides recommendation vectors; humans retain absolute executive decision authority.")
    
    act_col1, act_col2 = st.columns(2)
    
    # 点击选项 1：强制录用 / 允许面试
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
        
    # 点击选项 2：直接淘汰 / 婉拒
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
