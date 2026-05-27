"""
data_generator.py
-----------------
Generates synthetic hiring data for the AI Recruitment Bias project.

ETHICAL NOTE:
This data is intentionally imbalanced — male candidates are given higher
historical hiring probabilities via a proxy variable ("resume_gender_signal").
This simulates how real-world biased historical data (like Amazon's 10 years of 
male-dominated hiring records) causes AI systems to learn discriminatory patterns.
The bias is in the data, not in reality.

This is a core lesson of the project: garbage in, garbage out.
If historical data reflects discrimination, the AI will reproduce it.
"""

import numpy as np
import pandas as pd

# Fix the random seed so results are reproducible every time the code runs.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_hiring_dataset(n_candidates: int = 1000) -> pd.DataFrame:
    """
    Generate a synthetic dataset of junior software engineer candidates.

    The dataset intentionally contains historical gender imbalance and proxy bias:
    - ~72% of candidates are male (reflecting tech industry historical norms)
    - Candidates' resumes feature a 'resume_gender_signal' (proxy variable)
      which is highly correlated with gender due to historical phrasing patterns.
    - The historical 'hired' label is heavily biased toward high gender signals.
    """

    # ----------------------------------------------------------------
    # STEP 1: Assign gender with realistic tech-industry imbalance
    # ----------------------------------------------------------------
    gender = np.random.choice(
        ["Male", "Female"],
        size=n_candidates,
        p=[0.72, 0.28]  # 72% male, 28% female — intentional historical imbalance
    )

    # ----------------------------------------------------------------
    # STEP 2: Generate candidate qualifications (Fair & Equal)
    # ----------------------------------------------------------------
    # We deliberately make qualifications SIMILAR across genders.
    years_experience = np.random.randint(0, 8, size=n_candidates)

    education_level = np.random.choice(
        [0, 1, 2, 3],  # 0=High School, 1=Bachelor's, 2=Master's, 3=PhD
        size=n_candidates,
        p=[0.05, 0.60, 0.30, 0.05]
    )

    programming_skill = np.clip(
        np.random.normal(loc=65, scale=18, size=n_candidates), 0, 100
    ).astype(int)

    leadership_score = np.clip(
        np.random.normal(loc=60, scale=15, size=n_candidates), 0, 100
    ).astype(int)

    communication_score = np.clip(
        np.random.normal(loc=62, scale=14, size=n_candidates), 0, 100
    ).astype(int)

    company_tier = np.random.choice([1, 2, 3, 4], size=n_candidates, p=[0.3, 0.4, 0.2, 0.1])
    project_experience = np.random.randint(0, 6, size=n_candidates)

    interview_score = np.clip(
        np.random.normal(loc=63, scale=16, size=n_candidates), 0, 100
    ).astype(int)

    # ----------------------------------------------------------------
    # STEP 3: Generate the Proxy Variable (resume_gender_signal)
    # ----------------------------------------------------------------
    # This replicates the Amazon AI Case study.
    # The feature doesn't say "Gender", it represents text signals (clubs, sports, colleges)
    # Male resumes naturally skew high (0.7-0.95), Female resumes skew low (0.05-0.3) due to phrasing.
    resume_gender_signal = np.zeros(n_candidates)
    
    male_mask = (gender == "Male")
    female_mask = (gender == "Female")
    
    resume_gender_signal[male_mask] = np.random.uniform(0.68, 0.95, size=male_mask.sum())
    resume_gender_signal[female_mask] = np.random.uniform(0.05, 0.32, size=female_mask.sum())

    # ----------------------------------------------------------------
    # STEP 4: Generate biased historical hiring outcomes
    # ----------------------------------------------------------------
    # We calculate a hire capability based on real merit first
    base_hire_prob = (
        0.05 * years_experience +
        0.08 * education_level +
        0.003 * programming_skill +
        0.002 * leadership_score +
        0.001 * communication_score +
        0.04 * company_tier +
        0.02 * project_experience +
        0.003 * interview_score
    )

    # Normalize to a 0–1 probability range
    base_hire_prob = (base_hire_prob - base_hire_prob.min()) / (
        base_hire_prob.max() - base_hire_prob.min()
    )

    # 【核心注入】：让人类历史上的招聘结果（hired标签）高度依赖于 resume_gender_signal。
    # 这意味着包含男性化词汇、传统男性爱好的简历在历史记录中更容易被录用（+25% 权重权重提升）。
    proxy_bias_boost = 0.25 * resume_gender_signal
    biased_hire_prob = np.clip(base_hire_prob + proxy_bias_boost, 0, 1)

    # Convert probabilities to binary outcomes (hired = 1, rejected = 0)
    hired = np.random.binomial(1, biased_hire_prob)

    # ----------------------------------------------------------------
    # STEP 5: Assemble the DataFrame (Aligned perfectly with FEATURE_COLUMNS)
    # ----------------------------------------------------------------
    df = pd.DataFrame({
        "candidate_id": [f"C{str(i).zfill(4)}" for i in range(1, n_candidates + 1)],
        "gender": gender,
        "years_experience": years_experience,
        "education_level": education_level,
        "programming_skill": programming_skill,
        "leadership_score": leadership_score,
        "communication_score": communication_score,
        "company_tier": company_tier,
        "project_experience": project_experience,
        "interview_score": interview_score,
        "resume_gender_signal": resume_gender_signal,  # 成功包含此关键代理特征
        "hired": hired  # 1 = historically hired, 0 = historically rejected
    })

    return df


def get_education_label(level: int) -> str:
    """Convert numeric education level to human-readable label."""
    mapping = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}
    return mapping.get(level, "Unknown")


def get_company_tier_label(tier: int) -> str:
    """Convert numeric company tier to human-readable label."""
    mapping = {1: "Startup", 2: "Mid-size", 3: "Large Corp", 4: "Top Tech (FAANG)"}
    return mapping.get(tier, "Unknown")


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary of the dataset's gender distribution and hiring rates.
    """
    summary = {}
    for gender in ["Male", "Female"]:
        subset = df[df["gender"] == gender]
        summary[gender] = {
            "count": len(subset),
            "pct_of_dataset": round(len(subset) / len(df) * 100, 1),
            "hire_rate": round(subset["hired"].mean() * 100, 1),
            "avg_gender_signal": round(subset["resume_gender_signal"].mean(), 2),
            "avg_programming_skill": round(subset["programming_skill"].mean(), 1),
            "avg_interview_score": round(subset["interview_score"].mean(), 1),
        }
    return summary


if __name__ == "__main__":
    df = generate_hiring_dataset(1000)
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nGender distribution and hiring rates:")
    summary = get_dataset_summary(df)
    for gender, stats in summary.items():
        print(f"\n  {gender}:")
        for k, v in stats.items():
            print(f"    {k}: {v}")
