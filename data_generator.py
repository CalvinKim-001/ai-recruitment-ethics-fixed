"""
data_generator.py
-----------------
Generates synthetic hiring data for the AI Recruitment Bias project.

ETHICAL NOTE:
This data is intentionally imbalanced — male candidates are given higher
historical hiring probabilities. This simulates how real-world biased historical 
data causes AI systems to learn discriminatory patterns.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_hiring_dataset(n_candidates: int = 1000) -> pd.DataFrame:
    """
    Generate a synthetic dataset of junior software engineer candidates.
    """
    # STEP 1: Assign gender with realistic tech-industry historical imbalance
    gender = np.random.choice(
        ["Male", "Female"],
        size=n_candidates,
        p=[0.72, 0.28]  # 72% male, 28% female
    )

    # STEP 2: Generate candidate qualifications (Distributions are identical across genders)
    years_experience = np.random.randint(0, 8, size=n_candidates)

    education_level = np.random.choice(
        [0, 1, 2, 3],  # 0=High School, 1=Bachelor's, 2=Master's, 3=PhD
        size=n_candidates,
        p=[0.05, 0.60, 0.30, 0.05]
    )

    programming_skill = np.clip(
        np.random.normal(loc=65, scale=15, size=n_candidates), 0, 100
    ).astype(int)

    leadership_score = np.clip(
        np.random.normal(loc=60, scale=14, size=n_candidates), 0, 100
    ).astype(int)

    communication_score = np.clip(
        np.random.normal(loc=62, scale=12, size=n_candidates), 0, 100
    ).astype(int)

    company_tier = np.random.choice([1, 2, 3, 4], size=n_candidates, p=[0.3, 0.4, 0.2, 0.1])
    project_experience = np.random.randint(0, 6, size=n_candidates)
    interview_score = np.clip(
        np.random.normal(loc=63, scale=15, size=n_candidates), 0, 100
    ).astype(int)

    # STEP 3: Generate the Hidden Proxy Feature (resume_gender_signal)
    # Replicates the Amazon case: phrasing patterns implicitly tied to gender
    resume_gender_signal = np.zeros(n_candidates)
    resume_gender_signal[gender == "Male"] = np.random.normal(loc=0.78, scale=0.08, size=(gender == "Male").sum())
    resume_gender_signal[gender == "Female"] = np.random.normal(loc=0.28, scale=0.08, size=(gender == "Female").sum())
    resume_gender_signal = np.clip(resume_gender_signal, 0.0, 1.0)

    # STEP 4: Calculate historical hiring probability using capability + subtle bias
    # The bias coefficient (0.12) creates a visible but realistic, smooth historical skew
    ability_score = (
        0.04 * years_experience +
        0.06 * education_level +
        0.004 * programming_skill +
        0.002 * leadership_score +
        0.002 * communication_score +
        0.03 * company_tier +
        0.02 * project_experience +
        0.004 * interview_score
    )

    # Normalize capability score to 0-1
    ability_score = (ability_score - ability_score.min()) / (ability_score.max() - ability_score.min())
    
    # Inject smooth proxy signal bias into historical human decisions
    biased_prob = np.clip(ability_score * 0.8 + 0.20 * resume_gender_signal, 0, 1)

    # Convert to binary outcomes (1 = hired, 0 = rejected)
    hired = np.random.binomial(1, biased_prob)

    # STEP 5: Assemble full aligned DataFrame
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
        "resume_gender_signal": resume_gender_signal,
        "hired": hired
    })

    return df


def get_education_label(level: int) -> str:
    mapping = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}
    return mapping.get(level, "Unknown")


def get_company_tier_label(tier: int) -> str:
    mapping = {1: "Startup", 2: "Mid-size", 3: "Large Corp", 4: "Top Tech (FAANG)"}
    return mapping.get(tier, "Unknown")


def get_dataset_summary(df: pd.DataFrame) -> dict:
    summary = {}
    for g in ["Male", "Female"]:
        subset = df[df["gender"] == g]
        summary[g] = {
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
    summary = get_dataset_summary(df)
    print(summary)
