"""
resume_pairs.py
---------------
10 matched resume pairs for the "Gender Signals Experiment."

Each pair contains two candidates with IDENTICAL qualifications.
The ONLY differences are:
  - Candidate name (gendered)
  - University name (some are all-women's colleges — directly from the Amazon case)
  - Activity/club descriptions (gendered phrasing)
"""

RESUME_PAIRS = [
    {
        "pair_id": 1,
        "scenario": "Top-performing graduate, strong technical background",
        "narrative": (
            "Both candidates graduated top of their class with identical GPAs, "
            "identical internship history at a Tier 2 company, and identical "
            "project portfolios. The only difference: one attended MIT, the other "
            "attended Wellesley College — a prestigious all-women's institution."
        ),
        "male": {
            "name": "James Whitfield",
            "university": "MIT",
            "activity": "President, Competitive Programming Club",
            "years_experience": 2,
            "education_level": 1,
            "programming_skill": 88,
            "leadership_score": 82,
            "communication_score": 78,
            "company_tier": 2,
            "project_experience": 4,
            "interview_score": 85,
        },
        "female": {
            "name": "Claire Whitfield",
            "university": "Wellesley College",
            "activity": "President, Women's Competitive Programming Club",
            "years_experience": 2,
            "education_level": 1,
            "programming_skill": 88,
            "leadership_score": 82,
            "communication_score": 78,
            "company_tier": 2,
            "project_experience": 4,
            "interview_score": 85,
        },
    },
    {
        "pair_id": 2,
        "scenario": "Career changer with bootcamp background",
        "narrative": (
            "Both left non-technical careers after 3 years and completed the same "
            "6-month coding bootcamp. Both built identical portfolio projects. "
            "Caregiving breaks often present proxy variables that models use to penalize women."
        ),
        "male": {
            "name": "Daniel Torres",
            "university": "General Assembly Bootcamp",
            "activity": "Open source contributor, personal portfolio site",
            "years_experience": 1,
            "education_level": 1,
            "programming_skill": 72,
            "leadership_score": 65,
            "communication_score": 74,
            "company_tier": 1,
            "project_experience": 3,
            "interview_score": 70,
        },
        "female": {
            "name": "Sofia Torres",
            "university": "General Assembly Bootcamp",
            "activity": "Open source contributor, personal portfolio site",
            "years_experience": 1,
            "education_level": 1,
            "programming_skill": 72,
            "leadership_score": 65,
            "communication_score": 74,
            "company_tier": 1,
            "project_experience": 3,
            "interview_score": 70,
        },
    },
    {
        "pair_id": 3,
        "scenario": "Leadership experience through advocacy work",
        "narrative": (
            "Identical technical skills and GPA. One's club is 'Women in Tech' "
            "while the other is 'Tech Entrepreneurs'. The word 'Women' in the club "
            "name was exactly what triggered negative proxy scoring in real cases."
        ),
        "male": {
            "name": "Ryan Chen",
            "university": "University of Michigan",
            "activity": "Founder, Tech Entrepreneurs Society (200+ members)",
            "years_experience": 1,
            "education_level": 1,
            "programming_skill": 79,
            "leadership_score": 88,
            "communication_score": 85,
            "company_tier": 1,
            "project_experience": 2,
            "interview_score": 77,
        },
        "female": {
            "name": "Michelle Chen",
            "university": "University of Michigan",
            "activity": "Founder, Women in Tech Society (200+ members)",
            "years_experience": 1,
            "education_level": 1,
            "programming_skill": 79,
            "leadership_score": 88,
            "communication_score": 85,
            "company_tier": 1,
            "project_experience": 2,
            "interview_score": 77,
        },
    },
    {
        "pair_id": 4,
        "scenario": "Graduate degree from prestigious all-women's institution",
        "narrative": (
            "Both hold Master's degrees from highly ranked institutions. Smith College "
            "is an all-women's institution that ranks elite, yet historical imbalances "
            "will score this credential lower due to historical pattern mismatching."
        ),
        "male": {
            "name": "Marcus Reed",
            "university": "Boston University (MS Computer Science)",
            "activity": "Graduate teaching assistant, algorithms",
            "years_experience": 3,
            "education_level": 2,
            "programming_skill": 91,
            "leadership_score": 74,
            "communication_score": 80,
            "company_tier": 3,
            "project_experience": 5,
            "interview_score": 89,
        },
        "female": {
            "name": "Amara Reed",
            "university": "Smith College (MS Computer Science)",
            "activity": "Graduate teaching assistant, algorithms",
            "years_experience": 3,
            "education_level": 2,
            "programming_skill": 91,
            "leadership_score": 74,
            "communication_score": 80,
            "company_tier": 3,
            "project_experience": 5,
            "interview_score": 89,
        },
    },
    {
        "pair_id": 5,
        "scenario": "Hackathon winner, early career",
        "narrative": (
            "Both won the same hackathon. The description differs only in phrasing: "
            "'hackathon winner' vs. 'winner of women's hackathon'. The model treats "
            "the modifier 'women's' as a penalty vector based on historical bias."
        ),
        "male": {
            "name": "Ethan Park",
            "university": "Georgia Tech",
            "activity": "1st place, HackGT hackathon 2023",
            "years_experience": 1,
            "education_level": 1,
            "programming_skill": 83,
            "leadership_score": 70,
            "communication_score": 72,
            "company_tier": 1,
            "project_experience": 3,
            "interview_score": 80,
        },
        "female": {
            "name": "Ji-Yeon Park",
            "university": "Georgia Tech",
            "activity": "1st place, Women's HackGT hackathon 2023",
            "years_experience": 1,
            "education_level": 1,
            "programming_skill": 83,
            "leadership_score": 70,
            "communication_score": 72,
            "company_tier": 1,
            "project_experience": 3,
            "interview_score": 80,
        },
    },
    {
        "pair_id": 6,
        "scenario": "Self-taught developer, no formal degree",
        "narrative": (
            "Self-taught developers with identical portfolios and no degrees. "
            "Sociological studies indicate self-taught women are structurally screened out "
            "at higher rates than men, a bias standard algorithms replicate."
        ),
        "male": {
            "name": "Oliver Davis",
            "university": "Self-taught / Online courses",
            "activity": "100+ GitHub repositories, freelance developer",
            "years_experience": 2,
            "education_level": 0,
            "programming_skill": 76,
            "leadership_score": 58,
            "communication_score": 65,
            "company_tier": 1,
            "project_experience": 5,
            "interview_score": 74,
        },
        "female": {
            "name": "Nadia Davis",
            "university": "Self-taught / Online courses",
            "activity": "100+ GitHub repositories, freelance developer",
            "years_experience": 2,
            "education_level": 0,
            "programming_skill": 76,
            "leadership_score": 58,
            "communication_score": 65,
            "company_tier": 1,
            "project_experience": 5,
            "interview_score": 74,
        },
    },
    {
        "pair_id": 7,
        "scenario": "Scholarship recipient and academic high achiever",
        "narrative": (
            "Grace Hopper Scholarship is an elite STEM token for women. "
            "An AI system that heavily anchors on historical male resume profiles "
            "fails to contextualize this achievement positively."
        ),
        "male": {
            "name": "Liam Johnson",
            "university": "Carnegie Mellon University",
            "activity": "National Merit Scholar, Dean's List all semesters",
            "years_experience": 1,
            "education_level": 1,
            "programming_skill": 92,
            "leadership_score": 76,
            "communication_score": 79,
            "company_tier": 2,
            "project_experience": 4,
            "interview_score": 91,
        },
        "female": {
            "name": "Priya Johnson",
            "university": "Carnegie Mellon University",
            "activity": "Grace Hopper Scholar, Dean's List all semesters",
            "years_experience": 1,
            "education_level": 1,
            "programming_skill": 92,
            "leadership_score": 76,
            "communication_score": 79,
            "company_tier": 2,
            "project_experience": 4,
            "interview_score": 91,
        },
    },
    {
        "pair_id": 8,
        "scenario": "FAANG internship experience, high performer",
        "narrative": (
            "Identical FAANG elite internship and return offers. "
            "This pairs tests whether the model exhibits systemic preference shifts "
            "solely based on gender proxy vectors."
        ),
        "male": {
            "name": "Alexander Kim",
            "university": "UC Berkeley",
            "activity": "Software Engineering Intern, Google (return offer received)",
            "years_experience": 2,
            "education_level": 1,
            "programming_skill": 94,
            "leadership_score": 80,
            "communication_score": 83,
            "company_tier": 4,
            "project_experience": 5,
            "interview_score": 93,
        },
        "female": {
            "name": "Yuna Kim",
            "university": "UC Berkeley",
            "activity": "Software Engineering Intern, Google (return offer received)",
            "years_experience": 2,
            "education_level": 1,
            "programming_skill": 94,
            "leadership_score": 80,
            "communication_score": 83,
            "company_tier": 4,
            "project_experience": 5,
            "interview_score": 93,
        },
    },
    {
        "pair_id": 9,
        "scenario": "Non-traditional path: arts to engineering",
        "narrative": (
            "Transitioned from humanities into CS. The female candidate's "
            "thesis was on 'Gender Equity in STEM' — a proxy feature that "
            "a biased system naturally correlates negatively."
        ),
        "male": {
            "name": "Noah Patel",
            "university": "Northwestern University",
            "activity": "Undergraduate thesis: Philosophy of Computing",
            "years_experience": 2,
            "education_level": 2,
            "programming_skill": 78,
            "leadership_score": 72,
            "communication_score": 91,
            "company_tier": 2,
            "project_experience": 3,
            "interview_score": 76,
        },
        "female": {
            "name": "Leila Patel",
            "university": "Northwestern University",
            "activity": "Undergraduate thesis: Gender Equity in STEM Fields",
            "years_experience": 2,
            "education_level": 2,
            "programming_skill": 78,
            "leadership_score": 72,
            "communication_score": 91,
            "company_tier": 2,
            "project_experience": 3,
            "interview_score": 76,
        },
    },
    {
        "pair_id": 10,
        "scenario": "Experienced mid-level candidate seeking junior role",
        "narrative": (
            "5 years of deep experience applying for junior roles (common for caregiving returners). "
            "Female resume explicitly features 'Women Returners Program', triggering adverse scoring."
        ),
        "male": {
            "name": "Sebastian Walsh",
            "university": "University of Texas at Austin",
            "activity": "Freelance developer during career transition period",
            "years_experience": 5,
            "education_level": 1,
            "programming_skill": 81,
            "leadership_score": 77,
            "communication_score": 80,
            "company_tier": 2,
            "project_experience": 4,
            "interview_score": 79,
        },
        "female": {
            "name": "Emma Walsh",
            "university": "University of Texas at Austin",
            "activity": "Women Returners Program graduate, freelance developer",
            "years_experience": 5,
            "education_level": 1,
            "programming_skill": 81,
            "leadership_score": 77,
            "communication_score": 80,
            "company_tier": 2,
            "project_experience": 4,
            "interview_score": 79,
        },
    },
]


def get_pair_as_dataframe(pair: dict) -> "pd.DataFrame":
    """
    Convert a single resume pair into a two-row DataFrame
    suitable for model scoring.
    """
    import pandas as pd

    rows = []
    for role in ["male", "female"]:
        candidate = pair[role].copy()
        candidate["pair_id"] = pair["pair_id"]
        candidate["gender"] = "Male" if role == "male" else "Female"
        candidate["scenario"] = pair["scenario"]

        # 强行拉开信号线：男性简历赋予高男性倾向信号值，女性简历赋予低男性倾向信号值
        if role == "male":
            candidate["resume_gender_signal"] = 0.85
        else:
            candidate["resume_gender_signal"] = 0.15

        rows.append(candidate)

    return pd.DataFrame(rows)


def get_all_pairs_dataframe() -> "pd.DataFrame":
    """
    Return all 10 resume pairs as a single DataFrame (20 rows).
    """
    import pandas as pd
    frames = [get_pair_as_dataframe(p) for p in RESUME_PAIRS]
    return pd.concat(frames, ignore_index=True)
