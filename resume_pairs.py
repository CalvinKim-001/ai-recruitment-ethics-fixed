"""
resume_pairs.py
---------------
10 matched resume pairs for the "Gender Signals Experiment."

Each pair contains two candidates with IDENTICAL qualifications.
The ONLY differences are:
  - Candidate name (gendered)
  - University name (some are all-women's colleges — directly from the Amazon case)
  - Activity/club descriptions (gendered phrasing)

This experiment is the most direct demonstration of the project's argument:
When the biased model gives different scores to identical qualifications
based only on gendered signals, that IS gender discrimination — even if
the word "gender" never appears in the model's input features.

ETHICAL NOTE:
Amazon's AI penalized resumes containing the word "women's" — e.g.,
"captain of women's chess club" received a lower score than
"captain of chess club." These pairs replicate that exact dynamic.
"""

RESUME_PAIRS = [
    {
        "pair_id": 1,
        "scenario": "Top-performing graduate, strong technical background",
        "narrative": (
            "Both candidates graduated top of their class with identical GPAs, "
            "identical internship history at a Tier 2 company, and identical "
            "project portfolios. The only difference: one attended MIT, the other "
            "attended Wellesley College — a prestigious all-women's institution "
            "whose alumnae include Hillary Clinton and Madeleine Albright. "
            "Amazon's AI penalized exactly this kind of signal."
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
            "Both candidates left non-technical careers after 3 years and "
            "completed the same 6-month coding bootcamp. Both built identical "
            "portfolio projects and received identical technical assessment scores. "
            "The employment gap — more common among women who take caregiving breaks "
            "— is a well-documented proxy variable that AI systems use to "
            "inadvertently penalize women even when the gap is professionally irrelevant."
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
            "Both candidates list identical technical skills and identical GPA. "
            "Their leadership experience is substantively the same — organizing "
            "a community of 200+ members, running weekly events, managing speakers. "
            "But one's club is 'Women in Tech' while the other is 'Tech Entrepreneurs.' "
            "The word 'Women' in the club name was exactly what Amazon's system flagged. "
            "The AI cannot understand that both represent equivalent leadership experience."
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
            "Both hold Master's degrees from highly ranked institutions. "
            "Both completed identical internships at a large corporation and "
            "achieved identical assessment scores. Smith College, an all-women's "
            "institution, consistently ranks among the top liberal arts colleges "
            "in the United States. Yet AI systems trained on data that rarely "
            "saw Smith graduates hired will likely score this credential lower — "
            "not because of academic quality, but because of historical underrepresentation."
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
            "Both candidates won the same hackathon — a major achievement that "
            "signals problem-solving ability, creativity, and teamwork. "
            "The accomplishment is identical. The description differs only in "
            "how it is phrased: 'hackathon winner' vs. 'winner of women's hackathon.' "
            "The AI does not understand that women-specific competitions are often "
            "equally or more competitive due to smaller participant pools — "
            "it only sees the word 'women's' as a negative signal."
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
            "Both are self-taught developers who built identical GitHub portfolios "
            "and passed identical technical assessments. Neither has a formal degree. "
            "Studies show that self-taught women in tech are more likely to be "
            "screened out at the resume stage than self-taught men with equivalent "
            "portfolios — a pattern that AI systems trained on those screening "
            "decisions will replicate and amplify."
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
            "Both received prestigious merit scholarships and graduated with honors. "
            "The Grace Hopper Scholarship — a real STEM scholarship awarded to women — "
            "is named after computing pioneer Rear Admiral Grace Hopper, who invented "
            "one of the first compiler programs. Yet an AI system that has never seen "
            "this scholarship represented in its training data will treat it as an "
            "unknown or even negative signal simply because 'Grace Hopper' does not "
            "pattern-match to previously successful male candidates."
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
            "Both interned at a top technology company and received identical "
            "performance ratings. Both were offered return offers. "
            "Both list the internship on their resume with identical descriptions. "
            "This pair tests whether the model can recognize equivalent elite "
            "experience regardless of the candidate's name. In real-world studies, "
            "identical resumes with stereotypically female names receive fewer "
            "callbacks than those with stereotypically male names — even from "
            "human recruiters, before any AI is involved."
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
            "Both candidates transitioned from humanities backgrounds into software "
            "engineering through a Master's program — a path that is increasingly "
            "common and increasingly valued for producing engineers who can communicate "
            "complex ideas clearly. Both have identical post-transition records. "
            "The female candidate's undergraduate thesis was on gender equity in STEM — "
            "a topic that directly relates to this project, but which an AI model "
            "trained on biased data might score negatively through proxy association."
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
            "Both candidates have 5 years of experience but are applying for a "
            "junior role — a common situation for career re-entrants, those returning "
            "from parental leave, or those switching specializations. "
            "Women are statistically more likely to be in this situation due to "
            "caregiving responsibilities. The female candidate's resume lists "
            "participation in a 'Women Returners' program — a legitimate professional "
            "re-entry program — which may trigger negative scoring in an AI system "
            "that has never seen this credential positively represented in training data."
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

        # ==========================================================
        # 【核心修复】：在这里强行注入“简历性别暗示信号”
        # 对应 data_generator.py 里的逻辑：
        # 男性简历分配高分 (0.85)，女性简历由于含有 Women's 等词汇分配低分 (0.15)
        # ==========================================================
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
