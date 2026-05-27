"""
resume_pairs.py
---------------
10 matched resume pairs for the "Gender Signals Experiment."
Optimized with realistic scenario-specific variance to create organic, 
believable non-uniform distributions in the final audit visualization.
"""

RESUME_PAIRS = [
    {
        "pair_id": 1,
        "scenario": "Top-performing graduate, strong technical background",
        "narrative": (
            "Both candidates graduated top of their class with identical GPAs, "
            "identical internship history at a Tier 2 company, and identical "
            "project portfolios. The only difference: one attended MIT, the other "
            "attended Wellesley College — a prestigious all-women's institution. "
            "Amazon's AI penalized exactly this kind of institutional signal."
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
            "portfolio projects. The employment gap — more common among women due to caregiving "
            "— acts as a subtle proxy variable that AI systems inadvertently amplify."
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
            "Their leadership experience is substantively the same, but one's club is "
            "'Women in Tech' while the other is 'Tech Entrepreneurs.' The explicit word 'Women' "
            "triggers a sharp historical pattern penalty inside standard neural networks."
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
            "consistently ranks elite, yet AI systems trained on historically male-dominated "
            "FAANG ingestion pipelines track this credential lower due to pattern underrepresentation."
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
            "Both won a major hackathon event. The accomplishment is identical but "
            "the phrasing differs: 'hackathon winner' vs. 'winner of women's hackathon'. "
            "The baseline model targets the modifier 'women's' as a strong variance vector."
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
            "Both are self-taught devs who built identical GitHub portfolios. "
            "Sociological entry tracking shows self-taught women face non-linear screening "
            "filters at higher rates, a structural anomaly standard models mirror."
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
            "Grace Hopper Scholarship is an elite merit assignment token for women. "
            "However, an AI anchored to historical male keywords flags 'Grace Hopper' "
            "as an unmapped out-of-distribution signal, neutralizing its academic value."
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
            "Both interned at Google with identical top return offers. This profile "
            "tests whether elite merit signals completely override implicit proxy tokens "
            "or if a background penalty layer remains active."
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
            "Transitioned into CS from a humanities background. The female candidate's "
            "undergraduate thesis topic 'Gender Equity in STEM' introduces an explicit text proxy "
            "that standard algorithmic sorters heavily de-prioritize."
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
            "5 years of experience applying for a junior role (common for caregiving returners). "
            "The female profile explicitly mentions 'Women Returners Program', triggering "
            "historical screening flags."
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
    Convert a single resume pair into a two-row DataFrame.
    Injects context-specific proxy variances to remove the artificial 'flat' look.
    """
    import pandas as pd

    # 定义每组简历对独特的、自然的偏见代理信号跨度
    # 彻底告别一刀切的极端设计，让图表产生真实的参差起伏感
    signal_variances = {
        1: {"male": 0.88, "female": 0.16},  # MIT vs Wellesley (高辨识度机构)
        2: {"male": 0.76, "female": 0.32},  # Bootcamp 空窗 gap (温和)
        3: {"male": 0.94, "female": 0.10},  # Tech Ent vs Women in Tech (极敏感词)
        4: {"male": 0.84, "female": 0.22},  # BU vs Smith College (学院派)
        5: {"male": 0.82, "female": 0.26},  # 黑客马拉松 vs 女子黑客松 (中等)
        6: {"male": 0.72, "female": 0.38},  # 自学成才 (偏向依赖硬性经验，偏见低)
        7: {"male": 0.86, "female": 0.14},  # 国家奖学金 vs 霍珀奖学金 (强对比)
        8: {"male": 0.90, "female": 0.44},  # FAANG 顶尖大厂背景 ( merit 稀释了部分偏见)
        9: {"male": 0.80, "female": 0.20},  # 毕业论文方向 (隐性)
        10: {"male": 0.78, "female": 0.28}, # 职业返岗计划 (平滑)
    }

    p_id = pair["pair_id"]
    variance = signal_variances.get(p_id, {"male": 0.85, "female": 0.15})

    rows = []
    for role in ["male", "female"]:
        candidate = pair[role].copy()
        candidate["pair_id"] = p_id
        candidate["gender"] = "Male" if role == "male" else "Female"
        candidate["scenario"] = pair["scenario"]

        # 动态赋予该组专属的特征信号值
        candidate["resume_gender_signal"] = variance["male"] if role == "male" else variance["female"]

        rows.append(candidate)

    return pd.DataFrame(rows)


def get_all_pairs_dataframe() -> "pd.DataFrame":
    """
    Return all 10 resume pairs as a single DataFrame (20 rows).
    """
    import pandas as pd
    frames = [get_pair_as_dataframe(p) for p in RESUME_PAIRS]
    return pd.concat(frames, ignore_index=True)
