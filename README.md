# AI Recruitment Ethics & Bias Project

**Business Ethics & CSR University Project**
Inspired by the Amazon AI Recruitment Bias Case (Reuters, 2018)

---

## What This Project Does

This project demonstrates how gender bias emerges in AI recruitment systems and how fairness-aware design can reduce it. It includes:

- A **Jupyter Notebook** — full academic research paper with code, analysis, and ethical discussion (for submission)
- A **Streamlit web app** — interactive live demonstration dashboard (for presentations)

---

## Project Structure

```
ai_recruitment_ethics/
│
├── app.py                   ← Streamlit app entry point (run this for live demo)
├── generate_notebook.py     ← Generates the Jupyter notebook
├── requirements.txt         ← Python package dependencies
│
├── data_generator.py        ← Creates synthetic hiring dataset
├── resume_pairs.py          ← 10 matched resume pairs for gender experiment
├── models.py                ← Biased and fairness-aware model definitions
├── fairness_audit.py        ← All fairness metrics with ethical annotations
├── explainability.py        ← SHAP feature importance
├── audit_log.py             ← Governance audit logging
│
├── app/pages/
│   ├── page_home.py         ← Ethical background & Amazon case
│   ├── page_evaluation.py   ← Resume pairs + custom candidate evaluation
│   ├── page_fairness.py     ← Fairness audit dashboard
│   ├── page_comparison.py   ← Model comparison & tradeoff analysis
│   └── page_audit.py        ← Governance log & accountability
│
├── ethics/
│   └── ethics_discussion.md ← Standalone written ethical analysis
│
├── notebook/
│   └── ai_recruitment_ethics.ipynb  ← Generated Jupyter Notebook
│
├── data/
│   └── audit_log.json       ← Auto-created when candidates are evaluated
│
└── outputs/                 ← Auto-created: saved figures and CSV summaries
```

---

## Setup Instructions (Beginner-Friendly)

### Step 1: Make sure Python is installed
Open a terminal (Mac: Terminal app; Windows: Command Prompt or PowerShell) and type:
```bash
python --version
```
You need Python 3.9 or higher.

### Step 2: Install required packages
In the terminal, navigate to the project folder, then run:
```bash
pip install -r requirements.txt
```
This will take 2–5 minutes. Let it finish completely.

### Step 3: Create the data and outputs folders
```bash
mkdir -p data outputs
```

### Step 4: Generate the Jupyter Notebook
```bash
python generate_notebook.py
```
This creates `notebook/ai_recruitment_ethics.ipynb`.

### Step 5: Open the Jupyter Notebook (for submission)
```bash
jupyter notebook notebook/ai_recruitment_ethics.ipynb
```
Then run all cells: **Kernel → Restart & Run All**

### Step 6: Launch the Streamlit app (for live demo)
```bash
streamlit run app.py
```
A browser window will open automatically at `http://localhost:8501`

---

## How to Use the Streamlit App

1. **Start on the Home page** — read the ethical background and Amazon case summary
2. **Go to Fairness Audit Dashboard** — click "Train Both Models" to initialize everything
3. **Go to Candidate Evaluation** — explore the 10 matched resume pairs and their score gaps
4. **Go to Model Comparison** — see the accuracy vs. fairness tradeoff chart
5. **Go to Audit Log** — see the governance overview and checklist

> **Important:** Always train the models on the Fairness Audit page FIRST before using other pages.

---

## Key Ethical Disclaimer

All data in this project is **synthetic** — generated programmatically for educational purposes. No real candidates, real resumes, or real hiring decisions are involved.

The bias in the dataset is **intentional and fully documented**. It simulates the kind of historically biased data that Amazon's AI learned from. This is the point: we build the bias in explicitly so we can measure and demonstrate it.

The Amazon case is described using publicly reported facts from the Reuters investigation (October 2018).

---

## What You Can Claim in Your Presentation

✅ "Our fairness-aware model reduces measurable gender bias across multiple metrics"
✅ "The fairness tradeoff (slightly lower accuracy) is documented and ethically justified"
✅ "We demonstrate proxy discrimination — bias persists even without gender as an input"
✅ "Our system includes human oversight and audit logging per EU AI Act principles"

❌ Do NOT claim: "Our model is bias-free"
❌ Do NOT claim: "We have solved gender bias in AI hiring"

Intellectual honesty about limitations strengthens, not weakens, your academic argument.

---

## Libraries Used

| Library | Purpose |
|---|---|
| scikit-learn | Machine learning models |
| Fairlearn (Microsoft Research) | Fairness constraints and metrics |
| SHAP | Explainability (SHapley Additive exPlanations) |
| Streamlit | Web application framework |
| Pandas / NumPy | Data manipulation |
| Matplotlib | Visualizations |
