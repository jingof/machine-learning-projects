# Real-Time Credit Risk & Bias Mitigation System

This project is a full-stack data science application designed to assess credit risk in real-time while actively identifying, quantifying, and mitigating algorithmic bias. The goal is to create a system that is both accurate in its predictions and fair in its outcomes, with transparent, explainable results.

This system is built as a portfolio project to demonstrate advanced skills in data engineering, MLOps, explainable AI, and ethical AI development, particularly relevant for applications in the U.S. financial sector.

## Key Features 🚀

* **ETL Pipeline:** A data processing pipeline built with Pandas to clean and prepare data for modeling.
* **Dual Model Training:** Trains two models:
    1.  A baseline `XGBoost` model optimized purely for accuracy.
    2.  A fairness-aware model using the `Fairlearn` library's `ExponentiatedGradient` technique to mitigate bias.
* **Bias Analysis:** Measures and compares fairness using metrics like Demographic Parity and Equalized Odds.
* **Explainable AI (XAI):** Uses `SHAP` to generate local, feature-level explanations for every prediction, making the model's decisions transparent.
* **Interactive Dashboard:** A `Streamlit` web application provides a user-friendly interface for loan officers to score new applicants and understand the model's reasoning.

## Project Structure
Of course. Here is the complete code for the core files of the Real-Time Credit Risk Assessment and Bias Mitigation System.

This code is a comprehensive template. To make it run, you'll need to create a mock data file as described in the README.md and install the libraries from requirements.txt.

README.md
This is the most important file for your GitHub repository. It explains the project's purpose, structure, and how to run it.

Markdown

# Real-Time Credit Risk & Bias Mitigation System

This project is a full-stack data science application designed to assess credit risk in real-time while actively identifying, quantifying, and mitigating algorithmic bias. The goal is to create a system that is both accurate in its predictions and fair in its outcomes, with transparent, explainable results.

This system is built as a portfolio project to demonstrate advanced skills in data engineering, MLOps, explainable AI, and ethical AI development, particularly relevant for applications in the U.S. financial sector.

/fair-lending-ai/
|
├── data/
│   ├── raw/  
│   └── processed/
|
├── notebooks/
|
├── src/
│   ├── init.py
│   ├── pipeline.py
│   ├── train.py
│   ├── fairness.py
│   └── predict.py
|
├── app/
│   └── main.py
|
├── models/
|
├── requirements.txt
└── README.md

# How to Run

**1. Create Mock Data:**

Create a file named `mock_loan_data.csv` inside the `data/raw/` directory with the following content:

```csv
credit_score,annual_income,loan_amount,loan_term_months,race,loan_status
650,55000,25000,36,GroupA,1
720,85000,35000,60,GroupB,0
580,40000,15000,36,GroupA,1
780,120000,50000,24,GroupB,0
690,62000,10000,48,GroupA,0
640,48000,20000,60,GroupA,1
750,110000,40000,36,GroupB,0
710,75000,18000,48,GroupA,0
660,59000,22000,36,GroupA,0
590,42000,30000,60,GroupB,1
790,150000,80000,48,GroupB,0
620,35000,12000,36,GroupB,1
```
# Setup Environment:

# It's highly recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt


# Run the ETL pipeline
python src/pipeline.py

# Run the model training script
python src/train.py

# Launch the Web Application
streamlit run app/main.py

---

### **`requirements.txt`**

This file lists all project dependencies.

```text
pandas
scikit-learn
xgboost
fairlearn
shap
streamlit
matplotlib