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

# How to Test

## Setup Environment:

### It's highly recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```bash
pip install -r requirements.txt
```

## Run the ETL pipeline
```bash
python src/pipeline.py
```

## Run the model training script
```bash
python src/train.py
```

## Launch the Web Application
```bash
streamlit run app/main.py
```
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
```
