# Credit Risk Scoring System

An end-to-end machine learning and decision-support application for predicting credit-card default risk using customer financial and repayment information.

The project covers the complete workflow from data preparation and model development to probability calibration, threshold selection, model evaluation, risk segmentation, and deployment through an interactive Streamlit web application.

---

## 📌 Project Overview

Credit default prediction is an important problem in financial risk management. The objective of this project is to develop a machine-learning system that estimates the probability that a customer will default on their next credit-card payment.

The system uses the **UCI Default of Credit Card Clients dataset** and develops a binary classification model to predict whether a customer is likely to default.

Rather than stopping at model training, this project implements a complete workflow:

```text
Customer Data
     ↓
Data Preprocessing
     ↓
Model Development & Comparison
     ↓
Final XGBoost Model
     ↓
Probability Prediction
     ↓
Calibration / Threshold Selection
     ↓
Final Test Evaluation
     ↓
Risk Classification
     ↓
Interactive Streamlit Application
```

The resulting application provides individual customer prediction, customer exploration, segment-level analysis, portfolio-level risk analysis, and model methodology information.

---

## 🎯 Project Objectives

The main objectives of the project are to:

* Develop a machine-learning model for credit-card default prediction.
* Compare different classification approaches and establish a baseline.
* Select an appropriate final model based on the project evaluation.
* Generate probabilities of default rather than only binary predictions.
* Select a classification threshold using a separate calibration dataset.
* Evaluate the final model on an untouched test set.
* Develop interpretable customer risk categories.
* Provide customer-level and portfolio-level risk analysis.
* Build an interactive web application using Streamlit.
* Demonstrate how machine-learning predictions can support credit-risk decision making.

---

## 📊 Dataset

The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

The dataset contains customer demographic, credit-limit, repayment-status, billing-amount, and previous-payment information.

The model uses **23 predictor variables**.

### Main Feature Groups

| Feature Group           | Description                                   |
| ----------------------- | --------------------------------------------- |
| Credit information      | Customer credit limit                         |
| Demographic information | Gender, education, marital status and age     |
| Repayment history       | Repayment status over the previous six months |
| Bill statements         | Monthly billed amounts                        |
| Previous payments       | Monthly payment amounts                       |

The target variable represents whether the customer defaulted on the following month's payment.

> The application presents categorical variables in a more understandable form rather than exposing raw dataset codes directly to the user.

---

## 🤖 Machine Learning Approach

### Model Development

The project includes model development and comparison before selecting the final model.

A **Logistic Regression model** was used as a baseline because it provides a simple and interpretable reference point for binary classification.

Tree-based machine-learning approaches were also evaluated, including the final **XGBoost** model.

The final application uses the trained XGBoost pipeline.

### Why XGBoost?

XGBoost was selected as the final modelling approach following the project's model comparison and evaluation process.

The main reason for using a tree-based boosting model is its ability to capture non-linear relationships and interactions between customer credit characteristics and repayment behaviour.

The final XGBoost model is integrated with the preprocessing pipeline and saved for use by the frontend application.

---

## 🔬 Data Split and Evaluation Strategy

A key part of the project is keeping threshold selection separate from final model evaluation.

The available modelling data were divided into:

| Dataset     | Observations | Purpose             |
| ----------- | -----------: | ------------------- |
| Training    |       19,200 | Model training      |
| Calibration |        4,800 | Threshold selection |
| Test        |        6,000 | Final evaluation    |

The calibration dataset was used to determine the operating classification threshold.

The test dataset remained untouched until the final evaluation.

This prevents the test set from influencing threshold selection and provides a more appropriate estimate of final model performance.

---

## 🎚️ Classification Threshold

The final classification threshold selected during calibration was:

**0.31 (31%)**

The model first produces a probability of default.

The probability is then compared with the frozen threshold:

```text
Predicted probability < 0.31
        → Non-Default

Predicted probability ≥ 0.31
        → Default
```

The application also uses probability bands to provide more intuitive risk categories.

| Predicted Probability | Risk Category  |
| --------------------: | -------------- |
|                 < 20% | Low Risk       |
|           20% – < 31% | Medium Risk    |
|           31% – < 50% | High Risk      |
|                 ≥ 50% | Very High Risk |

The 31% value is a classification threshold selected during model evaluation. It should not be interpreted as a guarantee that a customer has exactly a 31% chance of default.

---

## 📈 Final Test Performance

The final model was evaluated on the untouched 6,000-observation test set.

| Metric                   |   Result |
| ------------------------ | -------: |
| Classification Threshold |     0.31 |
| Accuracy                 | 0.797333 |
| Balanced Accuracy        | 0.680497 |
| Precision                | 0.548727 |
| Recall                   | 0.470987 |
| F1 Score                 | 0.506894 |
| ROC-AUC                  | 0.761583 |
| PR-AUC                   | 0.526168 |
| Brier Score              | 0.139615 |
| Log Loss                 | 0.443703 |

### Final Confusion Matrix

|                        | Predicted Non-Default | Predicted Default |
| ---------------------- | --------------------: | ----------------: |
| **Actual Non-Default** |                 4,159 |               514 |
| **Actual Default**     |                   702 |               625 |

Total test observations:

**6,000**

---

## 🧠 Model Interpretation

The project also examines feature importance from the final model.

The most influential feature in the final feature-importance output is:

**PAY_0 — most recent repayment status**

Other important predictors include repayment-history variables, credit limit, previous payment amounts, and bill amounts.

This highlights the importance of recent repayment behaviour within the model's predictions.

Feature importance is used as a model-analysis tool and should not be interpreted as proof that a feature independently causes default.

---

# 🌐 Web Application

The trained model is integrated into an interactive **Streamlit** application.

The application provides several pages.

### 1. Dashboard

Provides an overall view of the demonstration customer portfolio and predicted risk distribution.

---

### 2. Risk Prediction

Allows the user to enter information for an individual customer and obtain:

* Predicted probability of default
* Risk category
* Classification based on the 31% threshold
* Supporting risk information

The interface converts raw dataset categories into more understandable user-facing descriptions.

---

### 3. Customer Risk Explorer

Allows users to explore the available demonstration customer records.

Customers can be filtered by:

* Low Risk
* Medium Risk
* High Risk
* Very High Risk

The page displays customer information together with predicted probability and risk classification.

---

### 4. Customer Segment Risk Analysis

This page compares predicted risk across different customer segments.

Available segmentation options include:

* Age Group
* Education
* Marital Status
* Credit Limit Group
* Risk Level

For each segment, the application calculates:

* Number of customers
* Average predicted default probability
* Percentage of customers at or above the 31% threshold

The application also provides visual comparisons using interactive charts.

Importantly, differences between customer groups are presented as differences in model-predicted risk and are not interpreted as evidence that a particular demographic characteristic causes default.

---

### 5. Portfolio Risk & Recommendations

This page provides portfolio-level analysis.

It includes:

* Total customer count
* Average predicted default probability
* High-risk customer proportion
* Risk distribution
* Probability distributions by risk group
* Demonstration expected-loss calculation
* General risk-management recommendations

### Expected Loss Demonstration

The application demonstrates the conceptual relationship:

```text
Expected Loss = PD × LGD × EAD
```

Where:

* **PD** = Probability of Default
* **LGD** = Loss Given Default
* **EAD** = Exposure at Default

LGD is provided as an adjustable assumption in the prototype rather than being estimated from a real bank recovery database.

Similarly, EAD is demonstrated using an adjustable fraction of the customer's credit limit.

Therefore, the expected-loss output is a **demonstration metric**, not a regulatory or production banking calculation.

---

### 6. Model & Methodology

This page explains:

* The XGBoost model
* Prediction workflow
* Classification threshold
* Risk categories
* Model input features
* Important interpretation considerations
* Prototype limitations

The application loads the already-trained model and does not retrain the model when Streamlit starts.

---

# 🏗️ System Architecture

The application follows a simple machine-learning deployment architecture:

```text
                    ┌─────────────────────┐
                    │   Customer Input    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Saved Preprocessing │
                    │     Pipeline        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      XGBoost        │
                    │   Classification    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Probability of      │
                    │ Default (PD)        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ 0.31 Classification │
                    │      Threshold      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Risk Classification │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       Customer Risk     Segment Analysis   Portfolio
          Explorer                           Analysis
```

---

# 📁 Project Structure

A simplified structure of the project is:

```text
Credit-Risk-Scoring-System/
│
├── frontend/
│   ├── app.py
│   ├── pages/
│   │   ├── ...
│   ├── utils/
│   │   └── risk_utils.py
│   └── ...
│
├── notebooks/
│   ├── ...
│   └── outputs/
│       └── final_artifacts/
│
├── data/
│   └── ...
│
├── README.md
└── requirements.txt
```

The exact files may vary depending on the project environment.

---

# 🧪 Project Verification

A dedicated final audit was performed to verify the completed project.

The audit checked:

* Dataset and split sizes
* Predictor dimensions
* Probability alignment
* Probability ranges
* Calibration/test separation
* Threshold selection
* Frozen final threshold
* Test-set evaluation
* Confusion-matrix consistency
* Metric consistency
* Feature-importance output
* Risk-group output
* Required project artifacts
* Reproducibility

The final audit confirmed that the reported test predictions and evaluation outputs were internally consistent.

---

# ⚠️ Limitations

This project is an academic machine-learning prototype and should not be interpreted as a production banking credit-decision system.

Important limitations include:

* The model is trained on a historical dataset.
* The application uses demonstration customer data rather than a live banking database.
* Model performance may change when applied to a different customer population.
* Predicted probabilities depend on the characteristics and quality of the training data.
* The expected-loss calculation uses simplified assumptions.
* The application does not implement production authentication or authorization.
* Production deployment would require appropriate security and privacy controls.
* Real financial institutions require additional validation, monitoring, governance, fairness assessment, and regulatory controls.
* Model predictions should support human review rather than automatically determine customer outcomes.

---

# 🚀 Future Improvements

Possible future improvements include:

* Integration with an authorized production customer database.
* Automated model monitoring and drift detection.
* Periodic model retraining using newly validated data.
* More extensive probability calibration.
* Model explainability using techniques such as SHAP.
* More detailed customer-level explanations.
* Authentication and role-based access control.
* Secure API-based model serving.
* Comprehensive audit logging.
* Fairness and subgroup-performance analysis.
* Production-grade expected-loss modelling using institution-specific PD, LGD and EAD methodologies.
* Containerized deployment and cloud infrastructure.

---

# 💻 Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Plotly**
* **Streamlit**
* **Joblib**
* **Jupyter Notebook**

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Credit-Risk-Scoring-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

From the project directory, run the Streamlit application using the entry-point file in the frontend directory.

For example:

```bash
streamlit run frontend/app.py
```

The exact command may need to be adjusted if your project's Streamlit entry point has a different filename.

---

# 🔐 Data and Model Files

The repository should not contain confidential customer information, credentials, API keys, or other sensitive data.

The application is designed around the project's trained model and demonstration data.

If large model/data files are excluded from GitHub, provide appropriate instructions for obtaining or generating them.

---

# 📚 Dataset Reference

**Yeh, I.-C., & Lien, C.-H. (2009).** The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480.

Dataset:

**Default of Credit Card Clients**

UCI Machine Learning Repository.

---

# 📖 Academic Context

This repository contains the implementation and supporting materials for an academic credit-risk machine-learning project.

The project demonstrates the integration of:

```text
Machine Learning
      +
Probability Prediction
      +
Threshold Calibration
      +
Risk Segmentation
      +
Portfolio Analysis
      +
Interactive Web Application
```

The system is intended for educational and demonstration purposes.


---

## ⭐ Project Summary

This project demonstrates an end-to-end credit-risk scoring workflow using machine learning.

The final system combines an XGBoost classification pipeline with probability-based risk assessment, calibration-based threshold selection, customer segmentation, portfolio analysis, and an interactive Streamlit frontend.

The goal is to demonstrate how a predictive machine-learning model can be transformed into a practical **credit-risk decision-support prototype** rather than simply producing a model evaluation notebook.
