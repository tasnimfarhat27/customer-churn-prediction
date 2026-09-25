# 🛡️ End-to-End Customer Churn Prediction System

An interactive, modular machine learning web application built with **Streamlit** to predict banking customer churn. This project integrates a robust ensemble classification pipeline with a custom-styled "Pastel Corporate" UI for real-time risk assessment and proactive retention strategies.

---

## 📊 Project Overview
Customer churn is a critical challenge in the banking sector. This application analyzes 14 distinct features spanning customer demographics, financial activity, and satisfaction metrics to forecast whether a customer is likely to close their account.

### Key Pillars:
1. **Customer Profile (Demographics):** Credit Score, Age, Tenure, Geography, and Gender.
2. **Behavior & Finances:** Account Balance, Number of Products, Credit Card Ownership, Member Activity, and Estimated Salary.
3. **Satisfaction & Feedback:** Recent Complaints, Satisfaction Scores, Loyalty Card Types, and Points Earned.

---

## 🚀 Tech Stack & Libraries
* **Frontend/UI:** Streamlit (Custom CSS styling with a pastel teal/mint and warm gray theme)
* **Machine Learning:** Scikit-learn (Preprocessing, scaling, and classification pipeline)
* **Model Persistence:** Joblib (`.pkl` model artifacts)
* **Data Manipulation:** Pandas, NumPy

---

## 📂 Project Structure
```text
Customer_Churn_Project/
│
├── data/
│   └── Customer-Churn-Records.csv    # Raw dataset
│
├── models/
│   ├── best_churn_model.pkl          # Trained classification model
│   ├── scaler.pkl                    # Feature standard scaler
│   └── feature_columns.pkl           # Saved feature schema list
│
├── app.py                            # Streamlit dashboard application
├── churn_project.py                  # Model training and pipeline script
└── README.md                         # Project documentation