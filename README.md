# Telco Customer Churn Prediction (End-to-End ML Project)

## 📌 Project Overview
This project focuses on predicting customer churn for a telecom company using classical machine learning techniques.  
The goal is not only to build an accurate model, but also to **handle real-world challenges** such as class imbalance, feature engineering, and model explainability.

The project follows a complete **end-to-end ML pipeline**:
- Data understanding & preprocessing
- Feature engineering
- Model training & evaluation
- Handling imbalanced data
- Explainability using SHAP
- Model comparison and selection

---

## 🎯 Business Problem
Customer churn is costly for telecom companies.  
The objective is to **identify customers who are likely to churn** so that proactive retention strategies can be applied.

From a business perspective:
- **Recall for churned customers is more important than accuracy**
- Missing a churner is more expensive than contacting a non-churner

---

## 📂 Dataset
- **Name:** Telco Customer Churn Dataset  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn  

The raw dataset is **not included** in this repository.  
It can be downloaded directly from the source link above.

---

## 🔍 Exploratory Data Analysis (EDA)
Key observations from EDA:
- Strong **class imbalance** (churned customers are the minority)
- Month-to-month contracts have significantly higher churn
- Customers with short tenure are more likely to churn
- Higher monthly charges correlate with higher churn

EDA confirmed that:
- Accuracy alone would be misleading
- Special handling for imbalance is required

---

## 🛠 Data Preprocessing
- Dropped non-informative identifier (`customerID`)
- Fixed missing values in `TotalCharges`
- Converted categorical features to appropriate formats
- Separated numerical and categorical features
- Built a reusable preprocessing pipeline using `ColumnTransformer`

---

## 🧠 Feature Engineering
Feature engineering was done **manually and intentionally**, not automatically.

Examples:
- Contract-based features (e.g., `IsMonthToMonth`)
- Behavioral indicators (e.g., number of internet services)
- Interaction features:
  - `MonthToMonth_HighCharge`
  - `NewCustomer_HighCharge`

These features significantly improved model understanding and performance.

---

## ⚖️ Handling Class Imbalance
This project **does not rely on SMOTE or naive oversampling**.

Instead, two strategies were explored:
1. **Baseline model** (no class weighting)
2. **Cost-sensitive learning** using `scale_pos_weight` in XGBoost

The second approach was selected as it:
- Improved churn recall significantly
- Preserved model stability
- Maintained explainability

---

## 🤖 Modeling
- Algorithm: **XGBoost Classifier**
- Evaluation metrics:
  - ROC-AUC
  - Precision / Recall
  - F1-score

### Baseline Model
- Churn Recall ≈ 0.51
- ROC-AUC ≈ 0.84

### Optimized Model (Class Weighting)
- **Churn Recall ≈ 0.77**
- ROC-AUC ≈ 0.84 (stable)
- Accuracy intentionally deprioritized

This reflects a **business-aware modeling decision**.

---

## 📊 Threshold Optimization
Instead of relying on the default threshold (0.5):
- Precision–Recall tradeoff was analyzed
- Threshold selection aligned with churn business costs
- Focus placed on maximizing recall with acceptable precision

---

## 🔍 Model Explainability (SHAP)
SHAP was used to explain:
- **Global behavior** (most important features)
- **Local decisions** (why a specific customer is predicted to churn)

Key insights:
- Month-to-month contracts strongly increase churn risk
- Longer tenure significantly reduces churn
- High monthly charges amplify churn risk
- Interaction features played a critical role

SHAP explanations remained **stable before and after class weighting**, confirming that:
- Model logic did not change
- Only decision sensitivity was adjusted

---

## 🧾 Model Artifacts
Saved artifacts:
- `xgb_weighted_recall_model.pkl` – final selected model
- `preprocessor.pkl` – preprocessing pipeline

These are included for **demonstration and reproducibility purposes**.

---

## ✅ Key Takeaways
- End-to-end ML pipeline implemented
- Business-driven evaluation strategy
- Thoughtful feature engineering
- Proper handling of class imbalance
- Transparent and explainable model decisions

This project reflects **real-world ML problem solving**, not just metric optimization.

---

## 🧰 Tools & Technologies
- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib / Seaborn


