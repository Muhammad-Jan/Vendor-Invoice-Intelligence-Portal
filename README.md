# Vendor Invoice Intelligence Portal

### AI-Powered Freight Cost Prediction & Vendor Invoice Approval Flagging

---

# 📌 Table of Contents

- [Project Overview](#project-overview)
- [Business Objectives](#business-objectives)
- [Project Structure](#project-structure)
- [Freight Cost Prediction Module](#freight-cost-prediction-module)
- [Vendor Invoice Approval Flagging Module](#vendor-invoice-approval-flagging-module)
- [Inference Pipeline](#inference-pipeline)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Application](#application)
- [Requirements](#requirements)
- [Sample Test Cases](#sample-test-cases)
- [Author & Contact](#author--contact)

---

# 📌 Project Overview

This project implements an end-to-end machine learning portal designed to support finance and procurement teams by:

1. Predicting the expected freight cost for vendor invoices.
2. Automatically identifying vendor invoices that require manual approval based on financial and operational patterns.

---

# 🎯 Business Objectives

## 1. Freight Cost Prediction (Regression)

**Objective**

Predict the expected freight cost for a vendor invoice using business-related features such as quantity, invoice value, and historical purchasing patterns.

**Why it matters**

- Improve freight cost estimation for better budgeting.
- Support procurement planning and vendor negotiations.
- Reduce unexpected transportation expenses.
- Enable data-driven financial decision-making.

<p align="center">
  <img src="images/Freight%20Cost%20Prediction.png" alt="Freight Cost Prediction" width="900">
</p>

---

## 2. Vendor Invoice Approval Flagging (Classification)

**Objective**

Predict whether a vendor invoice should be approved automatically or flagged for manual review based on invoice characteristics and operational patterns.

**Why it matters**

- Detect potentially risky or abnormal invoices.
- Reduce manual verification efforts.
- Improve approval efficiency.
- Minimize financial risks caused by incorrect invoice processing.

<p align="center">
  <img src="images/Vendor%20Invoice%20Approval%20Flag.png" alt="Vendor Invoice Approval Flagging" width="900">
</p>

## 📂 Data Sources

All project data is stored in a relational **SQLite** database (`inventory.db`). The database consists of the following tables:

| Table | Purpose |
|--------|----------|
| `vendor_invoice` | Stores invoice-level financial and timing information. |
| `purchases` | Contains item-level purchase records. |
| `purchase_prices` | Stores reference purchase prices. |
| `begin_inventory` | Beginning inventory snapshot. |
| `end_inventory` | Ending inventory snapshot. |

To build the machine learning dataset, invoice-level features are generated using **SQL aggregations** across these tables.

---

## 📊 Exploratory Data Analysis (EDA)

The EDA phase is designed to answer important business questions such as:

- Are flagged invoices associated with higher financial exposure?
- Does freight cost increase linearly with purchase quantity?
- Is freight cost primarily dependent on quantity?

Statistical hypothesis testing (**Independent t-tests**) is used to validate whether flagged invoices are significantly different from normal invoices.

## 🤖 Models Used

### 📈 Regression (Freight Cost Prediction)

The following regression models were trained and evaluated:

- **Linear Regression** *(Baseline Model)*
- **Decision Tree Regressor**
- **Random Forest Regressor** *(Final Selected Model)*

The **Random Forest Regressor** achieved the best performance and was selected as the final model for freight cost prediction.

---

### 🚩 Classification (Invoice Flagging)

The following classification models were trained and evaluated:

- **Logistic Regression** *(Baseline Model)*
- **Decision Tree Classifier**
- **Random Forest Classifier** *(Final Selected Model)*

Hyperparameter tuning was performed using **GridSearchCV**, with the **F1-score** as the optimization metric to effectively handle class imbalance. The tuned **Random Forest Classifier** delivered the best overall performance and was selected as the final classification model.

## 📈 Evaluation Metrics

### Freight Cost Prediction

The regression models were evaluated using the following performance metrics:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

These metrics measure prediction accuracy and quantify the difference between actual and predicted freight costs.

---

### 🚩 Invoice Flagging

The classification models were evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Classification Report**
- **Feature Importance Analysis**

These metrics provide a comprehensive evaluation of the model's classification performance and help identify the most influential features used for invoice flagging.

---

## 🔗 End-to-End Application

A **Streamlit** application demonstrates the complete machine learning pipeline by allowing users to:

- Enter invoice details
- Predict the expected freight cost
- Flag suspicious invoices in real time
- Display human-readable prediction explanations

The application integrates data preprocessing, trained machine learning models, and an interactive user interface into a single end-to-end solution.
