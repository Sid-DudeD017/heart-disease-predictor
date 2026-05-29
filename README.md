# Heart Disease Risk Prediction System

An end-to-end machine learning project that predicts the probability of a patient having heart disease. The project transforms raw clinical data through a production-grade automated pipeline and exposes the model via an interactive Streamlit web application.

## 🚀 Live Demo
👉 **[Insert Your Streamlit Cloud Link Here]**

---

## 📊 Project Overview
The objective of this project is to build a reliable screening tool to assess heart disease risk. Instead of just predicting a binary "healthy" or "sick" label, the application outputs the **exact percentage chance of risk**, giving healthcare providers better nuanced insights.

* **Dataset Size:** 918 patient records across 7 core clinical features.
* **Target Optimization:** Converted a complex multi-class target (varying severity levels 1–4) into a clean binary classification problem ($0$ = Low Risk, $1$ = High Risk) to optimize real-world screening utility.

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python
* **Machine Learning Framework:** Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Deployment:** GitHub & Streamlit Cloud

### Production-Grade Data Pipeline
To prevent data leakage and handle messy real-world data seamlessly, the project utilizes a nested **Scikit-Learn Pipeline** paired with a `ColumnTransformer`:

1. **Numerical Pipeline (`age`, `trestbps`, `thalach`):**
   * Handles missing values using `SimpleImputer` (Median strategy).
   * Feature scaling using `StandardScaler`.
2. **Categorical Pipeline (`sex`, `chest pain`, `restecg`, `exang`):**
   * Handles missing values using `SimpleImputer` (Most Frequent strategy).
   * Structural encoding using `OneHotEncoder(handle_unknown='ignore')`.
3. **Model:** Trained using a `RandomForestClassifier` ensemble approach.

---

## 📈 Model Performance
* **Baseline Accuracy:** ~82% to 88%
* **Primary Metric Optimization:** High **Recall** on the positive class ($1$). In a medical context, missing a sick patient (False Negative) is significantly more dangerous than a False Positive. The model is specifically tuned to ensure high sensitivity/recall.

---

## 💻 Local Setup & Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/[YOUR_REPO_NAME].git
   cd [YOUR_REPO_NAME]
