# Machine-Learning-Model-to-predict-Loan-Defaulters
# 🏦 Loan Default Prediction App

This is a Streamlit-based web application that predicts the likelihood of a loan default using machine learning.

### 📌 Features

- Predicts if a loan will be fully paid or default
- Trained using a Random Forest Classifier on real-world loan data
- Interactive input form for credit policy, FICO score, interest rate, etc.
- Displays clear results: ✅ Fully Paid or ❌ Likely to Default
- Includes branding with a logo and creator credit

### 📊 Model

- Model: Random Forest Classifier (`sklearn`)
- Preprocessed with `pd.get_dummies` to handle categorical variables
- Saved with `joblib.dump((model, feature_names), "loan_default_model.joblib")`

### 🛠 Tech Stack

- Python 🐍
- Streamlit 🚀
- Scikit-learn
- Pandas
- Joblib
- VS Code / Jupyter

### 📁 Folder Structure
loan-default-prediction-app/
├── app.py
├── loan_default_model.joblib
├── images/
│ └── logo.JPG
├── requirements.txt
└── README.md

👨‍💻 Created by
Ebenezer Kwaw
LinkedIn | GitHub

