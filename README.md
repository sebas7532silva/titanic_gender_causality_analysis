# 🎓 Titanic Gender-Based Survival Analysis with T-Learner

This project explores the relationship between gender and survival probability on the Titanic by applying a **causal inference framework** — specifically, the **T-Learner** methodology. By training separate machine learning models for men and women, we estimate the **Average Treatment Effect (ATE)** of gender on survival outcomes.

---

## 📌 Project Objectives

- Train two classification models: **Random Forest** and **Logistic Regression**
- Use **SMOTE** to handle class imbalance separately for male and female training datasets
- Predict survival outcomes using models trained on gender-specific data
- Compute the **ATE** by evaluating how much better individuals perform under the model trained on their gender vs. the other
- Analyze whether gender had a significant impact on predicted survival rates

---

## ⚙️ Technologies & Tools

- **Python Notebooks**
- **Libraries used:**
  - `pandas` — data manipulation
  - `numpy` — numerical operations
  - `scikit-learn` — modeling, preprocessing, SMOTE, and pipelines
  - `matplotlib` & `seaborn` — data visualization
  - `IterativeImputer` — to handle missing values

---

## 📂 Project Structure

```
project-root/
│
├── Data/
│   ├── Train/                 # Training data split by gender
│   │
│   ├── Test/                  # Testing data split by gender
│   │
│   └── Titanic-Dataset.csv       # Original Titanic dataset
│
├── Notebooks/
│   ├── ExploratoryAnalysis.ipynb      # Initial EDA: distributions, correlations, etc.
│   ├── Preprocessing.ipynb             # Imputation, scaling, SMOTE
│   └── Models.ipynb     # Model training, predictions, and ATE computation
│
├── dictionary_columns.txt     # Definitions and descriptions of each dataset column
│
└── README.md                  # You are here 🚀
```

---

## 📈 Summary of Results

Two models were implemented — Random Forest and Logistic Regression — each trained separately on male and female subsets of the Titanic dataset. The **T-Learner** was used to estimate the **Average Treatment Effect (ATE)** of gender on survival.

- **Random Forest ATE:** -0.0076
- **Logistic Regression ATE:** -0.0185

Both ATE values are negative, meaning that on average, individuals were predicted to have slightly higher survival chances under the **female-trained models**, though the differences were **not large**.

---

## 🔍 Conclusion

This project illustrates how gender may subtly influence survival predictions on the Titanic. By using a T-Learner setup, we can quantify these effects and explore model biases or historical patterns such as the "women and children first" evacuation protocol. The analysis highlights the power of combining causal inference techniques with traditional machine learning to gain deeper insights from classic datasets.
