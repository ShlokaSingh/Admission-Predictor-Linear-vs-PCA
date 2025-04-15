# 🎓 Admission Predictor: Linear Models vs. PCA-Based Forecasting

This project explores the use of machine learning to predict a student's likelihood of graduate school admission. The focus is on comparing models trained on original features vs. models trained on reduced features using **Principal Component Analysis (PCA)**.

> 💡 Focus: Regression Modeling · Dimensionality Reduction · PCA  
> 🛠 Tools: Python, scikit-learn, pandas, Jupyter Notebook  
> 📊 Models: Linear Regression · Support Vector Regression (SVR) · PCA

---

## 🔍 Objective

- Predict the probability of graduate school admission using machine learning
- Analyze feature distributions and potential correlations with admission outcomes
- Train and evaluate models using Linear Regression, SVR, and Decision Tree Regressors
- Apply PCA to reduce dimensionality and retrain all models
- Compare performance using Mean Squared Error (MSE) and interpret results visually

---

## 🧠 Methods Used

- 📊 **EDA** – Feature distribution analysis, outlier detection, correlation heatmaps  
- 🔁 **Train-Test Split (80/20)**  
- 📈 **Linear Regression, SVR, and Decision Tree Regressors**  
- 🔍 **PCA** – Reduced features to 2D for alternate model training  
- 📉 **MSE** – Used to compare model performance  
- 📐 **2D Decision Boundary Plot** – Visualized PCA-based predictions

---

## 📊 Model Performance (MSE)

| Model            | Features           | Mean Squared Error |
|------------------|--------------------|---------------------|
| Linear Regression| Original Features  | **0.0053**          |
| Linear Regression| PCA Features       | **0.0077**          |
| SVR              | Original Features  | **0.0093**          |
| SVR              | PCA Features       | **0.0085**          |
| Decision Tree    | Original Features  | **0.0096**          |
| Decision Tree    | PCA Features       | **0.0131**          |

> 📌 *Linear Regression on original features performed best. PCA-reduced models offered simpler visualization but slightly higher error.*

---

## 👤 Author

**Shloka Singh**  
Data Scientist | Focused on real-world ML workflows  
🔗 [LinkedIn](https://www.linkedin.com/in/shloka-singh/)
