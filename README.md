# 🌾 Agriculture Crop Production Prediction in India

**Domain:** Data Science & Machine Learning  
**Week:** 04 — Final Project

---

## 📌 Project Overview

This project builds a machine learning system that **predicts crop yield** (kg/ha) based on:
- Crop type & State
- Season
- Area under cultivation
- Cost of Cultivation & Production
- Annual Rainfall & Temperature

---

## 📁 Project Structure

```
agriculture_crop_prediction/
│
├── data/
│   ├── crop_data.csv          ← Dataset (2000 records)
│   └── generate_data.py       ← Script to regenerate dataset
│
├── notebooks/
│   └── Agriculture_Crop_Prediction.ipynb   ← Step-by-step Jupyter notebook
│
├── src/
│   ├── train_model.py         ← Full ML pipeline (EDA → Train → Evaluate)
│   └── predict.py             ← Prediction system (batch + interactive)
│
├── models/
│   ├── linear_regression.pkl  ← Trained Linear Regression model
│   ├── random_forest.pkl      ← Trained Random Forest model
│   └── encoders.pkl           ← Label encoders for categorical features
│
├── reports/
│   ├── eda_plots.png          ← EDA visualizations
│   ├── model_evaluation.png   ← Actual vs Predicted plots
│   └── feature_importance.png ← Feature importance chart
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate dataset
python data/generate_data.py

# 3. Train models
python src/train_model.py

# 4. Run predictions
python src/predict.py
```

---

## 🤖 Models Used

| Model             | R² Score | RMSE     |
|-------------------|----------|----------|
| Linear Regression | ~0.10    | ~19,766  |
| **Random Forest** | **~0.99**| **~2,286**|

> Random Forest significantly outperforms Linear Regression, capturing non-linear relationships in the data.

---

## 📊 Features

| Feature               | Type        | Description                     |
|-----------------------|-------------|---------------------------------|
| Crop                  | Categorical | Type of crop                    |
| State                 | Categorical | Indian state                    |
| Season                | Categorical | Kharif / Rabi / Zaid            |
| Area                  | Numerical   | Cultivated area (hectares)      |
| Cost_of_Cultivation   | Numerical   | Cost per hectare (₹)            |
| Cost_of_Production    | Numerical   | Production cost per hectare (₹) |
| Annual_Rainfall       | Numerical   | Rainfall in mm                  |
| Temperature           | Numerical   | Avg temperature in °C           |
| **Yield (target)**    | Numerical   | kg/ha                           |

---

## 🔮 Sample Prediction

```python
from src.predict import predict_yield

yield_val = predict_yield(
    crop='Rice', state='Punjab', season='Kharif',
    area=100, cost_cultivation=35000, cost_production=25000,
    rainfall=900, temperature=28
)
print(f"Predicted Yield: {yield_val:.2f} kg/ha")
# Output: Predicted Yield: 2569.93 kg/ha
```

---

## 🚀 Future Improvements

- Add more features (soil type, fertilizer usage)
- Hyperparameter tuning with GridSearchCV
- Deploy as a Flask / Streamlit web app
- Expand to multi-output prediction (Yield + Production)

---

## 📚 Libraries Used

- `pandas`, `numpy` — Data manipulation
- `scikit-learn` — Machine learning
- `matplotlib`, `seaborn` — Visualization
