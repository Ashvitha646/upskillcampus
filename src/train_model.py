"""
Agriculture Crop Production Prediction - Model Training
Author: (Your Name)
Domain: Data Science & Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'crop_data.csv')
df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("  AGRICULTURE CROP PRODUCTION PREDICTION")
print("=" * 55)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn Info:")
print(df.dtypes)

# ─────────────────────────────────────────────
# 2. Data Preprocessing
# ─────────────────────────────────────────────
print("\n[2] Preprocessing...")
df.columns = df.columns.str.strip().str.replace(' ', '_')
df.dropna(inplace=True)
print(f"    Missing values after cleaning: {df.isnull().sum().sum()}")

le_crop  = LabelEncoder()
le_state = LabelEncoder()
le_season = LabelEncoder()

df['Crop_encoded']   = le_crop.fit_transform(df['Crop'])
df['State_encoded']  = le_state.fit_transform(df['State'])
df['Season_encoded'] = le_season.fit_transform(df['Season'])

# ─────────────────────────────────────────────
# 3. EDA — save plots
# ─────────────────────────────────────────────
print("\n[3] Generating EDA plots...")
os.makedirs('../reports', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Agriculture Crop Production — EDA', fontsize=16, fontweight='bold')

# Yield distribution
axes[0, 0].hist(df['Yield'], bins=40, color='steelblue', edgecolor='white')
axes[0, 0].set_title('Yield Distribution')
axes[0, 0].set_xlabel('Yield (kg/ha)')
axes[0, 0].set_ylabel('Frequency')

# Average yield by crop
crop_yield = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False)
axes[0, 1].barh(crop_yield.index, crop_yield.values, color='seagreen')
axes[0, 1].set_title('Average Yield by Crop')
axes[0, 1].set_xlabel('Avg Yield (kg/ha)')

# Cost vs Yield scatter
axes[1, 0].scatter(df['Cost_of_Cultivation'], df['Yield'], alpha=0.3, color='coral', s=10)
axes[1, 0].set_title('Cost of Cultivation vs Yield')
axes[1, 0].set_xlabel('Cost of Cultivation (₹)')
axes[1, 0].set_ylabel('Yield (kg/ha)')

# Correlation heatmap
num_cols = ['Cost_of_Cultivation', 'Cost_of_Production', 'Annual_Rainfall', 'Temperature', 'Yield', 'Area']
corr = df[num_cols].corr()
im = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(num_cols)))
axes[1, 1].set_yticks(range(len(num_cols)))
axes[1, 1].set_xticklabels(num_cols, rotation=45, ha='right', fontsize=8)
axes[1, 1].set_yticklabels(num_cols, fontsize=8)
axes[1, 1].set_title('Feature Correlation Heatmap')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('../reports/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("    EDA plots saved → reports/eda_plots.png")

# ─────────────────────────────────────────────
# 4. Feature / Target Split
# ─────────────────────────────────────────────
features = ['Crop_encoded', 'State_encoded', 'Season_encoded',
            'Area', 'Cost_of_Cultivation', 'Cost_of_Production',
            'Annual_Rainfall', 'Temperature']
target = 'Yield'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n[4] Train/Test split — Train: {len(X_train)}, Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 5. Train Models
# ─────────────────────────────────────────────
print("\n[5] Training models...")

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ─────────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────────
print("\n[6] Model Evaluation:")
results = {
    'Linear Regression': (y_pred_lr, lr),
    'Random Forest':     (y_pred_rf, rf),
}

best_model, best_r2 = None, -999
for name, (preds, model) in results.items():
    r2  = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    print(f"    {name:25s} | R²: {r2:.4f} | MSE: {mse:,.0f} | RMSE: {rmse:,.0f}")
    if r2 > best_r2:
        best_r2, best_model = r2, model

print(f"\n    Best model: Random Forest (R²={best_r2:.4f})")

# ─────────────────────────────────────────────
# 7. Evaluation plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Evaluation — Actual vs Predicted', fontsize=14, fontweight='bold')

for ax, (name, (preds, _)) in zip(axes, results.items()):
    ax.scatter(y_test, preds, alpha=0.3, s=10, color='steelblue')
    lims = [y_test.min(), y_test.max()]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel('Actual Yield')
    ax.set_ylabel('Predicted Yield')
    ax.set_title(f'{name}\nR² = {r2_score(y_test, preds):.4f}')
    ax.legend()

plt.tight_layout()
plt.savefig('../reports/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n    Evaluation plot saved → reports/model_evaluation.png")

# ─────────────────────────────────────────────
# 8. Feature Importance (RF)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
importances = pd.Series(rf.feature_importances_, index=features).sort_values()
importances.plot.barh(ax=ax, color='darkorange')
ax.set_title('Feature Importances — Random Forest')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('../reports/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Feature importance plot saved → reports/feature_importance.png")

# ─────────────────────────────────────────────
# 9. Save Models & Encoders
# ─────────────────────────────────────────────
os.makedirs('../models', exist_ok=True)
with open('../models/linear_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open('../models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('../models/encoders.pkl', 'wb') as f:
    pickle.dump({'crop': le_crop, 'state': le_state, 'season': le_season}, f)

print("\n[7] Models saved → models/")

# ─────────────────────────────────────────────
# 10. Sample Prediction
# ─────────────────────────────────────────────
print("\n[8] Sample Prediction:")
sample_input = {
    'Crop': 'Rice', 'State': 'Punjab', 'Season': 'Kharif',
    'Area': 100, 'Cost_of_Cultivation': 35000,
    'Cost_of_Production': 25000, 'Annual_Rainfall': 900, 'Temperature': 28
}
enc_input = [[
    le_crop.transform([sample_input['Crop']])[0],
    le_state.transform([sample_input['State']])[0],
    le_season.transform([sample_input['Season']])[0],
    sample_input['Area'],
    sample_input['Cost_of_Cultivation'],
    sample_input['Cost_of_Production'],
    sample_input['Annual_Rainfall'],
    sample_input['Temperature']
]]
pred = rf.predict(enc_input)[0]
print(f"    Input : {sample_input}")
print(f"    Predicted Yield : {pred:.2f} kg/ha")
print("\n✅ Training complete.")
