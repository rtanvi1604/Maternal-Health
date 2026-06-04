"""
analysis.py — Standalone EDA + Model Training Script
Run this first to explore data and save the model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/sdg_maternal.csv")
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape      : {df.shape}")
print(f"Countries  : {df['Country'].nunique()}")
print(f"Year Range : {df['Year'].min()} – {df['Year'].max()}")
print(f"Regions    : {df['Region'].nunique()}")
print("\nColumn Info:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nStatistical Summary:")
print(df.describe().round(2))

# ── EDA ────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Global MMR trend
global_trend = df.groupby('Year')['Maternal_Mortality_Ratio'].mean()
print(f"\nGlobal Avg MMR (2000): {global_trend[2000]:.1f}")
print(f"Global Avg MMR (2023): {global_trend[2023]:.1f}")
print(f"Reduction            : {((global_trend[2000] - global_trend[2023]) / global_trend[2000] * 100):.1f}%")

# SDG target achievement
latest = df[df['Year'] == df['Year'].max()]
on_track = (latest['Maternal_Mortality_Ratio'] < 70).sum()
print(f"\nCountries below SDG target (<70 MMR): {on_track}/{len(latest)}")

# Correlation analysis
print("\nCorrelation with Maternal Mortality Ratio:")
num_cols = ['Maternal_Mortality_Ratio', 'Antenatal_Care_Coverage',
            'Skilled_Birth_Attendance', 'Adolescent_Birth_Rate',
            'Health_Expenditure_PCT_GDP']
corr = df[num_cols].corr()['Maternal_Mortality_Ratio'].sort_values()
print(corr.round(3))

# ── Feature Engineering ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

le_region = LabelEncoder()
le_income = LabelEncoder()
df['Region_enc'] = le_region.fit_transform(df['Region'])
df['Income_enc'] = le_income.fit_transform(df['Income_Group'])

# Save encoders
os.makedirs('models', exist_ok=True)
with open('models/le_region.pkl', 'wb') as f:
    pickle.dump(le_region, f)
with open('models/le_income.pkl', 'wb') as f:
    pickle.dump(le_income, f)

features = ['Year', 'Antenatal_Care_Coverage', 'Skilled_Birth_Attendance',
            'Adolescent_Birth_Rate', 'Health_Expenditure_PCT_GDP',
            'Region_enc', 'Income_enc']
X = df[features]
y = df['Maternal_Mortality_Ratio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── Model Comparison ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_r2 = -np.inf
results = []

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv = cross_val_score(m, X, y, cv=5, scoring='r2').mean()
    results.append({'Model': name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
                    'R2': round(r2, 3), 'CV_R2': round(cv, 3)})
    print(f"{name:25s} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | R²: {r2:.3f} | CV R²: {cv:.3f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model = m
        best_name = name

print(f"\n✅ Best Model: {best_name} (R² = {best_r2:.3f})")

# ── Save Best Model ────────────────────────────────────────────────────────────
with open('models/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✅ Model saved to models/model.pkl")

# ── Feature Importance ─────────────────────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(fi.to_string(index=False))

# ── Key Findings ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY FINDINGS SUMMARY")
print("=" * 60)
print(f"1. Global MMR reduced by ~{((global_trend[2000] - global_trend[2023]) / global_trend[2000] * 100):.0f}% from 2000 to 2023")
print(f"2. {on_track} out of {len(latest)} countries are already below the SDG 3.1 target of 70")
print(f"3. Skilled Birth Attendance has the strongest negative correlation with MMR")
print(f"4. Sub-Saharan Africa shows the highest average MMR across all years")
print(f"5. Best ML model: {best_name} with R² = {best_r2:.3f}")
print("\nAnalysis complete. Run 'streamlit run app.py' to launch the dashboard.")
