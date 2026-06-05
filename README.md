# 🌍 Maternal Health Progress Tracker — SDG 3.1

**IBM Skills Build for University Engagements | AICTE-2026**
**Problem Statement No. 36** — Tracking Maternal Health Progress Toward SDG 3.1: A Global Data Analysis

---

## 📌 Problem Statement

The Sustainable Development Goal 3.1 aims to reduce the global maternal mortality ratio (MMR) to **less than 70 per 100,000 live births by 2030**. This project analyzes country-wise data on maternal mortality and associated health indicators to generate data-driven insights into the factors influencing maternal health across regions and income groups.

---

## 🎯 Objectives

- Analyze global trends in Maternal Mortality Ratio (MMR) from 2000–2023
- Identify regional and income-group disparities in maternal health outcomes
- Discover key indicators (ANC, skilled birth attendance, health expenditure) that correlate with MMR
- Build an ML model to predict MMR based on health indicators
- Visualize progress toward SDG 3.1 target through an interactive dashboard

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Dashboard | Streamlit |
| ML Models | Scikit-learn (Random Forest, Gradient Boosting, Linear Regression) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy |
| Cloud | IBM Cloud Lite (for deployment) |

---

## 📂 Project Structure

```
maternal-health-sdg/
├── app.py                  # Streamlit dashboard (5 pages)
├── analysis.py             # EDA + model training script
├── data/
│   └── sdg_maternal.csv    # SDG maternal health dataset
├── models/
│   ├── model.pkl           # Trained Random Forest model
│   ├── le_region.pkl       # Region label encoder
│   └── le_income.pkl       # Income group label encoder
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/maternal-health-sdg.git
cd maternal-health-sdg

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run EDA and train model (optional)
python analysis.py

# 4. Launch dashboard
streamlit run app.py
```

---

## 📊 Dashboard Features

| Page | Description |
|------|-------------|
| 📊 Overview Dashboard | Global MMR trends, KPIs, ANC vs MMR scatter |
| 🌎 Regional Analysis | Regional bar charts, radar chart, MMR heatmap |
| 📈 Trend Explorer | Country-level comparison, correlation heatmap |
| 🤖 ML Predictor | Predict MMR using health indicators |
| 💡 Key Insights | Policy recommendations, SDG progress gauge |

---

## 🤖 ML Model Performance

| Model | MAE | R² Score |
|-------|-----|----------|
| Linear Regression | ~45 | ~0.72 |
| Ridge Regression | ~44 | ~0.73 |
| **Random Forest** | **~18** | **~0.95** |
| Gradient Boosting | ~22 | ~0.93 |

---

## 💡 Key Findings

1. Global MMR reduced by ~45% from 2000 to 2023, but progress is uneven
2. **Skilled Birth Attendance** is the strongest predictor of lower MMR
3. Sub-Saharan Africa shows the highest average MMR across all years
4. Countries with ANC coverage >85% consistently achieve lower maternal mortality
5. Adolescent Birth Rate remains a significant risk factor for elevated MMR

---

## 📸 Application Snapshots

### Maternal Health - Home Page
![Dashboard_1](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/Overview_Dashboard_1.png)

![Dashboard_2](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/Overview_Dashboard_2.png)

### Regional Analysis
![Region_1](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/Regional_Analyis_1.png)

![Region_2](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/Regional_Analysis_2.png)

### Trend Explorer
![Trends_1](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/Trend_Explorer_1.png)

![Trends_2](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/Trend_Explorer_2.png)

### Machine Learning Predictor
![ML_1](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/ML_Predictor_1.png)

![ML_2](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/ML_Predictor_2.png)

### Key Insights
![Insight_1](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/Key_Insights_1.png)

### Policy Recommendations
![Policy](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/Policy_Recommendations.png)

### Overall Progress
![Progress](https://github.com/rtanvi1604/Maternal-Health/blob/main/Prototype%20Snapshots/Overall_Progres.png)


---

## 📋 Policy Recommendations

1. Scale skilled birth attendant programs in high-burden regions
2. Invest in community-based antenatal care outreach
3. Address adolescent pregnancy through school-based reproductive health programs
4. Redirect health spending toward primary obstetric emergency services
5. Strengthen civil registration systems for better SDG monitoring

---

## 👩‍💻 Developer

**Tanvi R** | B.Tech AI & Data Science | Panimalar Engineering College, Chennai
- GitHub: [github.com/rtanvi1604](https://github.com/rtanvi1604)
- LinkedIn: [linkedin.com/in/r-tanvi](https://linkedin.com/in/r-tanvi)

---

## 📄 Dataset Source

SDG National Indicator Framework — Government of India
[data.gov.in](https://www.data.gov.in/resource/sustainable-development-goals-national-indicator-framework-version-31-2021)
