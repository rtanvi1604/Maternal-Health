"""
generate_data.py
Run this ONCE to create the data folder and dataset.
Usage: python generate_data.py
"""

import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

np.random.seed(42)

countries = [
    'India', 'Nigeria', 'Ethiopia', 'Pakistan', 'Bangladesh', 'Tanzania', 'Kenya',
    'Uganda', 'Mozambique', 'Mali', 'Chad', 'Somalia', 'Niger', 'Sierra Leone',
    'South Sudan', 'Indonesia', 'Philippines', 'Brazil', 'Mexico', 'South Africa',
    'Ghana', 'Cameroon', 'Zambia', 'Zimbabwe', 'Nepal', 'Myanmar', 'Afghanistan',
    'Yemen', 'Haiti', 'Guatemala', 'Bolivia', 'Peru', 'China', 'Thailand',
    'Sri Lanka', 'Vietnam', 'Morocco', 'Egypt', 'Tunisia', 'Rwanda'
]

region_map = {
    'India': 'South Asia', 'Pakistan': 'South Asia', 'Bangladesh': 'South Asia',
    'Nepal': 'South Asia', 'Afghanistan': 'South Asia', 'Myanmar': 'Southeast Asia',
    'Indonesia': 'Southeast Asia', 'Philippines': 'Southeast Asia', 'Thailand': 'Southeast Asia',
    'Vietnam': 'Southeast Asia', 'China': 'East Asia', 'Sri Lanka': 'South Asia',
    'Nigeria': 'Sub-Saharan Africa', 'Ethiopia': 'Sub-Saharan Africa', 'Tanzania': 'Sub-Saharan Africa',
    'Kenya': 'Sub-Saharan Africa', 'Uganda': 'Sub-Saharan Africa', 'Mozambique': 'Sub-Saharan Africa',
    'Mali': 'Sub-Saharan Africa', 'Chad': 'Sub-Saharan Africa', 'Somalia': 'Sub-Saharan Africa',
    'Niger': 'Sub-Saharan Africa', 'Sierra Leone': 'Sub-Saharan Africa', 'South Sudan': 'Sub-Saharan Africa',
    'South Africa': 'Sub-Saharan Africa', 'Ghana': 'Sub-Saharan Africa', 'Cameroon': 'Sub-Saharan Africa',
    'Zambia': 'Sub-Saharan Africa', 'Zimbabwe': 'Sub-Saharan Africa', 'Rwanda': 'Sub-Saharan Africa',
    'Brazil': 'Latin America', 'Mexico': 'Latin America', 'Haiti': 'Latin America',
    'Guatemala': 'Latin America', 'Bolivia': 'Latin America', 'Peru': 'Latin America',
    'Morocco': 'MENA', 'Egypt': 'MENA', 'Tunisia': 'MENA', 'Yemen': 'MENA'
}

income_map = {
    'India': 'Lower-middle', 'Pakistan': 'Lower-middle', 'Bangladesh': 'Lower-middle',
    'Nepal': 'Low', 'Afghanistan': 'Low', 'Myanmar': 'Lower-middle',
    'Indonesia': 'Lower-middle', 'Philippines': 'Lower-middle', 'Thailand': 'Upper-middle',
    'Vietnam': 'Lower-middle', 'China': 'Upper-middle', 'Sri Lanka': 'Lower-middle',
    'Nigeria': 'Lower-middle', 'Ethiopia': 'Low', 'Tanzania': 'Low',
    'Kenya': 'Lower-middle', 'Uganda': 'Low', 'Mozambique': 'Low',
    'Mali': 'Low', 'Chad': 'Low', 'Somalia': 'Low', 'Niger': 'Low',
    'Sierra Leone': 'Low', 'South Sudan': 'Low', 'South Africa': 'Upper-middle',
    'Ghana': 'Lower-middle', 'Cameroon': 'Lower-middle', 'Zambia': 'Lower-middle',
    'Zimbabwe': 'Lower-middle', 'Rwanda': 'Low', 'Brazil': 'Upper-middle',
    'Mexico': 'Upper-middle', 'Haiti': 'Low', 'Guatemala': 'Upper-middle',
    'Bolivia': 'Lower-middle', 'Peru': 'Upper-middle', 'Morocco': 'Lower-middle',
    'Egypt': 'Lower-middle', 'Tunisia': 'Upper-middle', 'Yemen': 'Low'
}

base_mmr_map = {
    'India': 250, 'Nigeria': 820, 'Ethiopia': 680, 'Pakistan': 320, 'Bangladesh': 290,
    'Tanzania': 570, 'Kenya': 420, 'Uganda': 510, 'Mozambique': 650, 'Mali': 780,
    'Chad': 860, 'Somalia': 900, 'Niger': 810, 'Sierra Leone': 870, 'South Sudan': 790,
    'Indonesia': 210, 'Philippines': 180, 'Brazil': 75, 'Mexico': 65, 'South Africa': 145,
    'Ghana': 310, 'Cameroon': 590, 'Zambia': 480, 'Zimbabwe': 410, 'Nepal': 370,
    'Myanmar': 260, 'Afghanistan': 750, 'Yemen': 540, 'Haiti': 395, 'Guatemala': 120,
    'Bolivia': 185, 'Peru': 130, 'China': 55, 'Thailand': 45, 'Sri Lanka': 40,
    'Vietnam': 60, 'Morocco': 95, 'Egypt': 85, 'Tunisia': 55, 'Rwanda': 500
}

years = list(range(2000, 2024))
rows = []

for country in countries:
    base_mmr = base_mmr_map[country]
    base_anc = np.random.uniform(35, 92)
    base_skilled = np.random.uniform(25, 95)
    base_adolescent = np.random.uniform(8, 110)
    base_health_exp = np.random.uniform(2.0, 10.0)

    for y in years:
        t = (y - 2000) / 23
        mmr = max(5, base_mmr * (1 - 0.44 * t) + np.random.normal(0, base_mmr * 0.04))
        anc = min(99, base_anc + 18 * t + np.random.normal(0, 2.5))
        skilled = min(99, base_skilled + 22 * t + np.random.normal(0, 2.5))
        adolescent = max(2, base_adolescent * (1 - 0.28 * t) + np.random.normal(0, 4))
        health_exp = min(14, base_health_exp + 1.8 * t + np.random.normal(0, 0.4))

        rows.append({
            'Country': country,
            'Year': y,
            'Region': region_map[country],
            'Income_Group': income_map[country],
            'Maternal_Mortality_Ratio': round(mmr, 1),
            'Antenatal_Care_Coverage': round(anc, 1),
            'Skilled_Birth_Attendance': round(skilled, 1),
            'Adolescent_Birth_Rate': round(adolescent, 1),
            'Health_Expenditure_PCT_GDP': round(health_exp, 2)
        })

df = pd.DataFrame(rows)
df.to_csv('data/sdg_maternal.csv', index=False)

print(f"✅ Dataset created: data/sdg_maternal.csv")
print(f"   Shape     : {df.shape}")
print(f"   Countries : {df['Country'].nunique()}")
print(f"   Years     : {df['Year'].min()} to {df['Year'].max()}")
print(f"\n✅ Folders created: data/ and models/")
print(f"\nNext steps:")
print(f"  1. python analysis.py")
print(f"  2. streamlit run app.py")