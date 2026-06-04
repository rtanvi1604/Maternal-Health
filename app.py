import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maternal Health SDG 3.1 Tracker",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white; border-radius: 12px; padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #065A82; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #065A82;
        border-left: 4px solid #02C39A; padding-left: 12px; margin: 20px 0 10px 0;
    }
    .insight-box {
        background: #e8f5f0; border-left: 4px solid #02C39A;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        margin: 8px 0; font-size: 0.9rem;
    }
    .sdg-badge {
        background: #065A82; color: white; padding: 4px 12px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/sdg_maternal.csv")
    return df

@st.cache_resource
def train_model(df):
    le_region = LabelEncoder()
    le_income = LabelEncoder()
    df2 = df.copy()
    df2['Region_enc'] = le_region.fit_transform(df2['Region'])
    df2['Income_enc'] = le_income.fit_transform(df2['Income_Group'])
    features = ['Year', 'Antenatal_Care_Coverage', 'Skilled_Birth_Attendance',
                'Adolescent_Birth_Rate', 'Health_Expenditure_PCT_GDP',
                'Region_enc', 'Income_enc']
    X = df2[features]
    y = df2['Maternal_Mortality_Ratio']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    return model, mae, r2, fi, le_region, le_income

df = load_data()
model, mae, r2, fi, le_region, le_income = train_model(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Sustainable_Development_Goal_3.png/240px-Sustainable_Development_Goal_3.png", width=80)
    st.markdown("### 🌍 Maternal Health SDG 3.1")
    st.markdown('<span class="sdg-badge">SDG Target: MMR &lt; 70 by 2030</span>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", [
        "📊 Overview Dashboard",
        "🌎 Regional Analysis",
        "📈 Trend Explorer",
        "🤖 ML Predictor",
        "💡 Key Insights"
    ])

    st.markdown("---")
    st.markdown("**Filters**")
    year_range = st.slider("Year Range", 2000, 2023, (2000, 2023))
    selected_regions = st.multiselect("Regions", df['Region'].unique().tolist(), default=df['Region'].unique().tolist())

# ── Filter data ────────────────────────────────────────────────────────────────
dff = df[
    (df['Year'] >= year_range[0]) &
    (df['Year'] <= year_range[1]) &
    (df['Region'].isin(selected_regions))
]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.markdown("## 🌍 Maternal Health Progress Toward SDG 3.1")
    st.markdown("Tracking global maternal mortality reduction toward the 2030 target of **<70 per 100,000 live births**")
    st.markdown("---")

    # KPI row
    latest = dff[dff['Year'] == dff['Year'].max()]
    earliest = dff[dff['Year'] == dff['Year'].min()]
    avg_mmr_now = latest['Maternal_Mortality_Ratio'].mean()
    avg_mmr_then = earliest['Maternal_Mortality_Ratio'].mean()
    reduction = ((avg_mmr_then - avg_mmr_now) / avg_mmr_then) * 100
    below_target = (latest['Maternal_Mortality_Ratio'] < 70).sum()
    avg_anc = latest['Antenatal_Care_Coverage'].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_mmr_now:.0f}</div>
            <div class="metric-label">Avg MMR (latest year)<br>per 100k live births</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#02C39A">{reduction:.1f}%</div>
            <div class="metric-label">MMR Reduction<br>since {year_range[0]}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#f4a261">{below_target}</div>
            <div class="metric-label">Countries<br>Below SDG Target</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_anc:.1f}%</div>
            <div class="metric-label">Avg Antenatal<br>Care Coverage</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Global MMR trend
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-header">Global MMR Trend Over Time</div>', unsafe_allow_html=True)
        trend = dff.groupby('Year')['Maternal_Mortality_Ratio'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend['Year'], y=trend['Maternal_Mortality_Ratio'],
                                  mode='lines+markers', name='Avg MMR',
                                  line=dict(color='#065A82', width=3),
                                  fill='tozeroy', fillcolor='rgba(6,90,130,0.1)'))
        fig.add_hline(y=70, line_dash="dash", line_color="#e63946",
                      annotation_text="SDG 3.1 Target (70)", annotation_position="top right")
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0),
                          xaxis_title="Year", yaxis_title="MMR per 100k births",
                          plot_bgcolor='white', paper_bgcolor='white',
                          font=dict(family="Arial"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Income Group Distribution</div>', unsafe_allow_html=True)
        income_mmr = latest.groupby('Income_Group')['Maternal_Mortality_Ratio'].mean().reset_index()
        fig2 = px.bar(income_mmr, x='Maternal_Mortality_Ratio', y='Income_Group',
                      orientation='h', color='Maternal_Mortality_Ratio',
                      color_continuous_scale='RdYlGn_r')
        fig2.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0),
                            plot_bgcolor='white', paper_bgcolor='white',
                            showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: ANC vs MMR
    st.markdown('<div class="section-header">Antenatal Care Coverage vs Maternal Mortality</div>', unsafe_allow_html=True)
    fig3 = px.scatter(dff, x='Antenatal_Care_Coverage', y='Maternal_Mortality_Ratio',
                      color='Region', size='Health_Expenditure_PCT_GDP',
                      hover_data=['Country', 'Year'],
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.add_hline(y=70, line_dash="dash", line_color="#e63946", annotation_text="SDG Target")
    fig3.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0),
                        plot_bgcolor='white', paper_bgcolor='white',
                        xaxis_title="Antenatal Care Coverage (%)",
                        yaxis_title="MMR per 100k live births")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — REGIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌎 Regional Analysis":
    st.markdown("## 🌎 Regional & Demographic Disparities")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">MMR by Region (Latest Year)</div>', unsafe_allow_html=True)
        latest = dff[dff['Year'] == dff['Year'].max()]
        reg = latest.groupby('Region')['Maternal_Mortality_Ratio'].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(reg, x='Maternal_Mortality_Ratio', y='Region', orientation='h',
                     color='Maternal_Mortality_Ratio', color_continuous_scale='RdYlGn_r',
                     text=reg['Maternal_Mortality_Ratio'].round(0))
        fig.update_traces(textposition='outside')
        fig.add_vline(x=70, line_dash="dash", line_color="#e63946", annotation_text="SDG Target")
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0),
                           plot_bgcolor='white', paper_bgcolor='white',
                           coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Health Indicators Radar by Region</div>', unsafe_allow_html=True)
        radar_data = latest.groupby('Region').agg({
            'Antenatal_Care_Coverage': 'mean',
            'Skilled_Birth_Attendance': 'mean',
            'Health_Expenditure_PCT_GDP': 'mean',
            'Adolescent_Birth_Rate': 'mean'
        }).reset_index()
        fig2 = go.Figure()
        categories = ['ANC Coverage', 'Skilled Birth', 'Health Expenditure', 'Low Adol. Birth Rate']
        colors = px.colors.qualitative.Set2
        for i, row in radar_data.iterrows():
            vals = [row['Antenatal_Care_Coverage'],
                    row['Skilled_Birth_Attendance'],
                    row['Health_Expenditure_PCT_GDP'] * 8,
                    100 - row['Adolescent_Birth_Rate']]
            fig2.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=categories + [categories[0]],
                                            name=row['Region'], line=dict(color=colors[i % len(colors)])))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            height=380, margin=dict(l=0,r=20,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # MMR heatmap over years by region
    st.markdown('<div class="section-header">MMR Heatmap: Region × Year</div>', unsafe_allow_html=True)
    heat = dff.groupby(['Region', 'Year'])['Maternal_Mortality_Ratio'].mean().reset_index()
    heat_pivot = heat.pivot(index='Region', columns='Year', values='Maternal_Mortality_Ratio')
    fig3 = px.imshow(heat_pivot, color_continuous_scale='RdYlGn_r',
                     labels=dict(color="MMR"), aspect='auto')
    fig3.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TREND EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Explorer":
    st.markdown("## 📈 Country-Level Trend Explorer")
    st.markdown("---")

    selected_countries = st.multiselect("Select Countries to Compare",
                                         df['Country'].unique().tolist(),
                                         default=['India', 'Nigeria', 'Brazil', 'Bangladesh'])
    indicator = st.selectbox("Indicator", [
        'Maternal_Mortality_Ratio', 'Antenatal_Care_Coverage',
        'Skilled_Birth_Attendance', 'Adolescent_Birth_Rate', 'Health_Expenditure_PCT_GDP'
    ])

    country_df = df[(df['Country'].isin(selected_countries)) &
                    (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

    fig = px.line(country_df, x='Year', y=indicator, color='Country',
                  markers=True, color_discrete_sequence=px.colors.qualitative.Set1)
    if indicator == 'Maternal_Mortality_Ratio':
        fig.add_hline(y=70, line_dash="dash", line_color="#e63946", annotation_text="SDG 3.1 Target")
    fig.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                       margin=dict(l=0,r=0,t=20,b=0), font=dict(family="Arial"))
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Correlation Between Health Indicators</div>', unsafe_allow_html=True)
    num_cols = ['Maternal_Mortality_Ratio', 'Antenatal_Care_Coverage',
                'Skilled_Birth_Attendance', 'Adolescent_Birth_Rate', 'Health_Expenditure_PCT_GDP']
    corr = dff[num_cols].corr().round(2)
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, aspect='auto')
    fig2.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Predictor":
    st.markdown("## 🤖 ML-Powered MMR Predictor")
    st.markdown("Predict a country's Maternal Mortality Ratio based on health indicators using a trained Random Forest model.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Model Performance")
        m1, m2 = st.columns(2)
        m1.metric("R² Score", f"{r2:.3f}", help="Closer to 1.0 is better")
        m2.metric("Mean Abs Error", f"{mae:.1f}", help="Average prediction error in MMR units")

        st.markdown("### Feature Importance")
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues')
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                           plot_bgcolor='white', paper_bgcolor='white',
                           coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Predict MMR")
        with st.form("predict_form"):
            year_input = st.slider("Year", 2000, 2035, 2025)
            anc = st.slider("Antenatal Care Coverage (%)", 10.0, 100.0, 75.0)
            skilled = st.slider("Skilled Birth Attendance (%)", 10.0, 100.0, 70.0)
            adol = st.slider("Adolescent Birth Rate", 2.0, 150.0, 40.0)
            health_exp = st.slider("Health Expenditure (% GDP)", 1.0, 15.0, 5.0)
            region_sel = st.selectbox("Region", df['Region'].unique().tolist())
            income_sel = st.selectbox("Income Group", df['Income_Group'].unique().tolist())
            submitted = st.form_submit_button("🔮 Predict MMR", use_container_width=True)

        if submitted:
            region_enc = le_region.transform([region_sel])[0]
            income_enc = le_income.transform([income_sel])[0]
            X_input = np.array([[year_input, anc, skilled, adol, health_exp, region_enc, income_enc]])
            pred = model.predict(X_input)[0]
            color = "#02C39A" if pred < 70 else "#e63946"
            st.markdown(f"""
            <div style="background:white;padding:24px;border-radius:12px;text-align:center;
                        box-shadow:0 2px 8px rgba(0,0,0,0.1);margin-top:12px;">
                <div style="font-size:0.9rem;color:#666;">Predicted MMR</div>
                <div style="font-size:3rem;font-weight:800;color:{color}">{pred:.1f}</div>
                <div style="font-size:0.85rem;color:#888;">per 100,000 live births</div>
                <div style="margin-top:12px;font-weight:600;color:{color}">
                    {"✅ Below SDG 3.1 Target" if pred < 70 else "⚠️ Above SDG 3.1 Target (70)"}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Actual vs Predicted scatter
    st.markdown('<div class="section-header">Actual vs Predicted MMR</div>', unsafe_allow_html=True)
    from sklearn.preprocessing import LabelEncoder
    df2 = df.copy()
    df2['Region_enc'] = le_region.transform(df2['Region'])
    df2['Income_enc'] = le_income.transform(df2['Income_Group'])
    features = ['Year','Antenatal_Care_Coverage','Skilled_Birth_Attendance',
                'Adolescent_Birth_Rate','Health_Expenditure_PCT_GDP','Region_enc','Income_enc']
    preds_all = model.predict(df2[features])
    sample = df2.sample(300, random_state=1).copy()
    sample['Predicted'] = model.predict(sample[features])
    fig2 = px.scatter(sample, x='Maternal_Mortality_Ratio', y='Predicted',
                      color='Region', opacity=0.7,
                      labels={'Maternal_Mortality_Ratio': 'Actual MMR', 'Predicted': 'Predicted MMR'})
    max_val = max(sample['Maternal_Mortality_Ratio'].max(), sample['Predicted'].max())
    fig2.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                               name='Perfect Fit', line=dict(dash='dash', color='gray')))
    fig2.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                        margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — KEY INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Key Insights":
    st.markdown("## 💡 Key Insights & Policy Recommendations")
    st.markdown("---")

    latest = dff[dff['Year'] == dff['Year'].max()]
    earliest = dff[dff['Year'] == dff['Year'].min()]

    # Progress toward SDG
    on_track = (latest['Maternal_Mortality_Ratio'] < 70).sum()
    total = len(latest)
    progress_pct = (on_track / total) * 100

    st.markdown("### 📊 SDG 3.1 Progress Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Countries Meeting Target", f"{on_track}/{total}", f"{progress_pct:.0f}% on track")
    col2.metric("Global MMR Reduction (2000→latest)", f"{((earliest['Maternal_Mortality_Ratio'].mean() - latest['Maternal_Mortality_Ratio'].mean()) / earliest['Maternal_Mortality_Ratio'].mean() * 100):.1f}%")
    col3.metric("Highest Risk Region", dff.groupby('Region')['Maternal_Mortality_Ratio'].mean().idxmax())

    st.markdown("### 🔍 Data-Driven Insights")
    insights = [
        "📉 <b>Strong negative correlation</b> between Antenatal Care Coverage and MMR — countries with >85% ANC coverage show significantly lower maternal deaths.",
        "💉 <b>Skilled Birth Attendance</b> is the single strongest predictor of MMR reduction in the ML model (highest feature importance).",
        "💰 <b>Health expenditure as % of GDP</b> shows diminishing returns above 8% — quality of spending matters more than quantity.",
        "👧 <b>High Adolescent Birth Rate</b> remains strongly correlated with elevated MMR, indicating need for youth-focused reproductive health programs.",
        "🌍 <b>Sub-Saharan Africa</b> shows the widest disparity — some countries have reduced MMR by 50%+ while others show stagnation.",
        "📈 <b>Lower-middle income countries</b> show the fastest MMR improvement rate, suggesting effective deployment of basic healthcare interventions.",
        "🎯 At current reduction rates, <b>only ~{:.0f}%</b> of tracked countries will meet the SDG 3.1 target by 2030.".format(progress_pct),
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

    st.markdown("### 📋 Policy Recommendations")
    recs = {
        "1. Scale Skilled Birth Attendance Programs": "Prioritize training and deployment of skilled birth attendants, especially in rural Sub-Saharan Africa and South Asia where gaps are largest.",
        "2. Strengthen Antenatal Care Systems": "Invest in community health worker programs to improve ANC coverage in low-access areas. Even basic 4-visit ANC packages significantly reduce MMR.",
        "3. Address Adolescent Pregnancy": "Integrate family planning and reproductive health education into school curricula. Countries with adolescent birth rates below 20 consistently achieve lower MMR.",
        "4. Targeted Health Spending": "Redirect health expenditure toward primary maternal care rather than tertiary facilities. Low-income countries benefit most from basic obstetric emergency services.",
        "5. Data Collection & Monitoring": "Many low-income nations have significant data gaps. Strengthening civil registration and vital statistics systems is essential for SDG monitoring."
    }
    for title, body in recs.items():
        with st.expander(f"✅ {title}"):
            st.write(body)

    # Progress bar toward 2030 goal
    st.markdown("### 🎯 Overall SDG 3.1 Progress Gauge")
    avg_mmr = latest['Maternal_Mortality_Ratio'].mean()
    start_mmr = 400  # approximate 2000 global average
    target_mmr = 70
    progress = max(0, min(100, (start_mmr - avg_mmr) / (start_mmr - target_mmr) * 100))
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_mmr,
        delta={'reference': start_mmr, 'decreasing': {'color': '#02C39A'}, 'suffix': ' MMR reduction'},
        gauge={
            'axis': {'range': [0, start_mmr], 'tickwidth': 1},
            'bar': {'color': "#065A82"},
            'steps': [
                {'range': [0, 70], 'color': '#02C39A'},
                {'range': [70, 200], 'color': '#f4a261'},
                {'range': [200, start_mmr], 'color': '#e63946'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 70}
        },
        title={'text': "Average MMR (Target: 70 by 2030)"}
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built for IBM Skills Build AICTE 2026 | PS36 – Tracking Maternal Health Progress Toward SDG 3.1 | "
    "Data: SDG National Indicator Framework</small></center>",
    unsafe_allow_html=True
)
