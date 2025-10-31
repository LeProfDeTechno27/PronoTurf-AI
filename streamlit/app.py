"""
PronoTurf - Dashboard Streamlit
Dashboard analytique interactif pour l'analyse des pronostics hippiques
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(
    page_title="PronoTurf Dashboard",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏇 PronoTurf - Dashboard Analytique")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    st.markdown("### Période d'analyse")

    date_range = st.selectbox(
        "Sélectionner la période",
        ["7 derniers jours", "30 derniers jours", "6 derniers mois", "Année", "Tout"]
    )

    st.markdown("### Filtres")
    hippodrome_filter = st.multiselect(
        "Hippodromes",
        ["Tous", "Longchamp", "Vincennes", "Chantilly", "Deauville"]
    )

    discipline_filter = st.multiselect(
        "Disciplines",
        ["Tous", "Plat", "Trot", "Obstacles"]
    )

    st.markdown("---")
    st.info("📊 Dashboard en cours de développement")

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="ROI Global",
        value="0.00%",
        delta="0.00%"
    )

with col2:
    st.metric(
        label="Win Rate",
        value="0.00%",
        delta="0.00%"
    )

with col3:
    st.metric(
        label="Bankroll",
        value="1000.00€",
        delta="+0.00€"
    )

with col4:
    st.metric(
        label="Paris Simulés",
        value="0",
        delta="0"
    )

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Évolution", "🏆 Performance", "📊 Analytics", "🔍 Détails"])

with tab1:
    st.subheader("Évolution du Bankroll")

    # Dummy data for chart
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    dummy_data = pd.DataFrame({
        'Date': dates,
        'Bankroll': [1000 + (i * 0) for i in range(len(dates))]  # Flat line for now
    })

    fig = px.line(dummy_data, x='Date', y='Bankroll', title='Évolution du Bankroll (30 derniers jours)')
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Bankroll (€)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Performance par Stratégie")

    col1, col2 = st.columns(2)

    with col1:
        # Dummy data for strategy comparison
        strategy_data = pd.DataFrame({
            'Stratégie': ['Kelly', 'Flat', 'Martingale'],
            'ROI': [0, 0, 0]
        })

        fig = px.bar(strategy_data, x='Stratégie', y='ROI', title='ROI par Stratégie')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Dummy data for terrain performance
        terrain_data = pd.DataFrame({
            'Terrain': ['Pelouse', 'Piste', 'Sable'],
            'Win Rate': [0, 0, 0]
        })

        fig = px.bar(terrain_data, x='Terrain', y='Win Rate', title='Win Rate par Terrain')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Analytics Avancées")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top 10 Jockeys")
        st.info("Données à venir...")

    with col2:
        st.markdown("### Top 10 Entraîneurs")
        st.info("Données à venir...")

with tab4:
    st.subheader("Détails des Paris")
    st.info("Tableau des paris simulés à venir...")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>PronoTurf v0.1.0 - Dashboard Analytique</p>
        <p>🔗 <a href='http://localhost:3000'>Retour à l'application principale</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
