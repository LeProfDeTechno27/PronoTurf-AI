"""PronoTurf - Dashboard Streamlit avancé.

Ce tableau de bord interactif repose sur des jeux de données synthétiques
fournis par :mod:`streamlit.data_providers`.  Les jeux de données peuvent être
remplacés par les vraies API du backend sans modifier l'interface : toute la
logique d'agrégation et d'affichage se trouve ici et manipule exclusivement des
``pandas.DataFrame``.

L'objectif de cette version est double :

* démontrer les analyses avancées prévues pour la V1 (évolution bankroll,
  performance par stratégie, suivi des value bets, calibration des probabilités
  et météo) ;
* fournir une base de code très commentée pour accélérer l'intégration avec les
  vraies données lors des prochains sprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from data_providers import (
    list_available_filters,
    load_bet_history,
    load_data_drift_metrics,
    load_data_quality_checks,
    load_feature_contributions,
    load_monitoring_snapshots,
    load_operational_milestones,
    load_operational_risks,
    load_predictions,
)


# ---------------------------------------------------------------------------
# Configuration Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PronoTurf Dashboard",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏇 PronoTurf - Dashboard Analytique")
st.caption(
    "Visualisation synthétique des performances des pronostics IA."
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Types et fonctions utilitaires
# ---------------------------------------------------------------------------


@dataclass
class PeriodDefinition:
    """Structure décrivant une période sélectionnée dans la sidebar."""

    label: str
    delta: timedelta | None


PERIODS: Tuple[PeriodDefinition, ...] = (
    PeriodDefinition("7 derniers jours", timedelta(days=7)),
    PeriodDefinition("30 derniers jours", timedelta(days=30)),
    PeriodDefinition("6 derniers mois", timedelta(days=180)),
    PeriodDefinition("Année", timedelta(days=365)),
    PeriodDefinition("Tout", None),
)


@st.cache_data(show_spinner=False)
def get_bet_history() -> pd.DataFrame:
    """Charge et met en cache l'historique des paris synthétiques."""

    return load_bet_history()


@st.cache_data(show_spinner=False)
def get_predictions() -> pd.DataFrame:
    """Charge et met en cache les meilleurs pronostics synthétiques."""

    return load_predictions()


@st.cache_data(show_spinner=False)
def get_monitoring_snapshots() -> pd.DataFrame:
    """Charge les indicateurs de monitoring synthétiques."""

    return load_monitoring_snapshots()


@st.cache_data(show_spinner=False)
def get_feature_contributions() -> pd.DataFrame:
    """Charge les contributions moyennes des variables explicatives."""

    return load_feature_contributions()


@st.cache_data(show_spinner=False)
def get_data_quality_checks() -> pd.DataFrame:
    """Charge les alertes qualité de données synthétiques."""

    return load_data_quality_checks()


@st.cache_data(show_spinner=False)
def get_data_drift_metrics() -> pd.DataFrame:
    """Charge les diagnostics de drift de données synthétiques."""

    return load_data_drift_metrics()


@st.cache_data(show_spinner=False)
def get_operational_milestones() -> pd.DataFrame:
    """Charge les jalons opérationnels synthétiques."""

    return load_operational_milestones()


@st.cache_data(show_spinner=False)
def get_operational_risks() -> pd.DataFrame:
    """Charge les risques opérationnels synthétiques."""

    return load_operational_risks()


def _filter_by_period(df: pd.DataFrame, period: PeriodDefinition) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne deux DataFrames : période courante et période de référence.

    La période de référence est utilisée pour calculer les deltas affichés dans
    les métriques principales.  Si aucun delta n'est applicable (période "Tout"),
    le second DataFrame est vide.
    """

    if period.delta is None:
        return df.copy(), df.iloc[0:0]

    end_date = df["date"].max()
    start_date = end_date - period.delta + timedelta(days=1)

    current = df[df["date"].between(start_date, end_date)].copy()

    # Période précédente (même durée juste avant la période courante).
    prev_end = start_date - timedelta(days=1)
    prev_start = prev_end - period.delta + timedelta(days=1)
    previous = df[df["date"].between(prev_start, prev_end)].copy()

    return current, previous


def _apply_filters(
    df: pd.DataFrame,
    hippodromes: Iterable[str],
    disciplines: Iterable[str],
    track_types: Iterable[str],
    min_confidence: float,
    only_value_bets: bool,
) -> pd.DataFrame:
    """Applique les filtres choisis dans la sidebar sur le DataFrame."""

    filtered = df.copy()

    if hippodromes:
        filtered = filtered[filtered["hippodrome"].isin(hippodromes)]

    if disciplines:
        filtered = filtered[filtered["discipline"].isin(disciplines)]

    if track_types:
        filtered = filtered[filtered["track_type"].isin(track_types)]

    filtered = filtered[filtered["confidence_score"] >= min_confidence]

    if only_value_bets:
        filtered = filtered[filtered["is_value_bet"]]

    return filtered.sort_values("date")


def _compute_metrics(current: pd.DataFrame, previous: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calcule les indicateurs clés à afficher sous forme de métriques."""

    def _to_delta(current_value: float, previous_value: float | None) -> float | None:
        if previous_value is None:
            return None
        if previous_value == 0:
            return None
        return ((current_value - previous_value) / abs(previous_value)) * 100

    metrics: Dict[str, Dict[str, float]] = {}

    total_profit = current["profit"].sum()
    total_stake = current["stake"].sum()
    roi = (total_profit / total_stake * 100) if total_stake > 0 else 0.0

    win_rate = (
        (current["result"] == "won").mean() * 100 if not current.empty else 0.0
    )

    initial_bankroll = 1000.0
    bankroll_series = initial_bankroll + current["profit"].cumsum()
    final_bankroll = bankroll_series.iloc[-1] if not bankroll_series.empty else initial_bankroll

    value_bets = current[current["is_value_bet"]]
    value_bets_success = (
        (value_bets["result"] == "won").mean() * 100 if not value_bets.empty else 0.0
    )

    metrics["roi"] = {
        "value": roi,
        "delta": _to_delta(
            roi,
            (previous["profit"].sum() / previous["stake"].sum() * 100)
            if previous["stake"].sum() > 0
            else None,
        ),
    }

    metrics["win_rate"] = {
        "value": win_rate,
        "delta": _to_delta(
            win_rate,
            (previous["result"] == "won").mean() * 100 if not previous.empty else None,
        ),
    }

    metrics["bankroll"] = {
        "value": final_bankroll,
        "delta": _to_delta(
            final_bankroll,
            1000.0 + previous["profit"].cumsum().iloc[-1] if not previous.empty else None,
        ),
    }

    metrics["value_bets"] = {
        "value": float(len(value_bets)),
        "extra": value_bets_success,
        "delta": _to_delta(
            len(value_bets),
            float(len(previous[previous["is_value_bet"]])) if not previous.empty else None,
        ),
    }

    metrics["total_profit"] = {"value": total_profit}
    metrics["total_stake"] = {"value": total_stake}

    return metrics


def _render_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Affiche les métriques principales en haut du dashboard."""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = metrics["roi"].get("delta")
        st.metric(
            label="ROI",
            value=f"{metrics['roi']['value']:.2f}%",
            delta=f"{delta:+.1f}%" if delta is not None else None,
        )

    with col2:
        delta = metrics["win_rate"].get("delta")
        st.metric(
            label="Win Rate",
            value=f"{metrics['win_rate']['value']:.1f}%",
            delta=f"{delta:+.1f}%" if delta is not None else None,
        )

    with col3:
        delta = metrics["bankroll"].get("delta")
        st.metric(
            label="Bankroll",
            value=f"{metrics['bankroll']['value']:.2f} €",
            delta=f"{delta:+.1f}%" if delta is not None else None,
        )

    with col4:
        delta = metrics["value_bets"].get("delta")
        success = metrics["value_bets"].get("extra", 0.0)
        st.metric(
            label="Value Bets joués",
            value=f"{metrics['value_bets']['value']:.0f}",
            delta=f"{delta:+.1f}%" if delta is not None else None,
            help=f"Taux de réussite : {success:.1f}%",
        )


# ---------------------------------------------------------------------------
# Chargement initial des données
# ---------------------------------------------------------------------------

bet_history = get_bet_history()
predictions = get_predictions()
filters = list_available_filters()
monitoring_snapshots = get_monitoring_snapshots()
feature_contributions = get_feature_contributions()
data_quality_checks = get_data_quality_checks()
data_drift_metrics = get_data_drift_metrics()
operational_milestones = get_operational_milestones()
operational_risks = get_operational_risks()


# ---------------------------------------------------------------------------
# Sidebar - configuration utilisateur
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    period_labels = [period.label for period in PERIODS]
    selected_label = st.selectbox("Période", period_labels, index=1)
    selected_period = next(period for period in PERIODS if period.label == selected_label)

    st.markdown("### Filtres avancés")

    hippodromes = st.multiselect(
        "Hippodromes",
        filters["hippodromes"],
        default=filters["hippodromes"],
    )

    disciplines = st.multiselect(
        "Disciplines",
        filters["disciplines"],
        default=filters["disciplines"],
    )

    track_types = st.multiselect(
        "Types de piste",
        filters["track_types"],
        default=filters["track_types"],
    )

    min_confidence = st.slider(
        "Confiance minimale du modèle (%)",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=5.0,
    )

    only_value_bets = st.toggle("Afficher uniquement les value bets", value=False)

    st.markdown("---")
    st.info(
        "📌 Les données affichées sont synthétiques mais suivent les mêmes\n"
        "structures que les futures API.",
        icon="ℹ️",
    )


# ---------------------------------------------------------------------------
# Application des filtres
# ---------------------------------------------------------------------------

filtered_history, previous_period_history = _filter_by_period(bet_history, selected_period)
filtered_history = _apply_filters(
    filtered_history,
    hippodromes,
    disciplines,
    track_types,
    min_confidence,
    only_value_bets,
)

metrics = _compute_metrics(filtered_history, previous_period_history)
_render_metrics(metrics)

st.markdown("---")


# ---------------------------------------------------------------------------
# Contenu principal avec onglets
# ---------------------------------------------------------------------------

tab_evolution, tab_performance, tab_analytics, tab_monitoring, tab_operations, tab_details = st.tabs(
    [
        "📈 Évolution",
        "🏆 Performance",
        "🧠 Analytics avancées",
        "🛡️ Monitoring",
        "🧭 Pilotage",
        "🔍 Détails",
    ]
)


with tab_evolution:
    st.subheader("Évolution du capital")

    if filtered_history.empty:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    else:
        initial_bankroll = 1000.0
        evolution_df = filtered_history[["date", "profit"]].copy()
        evolution_df["bankroll"] = initial_bankroll + evolution_df["profit"].cumsum()

        line_fig = px.line(
            evolution_df,
            x="date",
            y="bankroll",
            markers=True,
            title="Évolution de la bankroll cumulée",
        )
        line_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Bankroll (€)",
            hovermode="x unified",
        )
        st.plotly_chart(line_fig, use_container_width=True)

        st.markdown("### Résultat quotidien")
        daily_df = (
            filtered_history.groupby("date", as_index=False)["profit"].sum()
        )
        daily_df["statut"] = daily_df["profit"].apply(lambda x: "Gain" if x >= 0 else "Perte")

        bar_fig = px.bar(
            daily_df,
            x="date",
            y="profit",
            color="statut",
            color_discrete_map={"Gain": "#16a34a", "Perte": "#dc2626"},
            title="Gains et pertes par jour",
        )
        bar_fig.update_layout(xaxis_title="Date", yaxis_title="Profit (€)")
        st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("### Statistiques par semaine")
        weekly_df = (
            filtered_history.assign(week=lambda d: d["date"].dt.isocalendar().week)
            .groupby("week")
            .agg(
                total_paris=("course_id", "count"),
                profit_total=("profit", "sum"),
                mise_totale=("stake", "sum"),
            )
            .assign(
                roi=lambda d: (
                    d["profit_total"] / d["mise_totale"] * 100
                ).where(d["mise_totale"] > 0, other=0.0)
            )
            .reset_index()
        )
        st.dataframe(
            weekly_df,
            use_container_width=True,
            column_config={
                "roi": st.column_config.NumberColumn("ROI %", format="%.2f"),
                "profit_total": st.column_config.NumberColumn("Profit (€)", format="%.2f"),
                "mise_totale": st.column_config.NumberColumn("Mise (€)", format="%.2f"),
            },
        )


with tab_performance:
    st.subheader("Comparaison des stratégies")

    if filtered_history.empty:
        st.warning("Aucune donnée disponible pour évaluer les performances.")
    else:
        strategy_df = (
            filtered_history.groupby("strategy")
            .agg(
                paris=("course_id", "count"),
                profit=("profit", "sum"),
                mise=("stake", "sum"),
                win_rate=("result", lambda s: (s == "won").mean() * 100),
            )
            .assign(roi=lambda d: (d["profit"] / d["mise"]) * 100)
            .reset_index()
        )

        roi_fig = px.bar(
            strategy_df,
            x="strategy",
            y="roi",
            color="strategy",
            title="ROI par stratégie",
            labels={"roi": "ROI (%)", "strategy": "Stratégie"},
        )
        st.plotly_chart(roi_fig, use_container_width=True)

        col_left, col_right = st.columns(2)

        with col_left:
            track_fig = px.bar(
                filtered_history,
                x="track_type",
                y="profit",
                color="discipline",
                title="Profit par type de piste",
                labels={"track_type": "Type de piste", "profit": "Profit (€)"},
            )
            st.plotly_chart(track_fig, use_container_width=True)

        with col_right:
            discipline_fig = px.bar(
                filtered_history,
                x="discipline",
                y="profit",
                color="weather",
                title="Profit par discipline et météo",
                labels={"discipline": "Discipline", "profit": "Profit (€)"},
            )
            st.plotly_chart(discipline_fig, use_container_width=True)

        st.markdown("### Tableau de synthèse")
        st.dataframe(
            strategy_df,
            use_container_width=True,
            column_config={
                "profit": st.column_config.NumberColumn("Profit (€)", format="%.2f"),
                "mise": st.column_config.NumberColumn("Mise (€)", format="%.2f"),
                "win_rate": st.column_config.NumberColumn("Win Rate %", format="%.1f"),
                "roi": st.column_config.NumberColumn("ROI %", format="%.2f"),
            },
        )


with tab_analytics:
    st.subheader("Explorations avancées")

    if filtered_history.empty:
        st.warning("Aucune donnée à analyser.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            scatter_fig = px.scatter(
                filtered_history,
                x="odds",
                y="profit",
                size="confidence_score",
                color="strategy",
                hover_data=["hippodrome", "course_label", "result"],
                title="Profit vs Cote (taille = confiance)",
            )
            scatter_fig.update_layout(xaxis_title="Cote PMU", yaxis_title="Profit (€)")
            st.plotly_chart(scatter_fig, use_container_width=True)

        with col_b:
            heatmap_df = (
                filtered_history.groupby(["predicted_rank", "actual_rank"])
                .size()
                .reset_index(name="occurrences")
            )
            heatmap_fig = px.density_heatmap(
                heatmap_df,
                x="predicted_rank",
                y="actual_rank",
                z="occurrences",
                color_continuous_scale="Blues",
                title="Calibration classement prédit vs réel",
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

        st.markdown("### Value bets remarquables")
        value_bets_df = (
            filtered_history[filtered_history["is_value_bet"]]
            .nlargest(10, "profit")
            .loc[:, [
                "date",
                "hippodrome",
                "course_label",
                "strategy",
                "odds",
                "confidence_score",
                "profit",
                "result",
            ]]
        )
        st.dataframe(
            value_bets_df,
            use_container_width=True,
            column_config={
                "confidence_score": st.column_config.NumberColumn("Confiance %", format="%.1f"),
                "profit": st.column_config.NumberColumn("Profit (€)", format="%.2f"),
            },
        )

        st.markdown("### Répartition météo / performance")
        weather_df = (
            filtered_history.groupby("weather")
            .agg(
                paris=("course_id", "count"),
                profit=("profit", "sum"),
                temperature_moy=("temperature", "mean"),
                win_rate=("result", lambda s: (s == "won").mean() * 100),
            )
            .reset_index()
        )
        weather_chart = px.bar(
            weather_df,
            x="weather",
            y="profit",
            color="win_rate",
            title="Impact de la météo sur le profit",
            labels={"profit": "Profit (€)", "weather": "Météo", "win_rate": "Win Rate %"},
        )
        st.plotly_chart(weather_chart, use_container_width=True)


with tab_monitoring:
    st.subheader("Qualité modèle & santé des données")

    if monitoring_snapshots.empty:
        st.info(
            "Les indicateurs synthétiques ne sont pas encore disponibles pour ce sprint."
        )
    else:
        st.markdown("### Synthèse du sprint courant")

        latest_snapshots = (
            monitoring_snapshots.sort_values("snapshot_date")
            .groupby("metric", as_index=False)
            .tail(1)
            .sort_values("metric")
        )

        st.dataframe(
            latest_snapshots,
            use_container_width=True,
            column_config={
                "value": st.column_config.NumberColumn("Valeur", format="%.3f"),
                "target": st.column_config.NumberColumn("Cible", format="%.3f"),
                "status": st.column_config.TextColumn("Statut"),
                "comment": st.column_config.TextColumn("Commentaire"),
            },
            hide_index=True,
        )

        st.markdown("### Historique par indicateur")
        metric_options = latest_snapshots["metric"].tolist()
        selected_metric = st.selectbox(
            "Indicateur suivi",
            metric_options,
            help="Chaque série couvre les trois derniers sprints d'observation.",
        )

        metric_history = (
            monitoring_snapshots[monitoring_snapshots["metric"] == selected_metric]
            .sort_values("snapshot_date")
        )

        history_fig = px.line(
            metric_history,
            x="snapshot_date",
            y="value",
            markers=True,
            color="status",
            title=f"Évolution de {selected_metric}",
        )
        history_fig.add_hline(
            y=metric_history["target"].iloc[0],
            line_dash="dash",
            line_color="#7c3aed",
            annotation_text="Cible",
        )
        history_fig.update_layout(
            xaxis_title="Date de snapshot",
            yaxis_title="Valeur",
            legend_title="Statut",
        )
        st.plotly_chart(history_fig, use_container_width=True)

    st.markdown("### Contributions moyennes du modèle")

    if feature_contributions.empty:
        st.info("Les contributions de variables sont en cours de calcul.")
    else:
        top_features = feature_contributions.head(10).copy()
        bar_fig = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            color="category",
            labels={
                "importance": "Importance normalisée",
                "feature": "Variable",
                "category": "Famille",
            },
            title="Top 10 des facteurs explicatifs",
        )
        bar_fig.update_layout(yaxis_title="Variable", xaxis_title="Importance")
        st.plotly_chart(bar_fig, use_container_width=True)

        st.dataframe(
            feature_contributions,
            use_container_width=True,
            column_config={
                "importance": st.column_config.NumberColumn(
                    "Importance", format="%.3f"
                ),
                "avg_shap": st.column_config.NumberColumn(
                    "Impact moyen (SHAP)", format="%.3f"
                ),
                "description": st.column_config.TextColumn("Description"),
            },
            hide_index=True,
        )

    st.markdown("### Qualité des données & pipelines")

    if data_quality_checks.empty:
        st.success("Aucun incident de données détecté sur la période analysée.")
    else:
        severity_order = ["Critique", "Majeure", "Mineure"]
        severity_counts = (
            data_quality_checks.groupby("severity")["check"].count()
            .reindex(severity_order, fill_value=0)
        )
        severity_icons = {"Critique": "🔥", "Majeure": "⚠️", "Mineure": "ℹ️"}

        col_sev1, col_sev2, col_sev3 = st.columns(3)
        for column, severity in zip((col_sev1, col_sev2, col_sev3), severity_order):
            with column:
                st.metric(
                    label=f"{severity_icons[severity]} {severity}",
                    value=int(severity_counts[severity]),
                )

        st.dataframe(
            data_quality_checks,
            use_container_width=True,
            column_config={
                "check": st.column_config.TextColumn("Contrôle"),
                "severity": st.column_config.TextColumn("Sévérité"),
                "status": st.column_config.TextColumn("Statut"),
                "impacted_rows": st.column_config.NumberColumn(
                    "Lignes impactées", format="%d"
                ),
                "recommendation": st.column_config.TextColumn("Action recommandée"),
                "last_seen": st.column_config.DateColumn("Dernière occurrence"),
            },
            hide_index=True,
        )

    st.markdown("### Surveillance du drift de données")

    if data_drift_metrics.empty:
        st.success("Aucun drift détecté sur la fenêtre de 14 jours suivie.")
    else:
        drift_fig = px.bar(
            data_drift_metrics,
            x="drift_score",
            y="feature",
            color="status",
            orientation="h",
            labels={
                "drift_score": "Score de drift (PSI)",
                "feature": "Variable surveillée",
                "status": "Statut",
            },
            title="Priorisation des variables en drift",
        )
        drift_fig.update_layout(
            xaxis_title="Score de drift",
            yaxis_title="Variable surveillée",
            legend_title="Statut",
        )
        st.plotly_chart(drift_fig, use_container_width=True)

        st.dataframe(
            data_drift_metrics,
            use_container_width=True,
            column_config={
                "drift_score": st.column_config.NumberColumn(
                    "Score de drift", format="%.2f"
                ),
                "p_value": st.column_config.NumberColumn("p-value", format="%.3f"),
                "status": st.column_config.TextColumn("Statut"),
                "reference_mean": st.column_config.NumberColumn(
                    "Moyenne référence", format="%.3f"
                ),
                "current_mean": st.column_config.NumberColumn(
                    "Moyenne actuelle", format="%.3f"
                ),
                "comment": st.column_config.TextColumn("Commentaire"),
                "window_start": st.column_config.DateColumn("Début fenêtre"),
                "window_end": st.column_config.DateColumn("Fin fenêtre"),
            },
            hide_index=True,
        )


with tab_operations:
    st.subheader("Pilotage & opérations")

    if operational_milestones.empty and operational_risks.empty:
        st.info("Les jeux de données opérationnels seront intégrés prochainement.")
    else:
        if not operational_milestones.empty:
            st.markdown("### Jalons critiques & roadmap")

            status_order = ["À risque", "En cours", "Livré"]
            status_icons = {"À risque": "🚨", "En cours": "🛠️", "Livré": "✅"}
            status_counts = (
                operational_milestones.groupby("status")["milestone"].count()
                .reindex(status_order, fill_value=0)
            )

            col_status = st.columns(len(status_order))
            for column, status in zip(col_status, status_order):
                with column:
                    st.metric(
                        label=f"{status_icons[status]} {status}",
                        value=int(status_counts[status]),
                    )

            timeline_df = operational_milestones.copy()
            timeline_df["start_date"] = pd.to_datetime(timeline_df["start_date"])
            timeline_df["due_date"] = pd.to_datetime(timeline_df["due_date"])

            timeline_fig = px.timeline(
                timeline_df,
                x_start="start_date",
                x_end="due_date",
                y="workstream",
                color="status",
                hover_data={
                    "milestone": True,
                    "owner": True,
                    "confidence": True,
                    "impact": True,
                    "comment": True,
                },
                title="Roadmap opérationnelle consolidée",
            )
            timeline_fig.update_yaxes(autorange="reversed")
            timeline_fig.update_layout(
                xaxis_title="Calendrier",
                yaxis_title="Stream",
                legend_title="Statut",
            )
            st.plotly_chart(timeline_fig, use_container_width=True)

            st.dataframe(
                timeline_df,
                use_container_width=True,
                column_config={
                    "start_date": st.column_config.DateColumn("Début"),
                    "due_date": st.column_config.DateColumn("Échéance"),
                    "confidence": st.column_config.NumberColumn("Confiance %", format="%d"),
                    "impact": st.column_config.TextColumn("Impact"),
                    "comment": st.column_config.TextColumn("Notes"),
                },
                hide_index=True,
            )

        if not operational_risks.empty:
            st.markdown("### Registre des risques & actions")

            severity_order = ["Critique", "Élevée", "Modérée"]
            severity_icons = {"Critique": "🔥", "Élevée": "⚠️", "Modérée": "ℹ️"}
            severity_counts = (
                operational_risks.groupby("severity")["risk"].count()
                .reindex(severity_order, fill_value=0)
            )

            col_risk = st.columns(len(severity_order))
            for column, severity in zip(col_risk, severity_order):
                with column:
                    st.metric(
                        label=f"{severity_icons[severity]} {severity}",
                        value=int(severity_counts[severity]),
                    )

            st.dataframe(
                operational_risks,
                use_container_width=True,
                column_config={
                    "risk": st.column_config.TextColumn("Risque"),
                    "severity": st.column_config.TextColumn("Sévérité"),
                    "owner": st.column_config.TextColumn("Pilote"),
                    "status": st.column_config.TextColumn("Statut"),
                    "mitigation": st.column_config.TextColumn("Mitigation"),
                    "next_review": st.column_config.DateColumn("Revue"),
                    "trend": st.column_config.TextColumn("Tendance"),
                },
                hide_index=True,
            )


with tab_details:
    st.subheader("Table détaillée des paris")

    if filtered_history.empty:
        st.info("Filtre trop restrictif : aucun pari à afficher.")
    else:
        st.dataframe(
            filtered_history,
            use_container_width=True,
            column_config={
                "confidence_score": st.column_config.NumberColumn("Confiance %", format="%.1f"),
                "profit": st.column_config.NumberColumn("Profit (€)", format="%.2f"),
                "stake": st.column_config.NumberColumn("Mise (€)", format="%.2f"),
                "odds": st.column_config.NumberColumn("Cote", format="%.2f"),
            },
        )

        st.markdown("### Focus course et pronostics associés")

        available_courses = filtered_history["course_id"].unique().tolist()
        if not available_courses:
            st.info("Sélectionnez une autre période pour explorer les courses.")
        else:
            course_id = st.selectbox(
                "Course", available_courses, format_func=lambda cid: f"Course #{cid}"
            )
            course_predictions = predictions[predictions["course_id"] == course_id]

            if course_predictions.empty:
                st.warning("Aucune prédiction synthétique disponible pour cette course.")
            else:
                st.dataframe(
                    course_predictions,
                    use_container_width=True,
                    column_config={
                        "win_probability": st.column_config.NumberColumn("Proba victoire", format="%.2f"),
                        "value_score": st.column_config.NumberColumn("Score de value", format="%.2f"),
                    },
                )

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>PronoTurf v0.7.0 - Dashboard avancé (données synthétiques)</p>
        <p>🔗 <a href='http://localhost:3000'>Retour à l'application principale</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)

