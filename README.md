# PronoTurf - Application de Pronostics Hippiques Intelligents

Application web complète de pronostics hippiques utilisant l'intelligence artificielle et le machine learning pour analyser les courses hippiques et fournir des recommandations de paris optimisées.

> **🚀 Nouveau ici ?** Consultez le [Guide de Démarrage Rapide (QUICKSTART.md)](QUICKSTART.md) pour lancer l'application en 3 minutes !

## Description

PronoTurf est une plateforme moderne qui combine :
- **Intelligence Artificielle** : Modèle Gradient Boosting pour scorer chaque cheval
- **Explicabilité** : SHAP values pour comprendre les facteurs influençant chaque pronostic
- **Gestion de Bankroll** : Stratégies éprouvées (Kelly Criterion, Flat Betting, Martingale)
- **Mode Entraînement** : Simulation sur courses passées pour améliorer ses compétences
- **Notifications** : Alertes Telegram pour ne manquer aucune opportunité

## Avancement du projet

- 🏇 **Sprint monitoring ML en cours** : nous enrichissons itérativement le tableau de bord d'évaluation du modèle. Le dernier incrément introduit les vues `horse_age_performance`, `horse_gender_performance`, `owner_performance`, `day_part_performance`, `month_performance`, `weekday_performance`, `race_order_performance`, `hippodrome_performance`, `track_type_performance`, `country_performance`, `city_performance`, `odds_band_performance`, `handicap_performance`, `weight_performance`, `equipment_performance`, `weather_performance`, `recent_form_performance`, `prediction_rank_performance`, `lead_time_performance`, `win_probability_performance`, `reunion_number_performance`, `value_bet_flag_performance`, `discipline_surface_performance` et désormais `owner_trainer_performance` pour comparer la précision selon la maturité des chevaux, leur genre, les propriétaires engagés, la plage horaire des réunions, la saisonnalité mensuelle, le jour de la semaine, l'ordre des courses dans la réunion, le numéro de réunion (R1, R2, ...), l'hippodrome, la nature de la piste, le pays hôte, la ville hôte, le profil de cote, la charge de handicap, le poids porté, le matériel déclaré, les conditions météo, la lecture de la forme récente, la hiérarchie de sélection du modèle, le délai de publication des pronostics, les bandes de probabilité brutes, l'étiquetage value bet des pronostics ainsi que la vue combinée discipline/surface, un suivi des nationalités des jockeys et entraîneurs via les tableaux `jockey_nationality_performance` et `trainer_nationality_performance`, la synthèse `jockey_trainer_performance` pour identifier les binômes les plus performants, le tableau `owner_trainer_performance` dédié aux alliances propriétaires/entraîneurs et la nouvelle vue `owner_jockey_performance` pour suivre les duos propriétaire/jockey qui convertissent le plus souvent, ainsi que le tableau `start_delay_performance` pour monitorer l'impact des départs en avance ou retardés sur la fiabilité des pronostics, et maintenant la synthèse `final_position_performance` pour contrôler la régularité du modèle selon que les chevaux terminent vainqueurs, sur le podium ou au-delà, tandis que la vue `confidence_level_metrics` est enrichie avec le volume traité, la part de trafic et l'exposition par pronostic pour chaque niveau de confiance. Nous venons également d'ajouter le score global `matthews_correlation` pour suivre l'équilibre des prédictions sur les classes gagnantes et perdantes ainsi que le tableau `api_source_performance` pour comparer la fiabilité des données en fonction de la source d'alimentation (Turfinfo, PMU, Aspiturf, etc.), et un jeu de métriques complémentaires (`specificity`, `false_positive_rate`, `negative_predictive_value`, `balanced_accuracy`) pour diagnostiquer précisément la capacité du modèle à filtrer les faux positifs tout en conservant l'équilibre global.
- 🚧 **Version 1 en préparation** : la V1 n'est pas encore finalisée ; une notification sera partagée dès la clôture de l'ensemble des sprints.

## Nouveautés Analytics Aspiturf

- **Recherche instantanée** des chevaux, jockeys, entraîneurs et hippodromes via l'endpoint `/api/v1/analytics/search`.
- **Explorateur frontend enrichi** avec autocomplétion : sélectionnez un identifiant en quelques clics et pré-remplissez les filtres.
- **Mise à jour du tableau de bord** : filtres hippodrome synchronisés sur toutes les cartes analytics pour accélérer l'analyse pré-course.
- **Classements express multi-entités** : l'endpoint `/api/v1/analytics/insights` calcule en temps réel les meilleurs chevaux, jockeys et entraîneurs selon vos filtres (dates, hippodrome, limite), directement consommé depuis la page Analytics.
- **Tendances de performance** : l'endpoint `/api/v1/analytics/trends` agrège les résultats par semaine ou par mois et l'interface React restitue les courbes d'évolution pour un cheval, un jockey ou un entraîneur.
- **Analyse des séries** : le nouvel endpoint `/api/v1/analytics/streaks` met en évidence les meilleures séries de victoires/podiums par entité et le frontend expose un module dédié pour suivre les séquences en cours.
- **Répartition des performances** : l'endpoint `/api/v1/analytics/distributions` regroupe les courses par distance, numéro de corde, hippodrome ou discipline et la page Analytics propose un tableau interactif pour comparer les segments dominants.
- **Saisonnalité des performances** : l'endpoint `/api/v1/analytics/seasonality` regroupe les résultats par mois ou jour de semaine afin de repérer les périodes les plus rentables, avec un module interactif dans l'explorateur React.
- **Comparateur multi-entités** : l'endpoint `/api/v1/analytics/comparisons` consolide les statistiques de plusieurs chevaux/jockeys/entraîneurs et mesure leurs confrontations directes, accessible depuis un nouveau module de la page Analytics.
- **Indice de forme récent** : l'endpoint `/api/v1/analytics/form` calcule un score (0-5) et un indice de constance sur les N dernières courses d'une entité avec un tableau détaillé directement exploitable dans l'explorateur React.
- **Calendrier de performances** : l'endpoint `/api/v1/analytics/calendar` agrège les résultats jour par jour et l'explorateur React affiche un tableau détaillé des réunions et partants associés.
- **Chasse aux value bets** : l'endpoint `/api/v1/analytics/value` compare la cote observée et la cote probable Aspiturf pour identifier les opportunités les plus rentables et restitue un ROI théorique directement dans l'interface.
- **Analyse de volatilité** : l'endpoint `/api/v1/analytics/volatility` calcule les écarts-types de positions et de cotes pour visualiser la régularité d'une entité, avec un module dédié dans la page Analytics.
- **Efficacité vs cotes** : l'endpoint `/api/v1/analytics/efficiency` confronte les probabilités implicites aux résultats observés afin de repérer les profils surperformants/sous-performants, avec un panneau React détaillant ROI, écarts attendus et tableau des courses.
- **Segments de cotes** : l'endpoint `/api/v1/analytics/odds` répartit les courses par profils (favori, challenger, outsider, long shot) et expose taux de réussite, profits et ROI dans l'explorateur React.
- **Momentum comparatif** : l'endpoint `/api/v1/analytics/momentum` confronte les dernières courses d'une entité à sa période de référence pour mesurer l'évolution des taux de victoire/podium et du ROI, le tout visualisé dans un nouveau panneau React.
- **Charge de travail & repos** : l'endpoint `/api/v1/analytics/workload` analyse les jours de repos entre chaque participation, synthétise les rythmes d'engagement et fournit un tableau chronologique détaillé côté frontend.
- **Progression chronologique** : l'endpoint `/api/v1/analytics/progression` calcule les variations de classement course par course, détecte les séries d'amélioration/régression et alimente un tableau interactif de suivi dans l'interface analytics.

## Suivi de la performance du modèle ML

- **Table de calibration automatique** : chaque exécution de la tâche Celery `update_model_performance` construit désormais des quantiles de probabilité (5 tranches) afin de comparer probabilité moyenne et taux de réussite observé. Cela permet d'identifier immédiatement les sur/sous-estimations du modèle.
- **Indicateurs de calibration synthétiques** : l'Expected Calibration Error (ECE), le biais signé et l'écart maximal sont calculés pour suivre d'un coup d'œil l'ampleur des écarts entre probabilités projetées et réalité terrain.
- **Analyse multi-seuils prête à l'emploi** : les métriques clés (accuracy, précision, rappel, F1, taux de positifs) sont recalculées pour un jeu de seuils standards (`0.20`, `0.30`, `0.40`, `0.50`). Les résultats sont historisés dans la table `ml_model` pour suivre la sensibilité de la stratégie de coupure.
- **Recommandations automatiques de seuil** : en complément de la grille multi-seuils, la tâche met désormais en évidence le meilleur seuil F1 ainsi que les seuils maximisant précision ou rappel. Les métriques associées sont stockées pour guider rapidement le pilotage des alertes et de l'exposition financière.
- **Lecture par niveau de confiance** : en parallèle des quantiles, un tableau de bord consolide précision, rappel et taux de positifs pour chaque niveau de confiance (« high », « medium », « low »). Cette vue directe permet d'ajuster les règles métiers (notifications, exposition financière) selon la fiabilité réelle de chaque segment.
- **Analyse par score de confiance pronostic** : la vue `confidence_score_performance` regroupe les résultats par tranches de pourcentage (très faible à très élevée), expose la volumétrie couverte, la part de trafic et les taux observés afin de piloter les seuils d'alerte lorsque les pronostics sont jugés plus ou moins sûrs.
- **Courbe de gain cumulative** : la même tâche produit désormais une "gain curve" qui mesure, palier par palier, la part des arrivées dans les trois premiers capturée lorsque l'on ne conserve que les meilleures probabilités. Idéal pour optimiser une stratégie de filtrage ou de paris progressifs.
- **Tableau de lift par quantile** : en complément de la courbe de gain, chaque tranche de probabilité est comparée au taux de réussite moyen du lot afin de repérer les segments qui surperforment (ou sous-performent) réellement et d'ajuster la sélection des paris.
- **Courbe précision-rappel synthétique** : une table compacte présente, seuil par seuil, le compromis précision/rappel et le score F1 associé pour piloter finement la stratégie de sélection selon l'appétence au risque.
- **Courbe ROC échantillonnée** : la tâche extrait désormais quelques points clés (taux de faux positifs, rappel, spécificité et indice de Youden) afin d'identifier rapidement le seuil qui maximise la détection tout en contrôlant les erreurs.
- **Diagnostic Kolmogorov-Smirnov** : la séparation entre gagnants et perdants est suivie via la statistique KS et une courbe cumulative compacte, idéale pour identifier un seuil discriminant même lorsque les métriques globales paraissent stables.
- **Chronologie quotidienne des performances** : un tableau `daily_performance` récapitule chaque journée analysée (volumétrie, précision, rappel, part de paris à valeur) pour repérer d'éventuelles dérives temporelles ou effets de calendrier sans quitter le tableau de bord.
- **Tableau de saisonnalité mensuelle** : la vue `month_performance` agrège précision, rappel, volume et taux observé pour chaque mois (avec la part des réunions et les dates couvertes) afin d'identifier rapidement les périodes de l'année où le modèle surperforme ou décroche.
- **Vue segmentée par discipline/surface/value bet** : les métriques clés sont désormais ventilées par discipline hippique, type de surface et présence d'un pari de valeur détecté pour cibler immédiatement les contextes où le modèle décroche.
- **Analyse par distance de course** : le tableau `distance_performance` compare précision, rappel et taux observé entre segments sprint (<1600 m), parcours intermédiaires (1600-2400 m) et épreuves de tenue (>2400 m) afin de repérer les contextes d'effort où le modèle doit être ajusté.
- **Analyse par taille de peloton** : un tableau `field_size_performance` compare précision, rappel et taux de réussite observé selon trois segments (« small », « medium », « large ») pour vérifier que le modèle reste stable face aux champs fournis ou très fournis.
- **Analyse fraîcheur / jours de repos** : le tableau `rest_period_performance` synthétise précision, rappel et taux de réussite observé selon le nombre de jours de repos (<2 semaines, 2-4 semaines, cycle normal, reprise après longue coupure) afin d'identifier les contextes où la fatigue ou la fraîcheur du cheval impacte la fiabilité du modèle.
- **Classement jockeys / entraîneurs** : les tableaux `jockey_performance` et `trainer_performance` agrègent précision, rappel, part de trafic, nombre de chevaux utilisés et taux de réussite observé pour les principaux intervenants. Cette vue permet d'identifier rapidement les profils qui maximisent (ou dégradent) les résultats du modèle et d'alimenter les retours terrain aux équipes sportives.
- **Segmentation par nationalité des intervenants** : les vues `jockey_nationality_performance` et `trainer_nationality_performance` ventilent la précision, le rappel, la part de trafic et la volumétrie de chevaux/courses selon le pays déclaré des jockeys et entraîneurs. Elles permettent de repérer en un coup d'œil les profils internationaux qui décrochent ou brillent afin d'orienter les plans de formation, de recrutement ou de suivi spécifique.
- **Analyse de rentabilité des paris** : une synthèse `betting_value_analysis` confronte les probabilités du modèle aux cotes PMU pour dériver ROI réalisé, espérance de gain théorique, écart moyen probabilité/cote et top 3 des opportunités offrant la meilleure valeur attendue.
- **Analyse value bet vs pronostic standard** : la vue `value_bet_flag_performance` compare la précision, la part de trafic et la part de pronostics entre les courses explicitement taguées « value bet » et le reste des publications afin de vérifier que les alertes mises en avant conservent un avantage réel.
- **Analyse par profil de cote** : le tableau `odds_band_performance` regroupe les partants par familles de cotes (favoris, challengers, outsiders, longues cotes) pour vérifier que la stratégie reste robuste lorsqu'on s'éloigne des chevaux les plus attendus par le marché.
- **Analyse par handicap officiel** : le tableau `handicap_performance` confronte les résultats observés selon la valeur de handicap attribuée pour déceler les segments qui profitent d'un allègement ou pâtissent d'une lourde pénalisation.
- **Analyse par poids porté** : la vue `weight_performance` segmente les résultats selon la charge déclarée afin de détecter les dérives de précision lorsque les chevaux portent très léger ou très lourd.
- **Analyse par équipement déclaré** : la vue `equipment_performance` regroupe les partants selon la présence d'œillères, d'un matériel isolé, d'une combinaison complexe ou l'absence d'information, tout en exposant la part de trafic, le taux de réussite observé et le volume de chevaux distincts.
- **Baromètre par version du modèle** : la vue `model_version_performance` compare précision, rappel, part de trafic et distribution des niveaux de confiance pour chaque version active afin de détecter immédiatement une régression localisée.
- **Alignement aux cotes publiques** : l'analyse `odds_alignment` mesure la corrélation avec les cotes, l'écart moyen modèle/marché et l'overround par course afin de détecter les segments où le modèle s'écarte significativement du marché PMU.
- **Analyse par numéro de corde** : le tableau `draw_performance` regroupe les prédictions selon les positions inside/middle/outside pour vérifier l'influence des stalles de départ sur la précision observée.
- **Analyse pair/impair du numéro de corde** : la vue `draw_parity_performance` compare précision, rappel, taux observé et moyenne des numéros pour les stalles paires vs impaires, afin de confirmer ou d'infirmer les biais liés aux couloirs extrêmes.
- **Analyse par mode de départ** : le tableau `start_type_performance` compare les résultats entre départs mécanisés (stalle, autostart) et départs manuels afin d'identifier un éventuel biais lié au protocole de lancement.
- **Analyse des retards de départ** : la vue `start_delay_performance` mesure précision, rappel, taux observé et part de trafic selon la ponctualité réelle du départ (en avance, à l'heure, retard modéré ou important). Elle expose également le retard moyen/min/max et le nombre de réunions impactées pour prioriser les investigations lorsque les courses démarrent loin de l'horaire officiel.
- **Distribution des positions d'arrivée** : la vue `final_position_performance` regroupe les partants selon leur classement final (vainqueurs, podium, positions 4-6, au-delà) et expose précision, rappel, taux observé, part de trafic ainsi que les positions moyenne/min/max. Elle permet de vérifier que le modèle capture bien les gagnants et podiums sans se contenter de bons taux globaux.
- **Analyse par catégorie/classe de course** : les tableaux `race_category_performance` et `race_class_performance` mesurent précision, rappel et part de trafic selon le niveau officiel de l'épreuve (Groupe, Classe A/B...). Ils permettent de vérifier que le modèle reste performant lors des courses les plus relevées.
- **Analyse par dotation** : la vue `prize_money_performance` segmente les performances selon la prime totale mise en jeu (petits lots, courses intermédiaires, grands prix). Elle révèle si le modèle conserve la même fiabilité lorsqu'il passe de meetings régionaux à des épreuves premium fortement dotées.
- **Analyse par âge du cheval** : le tableau `horse_age_performance` regroupe les résultats par classes d'âge (jeunes ≤3 ans, chevaux en pleine force 4-5 ans, expérimentés 6-8 ans, vétérans 9 ans et +, cas inconnus). Il révèle si le modèle se comporte différemment selon la maturité des partants et indique le nombre de chevaux distincts par segment pour appuyer l'analyse métier.
- **Analyse par genre du cheval** : la nouvelle vue `horse_gender_performance` compare précision, rappel, taux observé et part du trafic entre chevaux mâles, femelles et hongres. Les volumes distincts de chevaux et courses sont exposés afin d'étayer les discussions terrain lorsqu'un genre décroche.
- **Analyse par propriétaire** : le tableau `owner_performance` mesure précision, rappel, part de trafic ainsi que la diversité des chevaux, entraîneurs, jockeys et hippodromes associés à chaque écurie. Il permet d'identifier rapidement les structures qui surperforment ou décrochent pour prioriser les investigations data ou sportives.
- **Synergies propriétaire/entraîneur** : le tableau `owner_trainer_performance` suit les tandems les plus utilisés, calcule précision, rappel, part de trafic et volumétrie pour chaque alliance et permet de repérer les couples qui transforment réellement les partants alignés.
- **Analyse de forme récente** : la vue `recent_form_performance` classe les partants selon leurs dernières arrivées (gagnant récent, forme solide, irrégulière, en difficulté ou inconnue) et expose précision, rappel, taux observé, part de trafic, taille moyenne de l'historique récent ainsi que la fréquence de victoires récentes pour chaque segment.
- **Analyse par plage horaire** : le tableau `day_part_performance` regroupe les métriques selon les réunions du matin, de l'après-midi et du soir. Il expose précision, rappel, part de trafic et horaires moyens/min/max afin d'identifier les créneaux où le modèle doit être recalibré ou surveillé plus finement.
- **Analyse du délai de publication** : la vue `lead_time_performance` mesure la précision en fonction du nombre d'heures séparant la génération du pronostic et le départ officiel, tout en exposant moyenne, médiane et extrêmes du délai pour s'assurer que les publications tardives ou très anticipées restent fiables.
- **Analyse par bande de probabilité brute** : la vue `win_probability_performance` regroupe les pronostics par tranches de probabilité (`<20%`, `20-30%`, `30-40%`, etc.), expose les volumes, la part de trafic, la probabilité moyenne observée et l'écart de calibration moyen afin de vérifier que chaque bande conserve une cohérence entre confiance projetée et résultats terrain.
- **Analyse par jour de la semaine** : la vue `weekday_performance` met en lumière les différences de précision, rappel et taux observé entre les réunions du lundi au dimanche, tout en retraçant les dates couvertes et la part de trafic pour chaque journée.
- **Analyse par ordre de course** : le tableau `race_order_performance` compare les segments début/milieu/fin de réunion pour vérifier que la précision ne se dégrade pas en fin de programme, tout en exposant part de trafic, nombre de réunions couvertes et statistiques sur les numéros de course.
- **Analyse par type de piste** : la vue `track_type_performance` agrège précision, rappel, part de trafic et volumes de réunions par nature de piste (plat, trot, obstacles, mixte) pour vérifier que le modèle reste fiable quel que soit le tracé proposé par les hippodromes.
- **Analyse par hippodrome** : la synthèse `hippodrome_performance` classe chaque site selon la précision, le rappel, le volume de courses et le nombre de chevaux distincts analysés pour détecter rapidement les pistes nécessitant une investigation locale (qualité du datafeed, conditions météo ou configuration de tracé spécifiques).
- **Analyse par pays hôte** : le tableau `country_performance` regroupe les performances par pays d'hippodrome, expose précision, rappel, part de trafic, volume de réunions et liste des villes représentées afin d'identifier les juridictions où le modèle excelle ou doit être recalibré.
- **Analyse par ville hôte** : la vue `city_performance` suit la précision, le rappel, le volume de courses et la part de trafic par agglomération, tout en indiquant les hippodromes concernés et les pays associés pour détecter rapidement les places fortes (ou faibles) du modèle.
- **Analyse par rang de sélection** : le tableau `prediction_rank_performance` expose précision, rappel, taux observé, positions finales et part de trafic pour chaque rang de probabilité (top pick, deuxième choix, troisième choix…), afin de contrôler la hiérarchie de décision du modèle et ajuster les stratégies d'engagement.
- **Analyse par conditions météo** : la vue `weather_performance` regroupe les résultats par grandes familles de météo (ensoleillé, pluie, couvert…) en exposant précision, rappel, taux observé, part de trafic et statistiques de température pour relier la fiabilité du modèle à l'état du ciel ou à la fraîcheur ambiante.

## Technologies

### Backend
- **Python 3.11** : Langage principal
- **FastAPI** : Framework API REST haute performance
- **SQLAlchemy** : ORM pour MySQL (async)
- **Celery** : Orchestration tâches asynchrones
- **Redis** : Cache et message broker
- **scikit-learn** : Machine Learning (Gradient Boosting)
- **SHAP** : Explicabilité des modèles ML
- **httpx** : Client HTTP async
- **tenacity** : Retry logic avec backoff exponentiel
- **APScheduler** : Planification tâches

### Frontend
- **React 18** : Framework UI
- **TypeScript** : Typage statique
- **React Router** : Navigation
- **Axios** : Client HTTP
- **Plotly** : Graphiques interactifs
- **Tailwind CSS** : Styling moderne

### Base de Données
- **MySQL 8.x** : Base de données relationnelle

### Visualisation Alternative
- **Streamlit** : Dashboard exploratoire

### Infrastructure
- **Docker** : Containerisation
- **Docker Compose** : Orchestration services
- **Nginx** : Reverse proxy (production)

### APIs Externes

#### Sources de Données Hippiques (Architecture Multi-Source avec Fallback)

**1. Aspiturf (SOURCE PRINCIPALE) ✨**
- Format : Fichiers CSV ultra-détaillés (120+ colonnes)
- Contenu : Données complètes chevaux, jockeys, entraineurs, statistiques 365 jours
- Points forts :
  - Statistiques enrichies (gains, musiques, performances par hippodrome)
  - Couplages cheval-jockey avec historique
  - Appétence terrain, déferrage, équipement
  - Records, gains carrière complets
- Configuration : `ASPITURF_CSV_PATH` ou `ASPITURF_CSV_URL`

**2. TurfInfo API (FALLBACK) ⚡**
- Format : API REST JSON (sans clé)
- Endpoints :
  - OFFLINE: `https://offline.turfinfo.api.pmu.fr/rest/client/7` (optimisé)
  - ONLINE: `https://online.turfinfo.api.pmu.fr/rest/client/61` (détaillé)
- Contenu : Programme PMU, partants, performances, rapports
- Points forts :
  - Données temps réel
  - Performances détaillées
  - Cotes probables
- Configuration : Aucune clé requise

**3. Open-PMU API (RÉSULTATS) 🏆**
- Format : API REST JSON (sans clé)
- Endpoint : `https://open-pmu-api.vercel.app/api`
- Contenu : Résultats officiels, rapports PMU, arrivées
- Points forts :
  - Résultats définitifs
  - Rapports tous types de paris
  - Non-partants
- Configuration : Aucune clé requise

**4. Open-Meteo (MÉTÉO) 🌤️**
- Format : API REST JSON
- À intégrer : Conditions météo par hippodrome

## Architecture

```
pronoturf/
├── backend/                 # API FastAPI
│   ├── app/
│   │   ├── api/            # Endpoints API
│   │   ├── core/           # Config, sécurité, dépendances
│   │   ├── models/         # Modèles SQLAlchemy
│   │   ├── schemas/        # Schémas Pydantic
│   │   ├── services/       # Logique métier
│   │   ├── ml/             # Machine Learning
│   │   └── tasks/          # Tâches Celery
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env
├── frontend/               # Application React
│   ├── src/
│   │   ├── components/    # Composants réutilisables
│   │   ├── pages/         # Pages complètes
│   │   ├── services/      # Services API
│   │   ├── hooks/         # Custom hooks
│   │   ├── context/       # Context API
│   │   ├── types/         # Types TypeScript
│   │   └── utils/         # Utilitaires
│   ├── package.json
│   ├── Dockerfile
│   └── .env
├── streamlit/             # Dashboard Streamlit
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── database/              # Scripts SQL
│   ├── init.sql          # Schéma initial
│   └── seed.sql          # Données de test
├── ml_models/            # Modèles ML entraînés
├── docker-compose.yml    # Orchestration services
├── .gitignore
├── CDC.md               # Cahier des charges complet
└── README.md            # Ce fichier
```

## Installation

### Prérequis

- Docker 20.10+
- Docker Compose 2.0+
- Git

### Installation Rapide avec Script Automatisé ⚡

**Méthode recommandée** : Utiliser le script de démarrage automatique :

```bash
git clone https://github.com/LeProfDeTechno27/Prono_Gold.git
cd Prono_Gold
./scripts/start.sh
```

Ce script s'occupe de tout automatiquement ! Consultez [QUICKSTART.md](QUICKSTART.md) pour plus de détails.

### Installation Manuelle (Alternative)

1. **Cloner le repository**
```bash
git clone https://github.com/LeProfDeTechno27/Prono_Gold.git
cd Prono_Gold
```

2. **Configurer les variables d'environnement**
```bash
# Copier les fichiers d'exemple
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# Éditer les fichiers .env avec vos configurations (optionnel)
nano backend/.env
```

3. **Lancer l'application avec Docker**
```bash
# Construire et démarrer tous les services
docker-compose up -d --build

# Vérifier que tous les services sont opérationnels
docker-compose ps
```

4. **Initialiser la base de données**
```bash
# Exécuter les scripts d'initialisation
docker-compose exec backend python -m app.db.init_db
```

5. **Accéder à l'application**
- **Frontend** : http://localhost:3000
- **API Backend** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs
- **Dashboard Streamlit** : http://localhost:8501

### Installation Manuelle (Développement)

#### Backend

```bash
cd backend

# Créer environnement virtuel
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt

# Configurer variables d'environnement
cp .env.example .env
nano .env

# Lancer serveur de développement
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend

# Installer dépendances
npm install

# Configurer variables d'environnement
cp .env.example .env
nano .env

# Lancer serveur de développement
npm run dev
```

#### Base de données MySQL

```bash
# Démarrer MySQL avec Docker
docker run -d \
  --name pronoturf-mysql \
  -e MYSQL_ROOT_PASSWORD=root_password \
  -e MYSQL_DATABASE=pronoturf \
  -e MYSQL_USER=pronoturf_user \
  -e MYSQL_PASSWORD=pronoturf_password \
  -p 3306:3306 \
  mysql:8.0

# Importer le schéma
docker exec -i pronoturf-mysql mysql -u pronoturf_user -ppronoturf_password pronoturf < database/init.sql
```

#### Redis

```bash
# Démarrer Redis avec Docker
docker run -d \
  --name pronoturf-redis \
  -p 6379:6379 \
  redis:7-alpine
```

#### Celery

```bash
cd backend

# Démarrer Celery Worker
celery -A app.tasks.celery_app worker --loglevel=info

# Démarrer Celery Beat (dans un autre terminal)
celery -A app.tasks.celery_app beat --loglevel=info
```

## Configuration

### Variables d'environnement Backend

```env
# Base de données
DATABASE_URL=mysql+aiomysql://user:password@localhost:3306/pronoturf

# JWT
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# Redis
REDIS_URL=redis://localhost:6379/0

# APIs Externes
# Aspiturf (source principale - CSV)
ASPITURF_CSV_PATH=/path/to/aspiturf_data.csv  # Priorité sur CSV_URL
ASPITURF_CSV_URL=https://example.com/aspiturf.csv  # Si pas de CSV_PATH
ASPITURF_CSV_DELIMITER=,
ASPITURF_CSV_ENCODING=utf-8
ASPITURF_ENABLED=true

# TurfInfo (fallback)
TURFINFO_ENDPOINT_TYPE=online  # online ou offline

# Telegram
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_ENABLED=true

# Environment
ENVIRONMENT=development
DEBUG=true
```

### Variables d'environnement Frontend

```env
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
```

## Scripts Utilitaires 🛠️

Le projet inclut plusieurs scripts pour faciliter la gestion et le test de l'application.

### Démarrage automatique

```bash
./scripts/start.sh
```

Démarre automatiquement tous les services et initialise la base de données. Voir [QUICKSTART.md](QUICKSTART.md) pour plus de détails.

### Vérification de santé des services

```bash
./scripts/healthcheck.sh
```

Vérifie que tous les services sont opérationnels et communiquent correctement :
- État des containers Docker
- Endpoints HTTP (Backend, Frontend, Streamlit)
- Connectivité MySQL et Redis
- Logs des services

### Test de communication inter-services

```bash
docker-compose exec backend python scripts/test_connectivity.py
```

Test Python avancé de la communication entre tous les services.

### Endpoints de santé de l'API

L'API expose plusieurs endpoints de healthcheck :

```bash
# Health check simple
curl http://localhost:8000/health

# Test connexion MySQL
curl http://localhost:8000/api/v1/health/db

# Test connexion Redis
curl http://localhost:8000/api/v1/health/redis

# Test complet de tous les services
curl http://localhost:8000/api/v1/health/all

# Test broker Celery
curl http://localhost:8000/api/v1/health/celery
```

**Documentation complète** : Consultez [scripts/README.md](scripts/README.md) et [docs/ARCHITECTURE_COMMUNICATION.md](docs/ARCHITECTURE_COMMUNICATION.md)

## Utilisation

### Système de Rôles

L'application dispose de 3 rôles utilisateurs :

#### Administrateur (admin)
- Accès complet à toutes les fonctionnalités
- Gestion des utilisateurs
- Déclenchement manuel des tâches (sync, génération pronostics, ML)
- Accès aux logs et métriques système

#### Abonné (subscriber)
- Accès complet aux fonctionnalités principales
- Consultation illimitée des pronostics
- Gestion bankroll et paris simulés
- Mode entraînement
- Notifications Telegram
- Dashboard analytique

#### Invité (guest)
- Accès limité en lecture
- Consultation programme du jour
- 3 pronostics par jour maximum
- Pas d'accès bankroll ni historique

### Fonctionnalités Principales

#### 1. Consultation du Programme
- Liste des réunions hippiques du jour
- Détails courses (distance, terrain, discipline, partants)
- Conditions météo en temps réel
- Horaires et statuts

#### 2. Pronostics IA
- Scoring de chaque cheval par modèle ML
- Recommandations : gagnant, placé, tiercé, quarté, quinté
- Score de confiance (0-100%)
- Détection automatique de value bets

#### 3. Explicabilité SHAP
- Visualisation des facteurs influençant chaque pronostic
- Top 5 facteurs positifs et négatifs
- Graphiques interactifs (waterfall, bar charts)
- Explications textuelles vulgarisées

#### 4. Gestion de Bankroll
- Capital virtuel initial configurable
- 3 stratégies de mise :
  - **Kelly Criterion** : mise optimale mathématique
  - **Flat Betting** : mise fixe
  - **Martingale** : doublement après perte
- Suivi évolution en temps réel
- Alertes capital critique

#### 5. Paris Simulés
- Placement de paris virtuels
- Historique complet
- Calcul automatique gains/pertes
- Statistiques détaillées (ROI, win rate)

#### 6. Dashboard Analytique
- Graphiques interactifs Plotly
- Win rate par terrain
- ROI par stratégie
- Top jockeys/entraîneurs
- Évolution bankroll temporelle

#### 7. Mode Entraînement
- Simulation sur courses passées
- Comparaison pronostic utilisateur vs IA vs résultat réel
- Feedback immédiat
- Système de progression et badges

#### 8. Notifications Telegram
- Nouveaux pronostics disponibles
- Value bets détectés
- Rappels avant départ
- Résultats courses
- Bilans quotidiens/hebdomadaires
- Liaison directe via `POST /api/v1/notifications/telegram/register` (message de bienvenue automatique)
- Consultation du statut (`GET /api/v1/notifications/telegram/status`) et désactivation instantanée (`DELETE /api/v1/notifications/telegram/unlink`)

## Tâches Planifiées

### Quotidiennes
- **06:00** : Récupération programme PMU du jour
- **07:00** : Génération pronostics IA
- **12:00-23:00** : Vérification arrivées (toutes les heures)

### Hebdomadaires
- **Lundi 02:00** : Ré-entraînement modèle ML

## API Documentation

Documentation complète disponible via Swagger UI :
- **Local** : http://localhost:8000/docs
- **Redoc** : http://localhost:8000/redoc

### Endpoints Principaux

#### Authentification (`/api/v1/auth`)
- `POST /register` : Inscription utilisateur
- `POST /login` : Connexion JWT
- `POST /refresh` : Rafraîchir token
- `POST /logout` : Déconnexion

#### Hippodromes (`/api/v1/hippodromes`)
- `GET /` : Liste tous les hippodromes
- `GET /{hippodrome_id}` : Détails hippodrome
- `GET /{hippodrome_id}/stats` : Statistiques hippodrome

#### Réunions (`/api/v1/reunions`)
- `GET /today` : Réunions du jour
- `GET /date/{date}` : Réunions date spécifique
- `GET /{reunion_id}` : Détails réunion

#### Courses (`/api/v1/courses`)
- `GET /today` : Courses du jour
- `GET /{course_id}` : Détails course + partants
- `GET /{course_id}/results` : Résultats course

#### Pronostics ML (`/api/v1/pronostics`)
- `GET /today` : Pronostics du jour
- `GET /date/{date}` : Pronostics date spécifique
- `GET /course/{course_id}` : Pronostic course unique
- `POST /generate/course/{course_id}` : Générer pronostic async
- `GET /value-bets/today` : Value bets détectés
- `GET /stats/accuracy` : Performance modèle ML

#### Paris Simulés (`/api/v1/paris-simules`)
- `POST /` : Placer pari simulé (validation bankroll)
- `GET /` : Historique paris (pagination, filtres)
- `GET /stats/summary` : Statistiques (ROI, win rate)

#### Notifications (`/api/v1/notifications`)
- `GET /` : Notifications paginées de l'utilisateur
- `GET /unread` : Liste des alertes non lues
- `PATCH /{notification_id}/read` : Marquer une notification comme lue
- `POST /telegram/register` : Enregistrer un chat Telegram et envoyer un message test
- `GET /telegram/status` : Vérifier l'état de la liaison Telegram
- `DELETE /telegram/unlink` : Désactiver les notifications Telegram
- `POST /kelly-criterion` : Calculer mise optimale Kelly
- `DELETE /{pari_id}` : Annuler pari (avant départ)

#### Bankroll (`/api/v1/bankroll`)
- `GET /current` : Bankroll actuel + stats
- `GET /history` : Historique transactions
- `POST /reset` : Réinitialiser bankroll
- `POST /adjust` : Ajustement manuel (admin)
- `GET /stats` : Stats globales (peak, bottom, ROI)
- `GET /stats/period` : Stats par période
- `PATCH /strategy` : Changer stratégie de mise

#### Favoris (`/api/v1/favoris`)
- `GET /` : Liste favoris
- `POST /` : Ajouter favori
- `DELETE /{favori_id}` : Supprimer favori
- `PATCH /{favori_id}/alert` : Toggle alertes
- `GET /by-type/{type}` : Favoris par type
- `GET /{favori_id}/details` : Détails favori

#### Notifications (`/api/v1/notifications`)
- `GET /` : Liste notifications (filtres)
- `GET /unread` : Notifications non lues
- `PATCH /{id}/read` : Marquer comme lu
- `PATCH /read-all` : Tout marquer comme lu
- `DELETE /{id}` : Supprimer notification
- `DELETE /clear` : Nettoyer anciennes
- `GET /stats` : Statistiques notifications
- `GET /recent` : Notifications récentes

**Total**: 50+ endpoints API documentés

## Développement

### Tests

Avant d'exécuter les suites backend, assurez-vous d'installer les dépendances Python :

```bash
pip install -r backend/requirements.txt
```

```bash
# Backend - Tests unitaires
cd backend
pytest tests/ -v --cov=app

# Backend - Tests intégration
pytest tests/integration/ -v

# Frontend - Tests
cd frontend
npm run test

# Frontend - Tests E2E
npm run test:e2e
```

### Linting et Formatting

```bash
# Backend
cd backend
flake8 app/
black app/
mypy app/

# Frontend
cd frontend
npm run lint
npm run format
```

### Génération de Pronostics (Manuel)

```bash
# Via API (admin uniquement)
curl -X POST http://localhost:8000/api/v1/admin/generate-pronostics \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"

# Via CLI
docker-compose exec backend python -m app.tasks.generate_pronostics
```

### Ré-entraînement Modèle ML (Manuel)

```bash
# Via API (admin uniquement)
curl -X POST http://localhost:8000/api/v1/admin/retrain-model \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"

# Via CLI
docker-compose exec backend python -m app.ml.train
```

## Derniers Éléments Ajoutés

### ✅ Sprint 3 - Pipeline ML Complet (v0.3.0)
**45+ features, Gradient Boosting, SHAP, 19 fichiers, 4,309 lignes**

#### Machine Learning
- **Feature Engineering** (650 lignes): 45+ features pour scoring chevaux
  - Statistiques cheval (victoires, places, gains)
  - Performance jockey (win rate, places, courses récentes)
  - Statistiques entraîneur (succès, courses totales)
  - Caractéristiques course (distance, terrain, discipline)
  - Affinité hippodrome (historique par lieu)

- **Modèle Gradient Boosting** (480 lignes): prédiction top 3
  - 200 estimateurs, profondeur 5, learning rate 0.1
  - Train/validation split automatique
  - Sauvegarde modèles versionnés (.pkl)
  - Métriques complètes (accuracy, precision, recall, F1, ROC-AUC)

- **SHAP Explainer** (380 lignes): explicabilité complète
  - SHAP values par cheval
  - Explications en français automatiques
  - Top facteurs positifs/négatifs
  - Impact sur la prédiction

- **Prediction Service** (550 lignes): orchestration
  - Prédictions course complète
  - Détection value bets (edge > 10%)
  - Confidence scores 0-100%
  - Gagnant/Placé/Tiercé/Quarté/Quinté

- **Training Service** (230 lignes): gestion entraînement
  - Pipeline complet de training
  - Logs d'entraînement BDD
  - Métriques de performance
  - Support dates personnalisées

#### Base de Données
- **4 nouveaux modèles SQLAlchemy**:
  - `Pronostic`: prédictions ML par course
  - `PartantPrediction`: prédictions individuelles + SHAP
  - `MLModel`: métadonnées modèles entraînés
  - `TrainingLog`: historique entraînements

#### Tâches Celery
- **4 tâches ML automatisées**:
  - `generate_daily_predictions`: pronostics quotidiens
  - `generate_course_prediction`: prédiction course unique
  - `train_ml_model`: ré-entraînement périodique
  - `evaluate_model_performance`: métriques modèle

#### API Endpoints
- **7 routes `/api/v1/pronostics`**:
  - `GET /today`: pronostics du jour
  - `GET /date/{date}`: pronostics date spécifique
  - `GET /course/{course_id}`: pronostic course unique
  - `POST /generate/course/{course_id}`: génération async
  - `GET /value-bets/today`: value bets détectés
  - `GET /stats/accuracy`: performance modèle

#### Scripts CLI
- `train_initial_model.py`: entraîner modèle initial
- `test_predictions.py`: tester prédictions
- Documentation complète: `backend/scripts/README.md`

### ✅ Sprint 4 - API Clients + Endpoints Complets (v0.4.0)
**9 fichiers, 2,541 lignes, 26 nouvelles routes API**

#### Clients API Optimisés
- **TurfinfoClient** (450 lignes): données avant-course
  - Endpoints OFFLINE (optimisé) et ONLINE (détaillé)
  - `get_programme_jour()`: programme complet
  - `get_partants_course()`: partants + infos
  - `get_performances_detaillees()`: historique complet
  - `get_rapports_definitifs()`: rapports officiels
  - `get_cotes_probables()`: cotes PMU
  - Format date: JJMMAAAA
  - Retry automatique (3 tentatives, backoff exponentiel)
  - Sans clé API

- **OpenPMUClient** (420 lignes): résultats officiels
  - `get_arrivees_by_date()`: tous résultats du jour
  - `get_arrivees_by_hippodrome()`: par hippodrome
  - `get_arrivee_by_prix()`: course spécifique
  - `get_rapports_course()`: rapports PMU
  - `check_non_partants()`: liste non-partants
  - Format date: DD/MM/YYYY
  - Retry automatique avec tenacity
  - Cache recommandé: 24h (données stables)
  - Sans clé API

- **PMUService refactorisé**: utilise les nouveaux clients
  - Architecture séparant avant/après course
  - Méthodes optimisées pour chaque besoin

#### Endpoints API Bankroll (9 routes)
- `GET /current`: bankroll actuel + stats temps réel
- `GET /history`: historique transactions (filtres: type, période)
- `POST /reset`: réinitialiser bankroll
- `POST /adjust`: ajustement manuel (admin)
- `GET /stats`: stats globales (peak, bottom, ROI, win rate)
- `GET /stats/period`: stats par période (day/week/month)
- `PATCH /strategy`: changer stratégie (Kelly/Flat/Martingale)

#### Endpoints API Favoris (8 routes)
- `GET /`: liste favoris (filtres, pagination)
- `POST /`: ajouter favori (validation existence)
- `DELETE /{id}`: supprimer favori
- `PATCH /{id}/alert`: toggle alertes
- `GET /by-type/{type}`: par type avec détails
- `GET /{id}/details`: détails complets
- `DELETE /by-entity`: suppression par entité
- Support 4 types: Horse, Jockey, Trainer, Hippodrome

#### Endpoints API Notifications (9 routes)
- `GET /`: liste (filtres: type, statut, unread)
- `GET /unread`: uniquement non lues
- `PATCH /{id}/read`: marquer comme lu
- `PATCH /read-all`: tout marquer comme lu
- `DELETE /{id}`: supprimer notification
- `DELETE /clear`: nettoyer anciennes lues
- `GET /stats`: stats par type/canal/statut
- `GET /recent`: notifications récentes (X heures)

#### Dépendances
- **tenacity==8.2.3**: retry logic avec backoff exponentiel

#### Scripts
- `test_api_clients.py`: tests automatisés TurfInfo + Open-PMU

### ✅ Sprint 5 - Intégration Aspiturf & Architecture Multi-Source (v0.5.0)
**3 fichiers, 1,450+ lignes, architecture robuste avec fallback automatique**

#### Client Aspiturf (900 lignes)
- **AspiturfClient** : parsing CSV ultra-performant
  - Support fichier local (`csv_path`) et URL distante (`csv_url`)
  - Parsing asynchrone avec `aiofiles`
  - 120+ colonnes documentées (mapping complet)
  - Dataclasses pour typage fort : `AspiturfHorse`, `AspiturfJockey`, `AspiturfTrainer`, `AspiturfCourse`, `AspiturfCouple`
  - Nettoyage et typage automatique des données
  - Configuration flexible via `AspiturfConfig`

- **Méthodes principales** :
  - `get_courses_by_date()` : courses d'une journée
  - `get_partants_course()` : partants avec toutes stats
  - `get_horse_statistics()` : stats agrégées cheval
  - `get_jockey_statistics()` : stats agrégées jockey
  - `get_trainer_statistics()` : stats agrégées entraineur
  - `get_couple_statistics()` : stats couple cheval-jockey

- **Données enrichies disponibles** :
  - Gains carrière, annuels, par victoire/place
  - Musiques complètes (6 dernières courses)
  - Statistiques par hippodrome (% victoires, % places)
  - Appétence terrain (performance par type de piste)
  - Couplages cheval-jockey (historique commun)
  - Équipement (déferrage, œillères) et changements
  - Statuts (inédit, supplémenté, jument pleine)

#### PMUService Refactorisé (550 lignes)
- **Architecture Multi-Source avec Fallback Automatique** :
  1. **Aspiturf** (si configuré) : source principale
  2. **TurfInfo** : fallback automatique si Aspiturf échoue
  3. **OpenPMU** : résultats officiels

- **Nouvelles méthodes** :
  - `fetch_program_for_date()` : avec auto-fallback
  - `fetch_course_partants()` : support multi-source
  - `_fetch_program_aspiturf()` : récupération Aspiturf
  - `_fetch_partants_aspiturf()` : partants Aspiturf
  - `_convert_aspiturf_to_unified()` : normalisation format
  - `_convert_aspiturf_partants()` : mapping partants
  - `_map_aspiturf_discipline()` : mapping disciplines

- **Format unifié** : toutes sources converties en structure commune
- **Logging enrichi** : émojis pour traçabilité (✅ succès, ⚠️ warning, 🔄 fallback, ❌ erreur)
- **Mode strict** : `enable_fallback=False` pour forcer source spécifique

- **Configuration** :
```python
service = PMUService(
    db=session,
    aspiturf_csv_path="/path/to/data.csv",  # Priorité sur csv_url
    aspiturf_csv_url="https://...",         # Si pas de csv_path
    aspiturf_config=AspiturfConfig(
        csv_delimiter=",",
        csv_encoding="utf-8",
        cache_enabled=True,
        cache_ttl_seconds=3600
    ),
    enable_fallback=True  # Fallback automatique si erreur
)
```

#### Script de Test (360 lignes)
- **test_aspiturf_client.py** : suite complète de tests
  - Test 1 : Chargement et parsing CSV
  - Test 2 : Récupération courses par date
  - Test 3 : Récupération partants d'une course
  - Test 4 : Statistiques agrégées (cheval/jockey/entraineur/couple)

- **Usage CLI** :
```bash
# Depuis fichier local
python backend/scripts/test_aspiturf_client.py \
  --csv-path /path/to/aspiturf.csv \
  --date 2025-10-31

# Depuis URL
python backend/scripts/test_aspiturf_client.py \
  --csv-url https://example.com/data.csv
```

#### Avantages Architecture Multi-Source
- ✅ **Résilience** : fallback automatique garantit disponibilité
- ✅ **Données enrichies** : Aspiturf fournit 120+ colonnes vs 30 avec TurfInfo
- ✅ **Flexibilité** : switch facile entre sources sans changer code métier
- ✅ **Performance** : CSV pré-chargé en mémoire (queries ultra-rapides)
- ✅ **Testabilité** : mode force_source pour tests unitaires
- ✅ **Logging complet** : traçabilité source utilisée pour chaque requête

#### Colonnes Aspiturf (120+)
Voir fichier `Procédure Aspiturf.txt` pour documentation complète :
- Identifiants : `numcourse`, `reun`, `prix`, `idChe`, `idJockey`, `idEntraineur`
- Course : `jour`, `hippo`, `dist`, `typec`, `partant`, `cheque`
- Cheval : `sexe`, `age`, `gains*`, `musique*`, `courses*`, `victoires*`, `places*`
- Jockey : `jockey`, `courses*`, `victoires*`, `musiques*`, `pourc*`
- Entraineur : `entraineur`, `courses*`, `victoires*`, `musiques*`, `pourc*`
- Performance : `cotedirect`, `coteprob`, `vha`, `recence`, `recordG`
- Équipement : `defoeil`, `oeil`, `estSupplemente`, `indicateurInedit`
- Couple : `nbCourseCouple`, `TxVictCouple`, `nbVictCouple`
- Hippodrome : `pourcVict*Hippo`, `pourcPlace*Hippo`, `nbrCourse*Hippo`
- Historique : `dernier*` (hippo, alloc, dist, place, cote, joc, ent)

## Éléments à Venir

### Sprint 5.1 (Terminé)
- [x] Client Aspiturf avec parsing CSV
- [x] PMUService refactorisé avec fallback
- [x] Script de test complet
- [x] Intégration avec pipeline ML (features Aspiturf)
- [x] Endpoints API pour statistiques enrichies
- [x] Dashboard avec métriques Aspiturf
- [x] Recherche interactive pour chevaux, jockeys, entraîneurs et hippodromes

### ✅ Sprint 6 - Workspace Analytics Aspiturf (v0.6.0)
- [x] Endpoints `/api/v1/analytics/search`, `/insights`, `/trends`, `/streaks`, `/distributions`, `/form`, `/comparisons` et `/calendar`
- [x] Client Aspiturf enrichi avec agrégateurs (classements, tendances, séries, distributions, forme, comparaisons, calendrier) et stubs de tests
- [x] Suite de tests FastAPI couvrant recherche, classements, tendances, séries, distributions, forme, comparaisons et calendrier
- [x] Explorateur React Query commenté avec modules recherche, classements, tendances, séries, distributions, forme, comparaisons et calendrier, documenté dans le README
- [x] Ajout des modules `/analytics/value`, `/volatility`, `/efficiency`, `/odds`, `/momentum`, `/workload`, `/progression` et `/seasonality` avec UI dédiée (value bets, dispersion, efficacité vs cotes, segments de cotes, momentum, charge de travail, suivi des variations, saisonnalité)

### Sprint 7 (En cours)
- [x] Service Telegram pour notifications (API de liaison, statut et message test)
- [ ] Service Email pour notifications
- [ ] Tâches Celery de notifications
- [ ] Dashboard Streamlit avancé

### Sprint 8 (Planifié)
- [ ] Frontend React complet
- [ ] Pages authentification (Register, Login)
- [ ] Page programme des courses
- [ ] Page détails course avec pronostics
- [ ] Page bankroll et paris simulés

### Sprint 9 (Planifié)
- [ ] Dashboard analytics Plotly
- [ ] Graphiques évolution bankroll
- [ ] Statistiques par terrain, jockey, entraîneur
- [ ] Mode entraînement (simulation courses passées)
- [ ] Système de progression et badges

## Axes d'Amélioration

### Court Terme
- [ ] Tests unitaires complets (couverture > 80%)
- [ ] Documentation API complète
- [ ] Optimisation requêtes SQL
- [ ] Cache Redis pour endpoints fréquents
- [ ] Pagination systématique
- [ ] Gestion erreurs robuste

### Moyen Terme
- [ ] Application mobile native (React Native)
- [ ] Mode dark
- [ ] Internationalisation (i18n)
- [ ] Dashboard Grafana pour monitoring
- [ ] CI/CD avec GitHub Actions
- [ ] Tests E2E Playwright

### Long Terme
- [ ] Support courses internationales
- [ ] Fonctionnalités sociales (partage pronostics)
- [ ] API publique pour développeurs
- [ ] Intégration bookmakers (paris réels)
- [ ] Analyse vidéo courses (Computer Vision)
- [ ] Modèles ML avancés (Deep Learning)

## Sécurité

### Bonnes Pratiques Implémentées
- Authentification JWT avec refresh tokens
- Hash bcrypt pour mots de passe (cost factor 12)
- Rate limiting sur API
- CORS configuré strictement
- Validation inputs Pydantic
- Variables d'environnement pour secrets
- HTTPS obligatoire production
- Conformité RGPD

### Disclaimer Légal

**IMPORTANT** : Cette application est fournie à but éducatif uniquement. Les pronostics générés sont basés sur des analyses statistiques et ne garantissent aucun gain. Le pari comporte des risques. Pariez de manière responsable et dans la limite de vos moyens.

## Support et Contribution

### Reporting de Bugs
Ouvrir une issue sur GitHub avec :
- Description détaillée du problème
- Steps to reproduce
- Logs pertinents
- Environnement (OS, versions)

### Contribution
Les contributions sont les bienvenues :
1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir fichier `LICENSE` pour détails.

## Auteurs

- Développeur Principal : LeProfDeTechno27
- IA Assistant : Claude (Anthropic)

## Ressources

### Documentation Technique
- [Cahier des Charges Complet](CDC.md)
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Documentation React](https://react.dev/)
- [Documentation scikit-learn](https://scikit-learn.org/)
- [Documentation SHAP](https://shap.readthedocs.io/)

### APIs Externes
- [Turfinfo](https://www.turfinfo.fr/) - Programme PMU officiel
- [Open-PMU-API](https://github.com/nanaelie/open-pmu-api) - Résultats courses
- [Open-Meteo](https://open-meteo.com/) - Conditions météo

### Procédures Intégration APIs
- `docs/Procédure d'utilisation de TurfInfo.txt`
- `docs/Procédure Open-PMU-API.txt`

## Contact

- Email : support@pronoturf.ai
- GitHub : [@LeProfDeTechno27](https://github.com/LeProfDeTechno27)

---

**Version** : 0.5.0-beta
**Date de création** : 29 Octobre 2025
**Dernière mise à jour** : 31 Octobre 2025
