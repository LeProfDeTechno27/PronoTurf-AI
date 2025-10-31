# CAHIER DES CHARGES COMPLET
# Application Web de Pronostics Hippiques Intelligents

## 1. INTRODUCTION GÉNÉRALE

### 1.1 Présentation du Projet

Le projet consiste à développer from scratch une application web complète de pronostics hippiques destinée aux amateurs de paris sportifs hippiques en France. L'application exploitera l'intelligence artificielle et le machine learning pour analyser les courses hippiques du jour et fournir des recommandations de paris optimisées (placé, tiercé, quarté, quinté) basées sur une analyse multi-paramétrique avancée.

L'application permettra aux utilisateurs de s'entraîner quotidiennement sur des courses réelles, d'améliorer progressivement leurs stratégies de paris, et de gérer intelligemment leur bankroll grâce à des algorithmes éprouvés. Un système d'explicabilité des pronostics (SHAP values) rendra transparent le processus de décision de l'IA, permettant aux utilisateurs de comprendre les facteurs clés influençant chaque recommandation.

### 1.2 Objectifs Principaux

1. **Fiabilité prédictive** : Fournir des pronostics de haute qualité basés sur l'analyse de multiples sources de données officielles et l'entraînement continu de modèles de machine learning.
2. **Aide à la décision** : Offrir une transparence totale sur les facteurs influençant les pronostics via l'explicabilité SHAP, permettant aux utilisateurs d'affiner leur compréhension du turf.
3. **Optimisation financière** : Implémenter des stratégies de gestion de bankroll éprouvées (Kelly Criterion, flat betting, martingale) pour maximiser le ROI et minimiser les risques.
4. **Apprentissage continu** : Proposer un mode entraînement quotidien permettant aux utilisateurs de simuler des paris et de comparer leurs performances avec les recommandations IA.
5. **Accessibilité** : Créer une interface moderne, intuitive et performante accessible via navigateur web, avec notifications Telegram pour ne manquer aucune opportunité.

### 1.3 Public Cible

- **Profil principal** : Amateurs et passionnés de paris hippiques en France
- **Niveau d'expertise** : Du débutant souhaitant s'initier aux paris hippiques au parieur confirmé recherchant un outil d'aide à la décision avancé
- **Besoins** : Accès rapide aux pronostics du jour, compréhension des facteurs de performance, gestion rigoureuse du capital de paris

### 1.4 Technologies Utilisées

- **Backend** : Python avec FastAPI (API REST haute performance)
- **Frontend** : React avec TypeScript (interface utilisateur moderne et typée)
- **Base de données** : MySQL 8.x (stockage structuré des données)
- **Machine Learning** : scikit-learn (modèle Gradient Boosting, SHAP)
- **Task Scheduling** : Celery avec Redis (orchestration des tâches asynchrones)
- **Visualisation** : Plotly (graphiques interactifs) + Streamlit (dashboard exploratoire)
- **Scheduling ML** : APScheduler (ré-entraînement automatique du modèle)
- **APIs externes** : Turfinfo (avant course + historique), AspiTurf, open-pmu-api (après course), Open-Meteo (en complément)

## 2. DESCRIPTION DES FONCTIONNALITÉS

### 2.1 Fonctionnalités Essentielles (Core Features)

#### 2.1.1 Système d'Authentification et Gestion Utilisateurs

**Inscription/Connexion sécurisée**
- Création de compte avec email et mot de passe
- Authentification JWT (JSON Web Tokens) pour sécuriser les sessions
- Hash des mots de passe avec bcrypt
- Token refresh automatique

**Profil utilisateur personnalisé**
- Gestion des informations personnelles
- Paramètres de préférences (hippodromes favoris, types de courses préférés, seuils d'alerte)
- Photo de profil (optionnelle)

**Historique personnel**
- Journal complet des pronostics consultés
- Historique des paris simulés
- Statistiques personnelles de performance

#### 2.1.2 Récupération et Affichage du Programme des Courses

**Intégration multi-sources**
- Programme officiel PMU via API Turfinfo (procédure en annexe)
- Détails des partants via TurfInfo
- Contrôle des arrivées via open-pmu-api
- AspiTurf sera integré mais desactiver de base

**Affichage complet**
- Liste des réunions hippiques du jour
- Détails de chaque course : distance, terrain (pelouse/piste/sable), discipline (plat/trot/obstacles), nombre de partants
- Horaires précis de départ
- Statut en temps réel (à venir, en cours, terminée)

**Enrichissement météo**
- Conditions météo automatiques via Open-Meteo
- Impact prévu sur le terrain
- Fallback automatique vers jeux d'essai en cas d'indisponibilité des APIs

#### 2.1.3 Moteur de Scoring IA et Pronostics

**Modèle Gradient Boosting**
- Implémentation scikit-learn pour scoring de chaque cheval
- Sélection dynamique des features les plus pertinentes pour chaque type de course
- Pondération adaptative selon le contexte

**Analyse multi-critères pour chaque cheval**
- **Forme récente** : performances des 3-5 dernières courses
- **Valeur handicap/Rating** : évaluation officielle
- **Distance idéale** : historique sur distances similaires
- **Type de sol/Piste** : performances sur terrain similaire
- **Numéro de corde** : avantage/désavantage statistique
- **Poids porté** : impact sur la performance
- **Équipement** : œillères, ferrage
- **Santé/Repos** : jours depuis dernière course

**Analyse jockey/entraîneur**
- Statistiques récentes (win rate, place rate)
- Affinité historique avec le cheval
- Expérience sur la piste et la distance
- Style de course (attentiste, leader)

**Conditions de course et contextuels**
- Catégorie/Classe de la course
- Nombre de partants (impact sur la stratégie)
- Conditions météo en temps réel
- Topographie du parcours (virages, dénivelé)
- Type de départ (autostart, volte, élastique pour le trot)

**Facteurs statistiques avancés**
- Performance historique sur l'hippodrome spécifique
- Tendance des écuries/entraîneurs du moment
- Rapports probables/cote PMU
- Chronos/Réduction kilométrique (trot)
- Données de tracking GPS/vitesses sectionnelles (sectionals)

**Génération des pronostics**
- Calcul automatique pour gagnant, placé, tiercé, quarté, quinté
- Score de confiance (0-100%) pour chaque pronostic
- Classement des chevaux par probabilité de performance

#### 2.1.4 Explicabilité des Pronostics (SHAP Values)

**Calcul des contributions**
- Implémentation SHAP pour chaque cheval
- Identification des facteurs positifs et négatifs
- Quantification de l'impact de chaque feature

**Visualisation intuitive**
- Graphiques de contribution par facteur
- Code couleur (vert = positif, rouge = négatif)
- Affichage des top 5 facteurs influents
- Explications textuelles générées automatiquement

**Aide à la compréhension**
- Vulgarisation des concepts techniques
- Conseils basés sur l'analyse SHAP
- Comparaison entre chevaux d'une même course

#### 2.1.5 Détails Enrichis des Partants

**Intégration multi-sources**
- Données Turfinfo (programme officiel)
- AspiTurf (performances détaillées)
- Combinaison intelligente des sources

**Fiche complète par cheval**
- Identité (nom, âge, robe, sexe, propriétaire)
- Performances historiques (tableau des 10 dernières courses)
- Statistiques de carrière (nombre de victoires, places, gains)
- Rapports détaillés des dernières courses
- Entraîneur et jockey actuels avec leurs stats
- Poids porté et handicap
- Équipement spécial

**Statistiques contextuelles**
- Win rate sur terrain similaire
- Performance sur la distance
- Affinité avec l'hippodrome
- Historique face aux adversaires du jour

#### 2.1.6 Système de Gestion de Bankroll

**Simulation de stratégies**
- **Kelly Criterion** : mise optimale basée sur l'avantage probabiliste
- **Flat betting** : mise fixe sur chaque pari
- **Martingale** : doublement après perte (avec warnings sur les risques)

**Calcul automatique des mises**
- Recommandation de mise pour chaque pronostic
- Adaptation au capital disponible
- Gestion des seuils de risque

**Tracking financier**
- Capital virtuel initial configurable
- Évolution en temps réel du bankroll
- Alertes si capital critique (< 20% initial)

**Statistiques de performance**
- ROI (Return on Investment) global et par stratégie
- Win rate (taux de réussite)
- Profit/perte par période
- Comparaison des stratégies

### 2.2 Fonctionnalités Avancées

#### 2.2.1 Dashboard Analytique Interactif

**Graphiques Plotly interactifs**
- **Win rate par terrain** : comparaison pelouse/piste/sable
- **ROI par stratégie** : Kelly vs Flat vs Martingale
- **Top jockeys** : classement par performance
- **Top entraîneurs** : statistiques d'efficacité
- **Affinité profil hippodrome** : performances par lieu

**Visualisation temporelle**
- Évolution du bankroll au fil du temps (graphique en ligne)
- Historique des gains/pertes par semaine/mois
- Tendances saisonnières

**Filtres et personnalisation**
- Période personnalisable (7 jours, 30 jours, année)
- Filtrage par hippodrome, discipline, type de course
- Export des graphiques en PNG

#### 2.2.2 Fournisseur de Cotes en Direct

**Récupération des cotes**
- Interrogation périodique d'un endpoint d'odds
- Fallback synthétique si indisponibilité (calcul basé sur les performances)
- Mise à jour toutes les 5 minutes avant la course

**Calcul de value bets**
- Comparaison cote PMU vs probabilité IA
- Identification automatique des opportunités (expected value > 0)
- Score de value (low/medium/high)

**Alertes opportunités**
- Notification des value bets détectés
- Seuil de value configurable par l'utilisateur
- Temps restant avant départ

#### 2.2.3 Système de Notifications Telegram

**Configuration**
- Connexion du compte Telegram via bot API
- Paramétrage des types de notifications souhaitées
- Horaires d'envoi personnalisables

**Types de notifications**
- Nouveau pronostic du jour disponible
- Value bet détecté (cote intéressante)
- Rappel avant départ d'une course avec pronostic fort
- Résultats des courses suivies
- Bilan quotidien/hebdomadaire

**Orchestration Celery/Redis**
- Tâches asynchrones pour envoi sans bloquer l'application
- File d'attente pour gestion de charge
- Retry automatique en cas d'échec

#### 2.2.4 Mode Entraînement Quotidien

**Accès aux courses passées**
- Base de données historique des courses
- Sélection aléatoire ou par période
- Masquage des résultats réels initialement

**Simulation de paris**
- L'utilisateur fait son pronostic
- Saisie de la mise virtuelle
- Révélation progressive (avant/pendant/après course)

**Feedback immédiat**
- Comparaison pronostic utilisateur vs IA vs résultat réel
- Calcul du gain/perte théorique
- Analyse des erreurs (facteurs négligés)

**Progression et gamification**
- Score d'apprentissage évolutif
- Badges de réussite (10 bons pronostics, ROI > 10%, etc.)
- Historique de progression

#### 2.2.5 Contrôle des Arrivées et Validation

**Vérification automatique**
- API open-pmu pour récupération des arrivées officielles
- Mise à jour automatique post-course
- Validation des pronostics

**Calcul des performances**
- Gains/pertes automatiques pour paris simulés
- Mise à jour du bankroll virtuel
- Statistiques de réussite

**Historique vérifié**
- Archive complète des courses terminées
- Pronostics archivés avec résultats
- Rapports PMU officiels disponibles

#### 2.2.6 Pipeline MLOps et Ré-entraînement

**Planification automatique**
- APScheduler pour tâches périodiques
- Ré-entraînement hebdomadaire du modèle
- Intégration de nouvelles données de courses

**Suivi des performances**
- Métriques de qualité du modèle (accuracy, F1-score, ROC-AUC)
- Comparaison version N vs N-1
- Dashboard MLOps pour visualisation

**Gestion des modèles**
- Sérialisation des modèles entraînés (joblib/pickle)
- Versioning automatique (model_v1, model_v2, etc.)
- Rollback possible en cas de dégradation

**Archivage**
- Historique complet des runs d'entraînement
- Logs détaillés des performances
- Backup automatique des meilleurs modèles

#### 2.2.7 Vue React Temps Réel

**Monitoring dynamique**
- Auto-refresh toutes les 30 secondes
- Liste des meilleures opportunités actuelles
- Indicateur de nouveauté (nouveau pronostic non consulté)

**Courses à venir**
- Timeline visuelle des prochaines courses
- Countdown avant départ
- Badge "hot" pour value bets

**Synchronisation API**
- WebSocket ou polling pour mises à jour
- Notifications in-app en temps réel
- État de connexion visible

#### 2.2.8 Dashboard Streamlit (Vue Alternative)

**Interface exploratoire**
- Streamlit pour analyse rapide et interactive
- Consultation des indicateurs clés
- Filtres dynamiques

**Visualisation données brutes**
- Tables interactives (recherche, tri, pagination)
- Export CSV/Excel
- Graphiques générés à la volée

**Usage**
- Accès réservé aux utilisateurs premium (optionnel)
- Complément au dashboard React principal
- Idéal pour analyses ad-hoc

### 2.3 Fonctionnalités Complémentaires Retenues

#### 2.3.1 Système de Favoris et Watchlist

**Gestion des favoris**
- Sauvegarde de chevaux favoris (suivi carrière)
- Liste de jockeys/entraîneurs préférés
- Hippodromes favoris

**Alertes personnalisées**
- Notification quand un cheval favori court
- Performance automatique des favoris
- Statistiques dédiées

#### 2.3.2 Comparateur de Performances

**Analyse comparative**
- Comparaison tête-à-tête entre 2+ chevaux d'une course
- Historique des confrontations directes
- Avantages/désavantages de chacun

**Visualisation**
- Radar chart des compétences (vitesse, endurance, régularité)
- Tableau comparatif détaillé
- Recommandation IA sur le meilleur choix

## 3. ARCHITECTURE DE LA BASE DE DONNÉES MYSQL

### 3.1 Principes de Conception

L'architecture de la base de données suit les meilleures pratiques de conception MySQL :
- Normalisation 3NF pour éviter la redondance
- Clés primaires auto-incrémentées (INT UNSIGNED)
- Clés étrangères avec contraintes d'intégrité référentielle
- Indexes sur colonnes fréquemment recherchées
- Types de données optimisés pour minimiser l'espace disque
- Naming conventions cohérentes (snake_case)

### 3.2 Schéma Complet de la Base de Données

#### 3.2.1 Table users (Utilisateurs)

```sql
CREATE TABLE users (
    user_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role ENUM('admin', 'subscriber', 'guest') DEFAULT 'guest',
    telegram_id VARCHAR(100) UNIQUE,
    profile_picture_url VARCHAR(500),
    initial_bankroll DECIMAL(10, 2) DEFAULT 1000.00,
    current_bankroll DECIMAL(10, 2) DEFAULT 1000.00,
    preferred_strategy ENUM('kelly', 'flat', 'martingale') DEFAULT 'flat',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_email (email),
    INDEX idx_telegram_id (telegram_id),
    INDEX idx_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.2 Table user_preferences (Préférences Utilisateur)

```sql
CREATE TABLE user_preferences (
    preference_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    favorite_hippodromes JSON,
    favorite_disciplines JSON,
    value_bet_threshold DECIMAL(5, 2) DEFAULT 5.00,
    notification_email BOOLEAN DEFAULT TRUE,
    notification_telegram BOOLEAN DEFAULT FALSE,
    notification_types JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.3 Table hippodromes (Hippodromes)

```sql
CREATE TABLE hippodromes (
    hippodrome_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    city VARCHAR(100),
    country VARCHAR(50) DEFAULT 'France',
    track_type ENUM('plat', 'trot', 'obstacles', 'mixte') NOT NULL,
    track_length INT UNSIGNED,
    track_surface VARCHAR(50),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_code (code),
    INDEX idx_country (country)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.4 Table reunions (Réunions Hippiques)

```sql
CREATE TABLE reunions (
    reunion_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    hippodrome_id INT UNSIGNED NOT NULL,
    reunion_date DATE NOT NULL,
    reunion_number TINYINT UNSIGNED NOT NULL,
    status ENUM('scheduled', 'ongoing', 'completed', 'cancelled') DEFAULT 'scheduled',
    api_source VARCHAR(50),
    weather_conditions JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (hippodrome_id) REFERENCES hippodromes(hippodrome_id) ON DELETE CASCADE,
    INDEX idx_reunion_date (reunion_date),
    INDEX idx_hippodrome_date (hippodrome_id, reunion_date),
    UNIQUE KEY unique_reunion (hippodrome_id, reunion_date, reunion_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.5 Table courses (Courses)

```sql
CREATE TABLE courses (
    course_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    reunion_id INT UNSIGNED NOT NULL,
    course_number TINYINT UNSIGNED NOT NULL,
    course_name VARCHAR(300),
    discipline ENUM('plat', 'trot_monte', 'trot_attele', 'haies', 'steeple', 'cross') NOT NULL,
    distance INT UNSIGNED NOT NULL,
    prize_money DECIMAL(10, 2),
    race_category VARCHAR(100),
    race_class VARCHAR(50),
    surface_type ENUM('pelouse', 'piste', 'sable', 'fibre') NOT NULL,
    start_type ENUM('autostart', 'volte', 'elastique', 'stalle', 'corde') DEFAULT 'stalle',
    scheduled_time TIME NOT NULL,
    actual_start_time TIME,
    number_of_runners TINYINT UNSIGNED,
    status ENUM('scheduled', 'running', 'finished', 'cancelled') DEFAULT 'scheduled',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (reunion_id) REFERENCES reunions(reunion_id) ON DELETE CASCADE,
    INDEX idx_reunion_course (reunion_id, course_number),
    INDEX idx_scheduled_time (scheduled_time),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.6 Table horses (Chevaux)

```sql
CREATE TABLE horses (
    horse_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    official_id VARCHAR(50) UNIQUE,
    name VARCHAR(200) NOT NULL,
    birth_year YEAR,
    gender ENUM('male', 'female', 'hongre') NOT NULL,
    coat_color VARCHAR(50),
    breed VARCHAR(100),
    sire VARCHAR(200),
    dam VARCHAR(200),
    owner VARCHAR(300),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_name (name),
    INDEX idx_official_id (official_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.7 Table jockeys (Jockeys)

```sql
CREATE TABLE jockeys (
    jockey_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    official_id VARCHAR(50) UNIQUE,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    birth_date DATE,
    nationality VARCHAR(50),
    weight DECIMAL(4, 1),
    career_wins INT UNSIGNED DEFAULT 0,
    career_places INT UNSIGNED DEFAULT 0,
    career_starts INT UNSIGNED DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_full_name (last_name, first_name),
    INDEX idx_official_id (official_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.8 Table trainers (Entraîneurs)

```sql
CREATE TABLE trainers (
    trainer_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    official_id VARCHAR(50) UNIQUE,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    stable_name VARCHAR(200),
    nationality VARCHAR(50),
    career_wins INT UNSIGNED DEFAULT 0,
    career_places INT UNSIGNED DEFAULT 0,
    career_starts INT UNSIGNED DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_full_name (last_name, first_name),
    INDEX idx_official_id (official_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.9 Table partants (Partants/Runners)

```sql
CREATE TABLE partants (
    partant_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    course_id INT UNSIGNED NOT NULL,
    horse_id INT UNSIGNED NOT NULL,
    jockey_id INT UNSIGNED,
    trainer_id INT UNSIGNED NOT NULL,
    numero_corde TINYINT UNSIGNED NOT NULL,
    poids_porte DECIMAL(4, 1),
    handicap_value INT,
    equipment JSON,
    days_since_last_race INT UNSIGNED,
    recent_form VARCHAR(50),
    odds_pmu DECIMAL(6, 2),
    final_position TINYINT UNSIGNED,
    disqualified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE,
    FOREIGN KEY (horse_id) REFERENCES horses(horse_id) ON DELETE CASCADE,
    FOREIGN KEY (jockey_id) REFERENCES jockeys(jockey_id) ON DELETE SET NULL,
    FOREIGN KEY (trainer_id) REFERENCES trainers(trainer_id) ON DELETE CASCADE,
    INDEX idx_course_horse (course_id, horse_id),
    INDEX idx_course_numero (course_id, numero_corde),
    UNIQUE KEY unique_partant (course_id, numero_corde)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.10 Table performances_historiques (Historique Performances Chevaux)

```sql
CREATE TABLE performances_historiques (
    performance_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    horse_id INT UNSIGNED NOT NULL,
    course_date DATE NOT NULL,
    hippodrome_code VARCHAR(10),
    discipline VARCHAR(50),
    distance INT UNSIGNED,
    surface_type VARCHAR(50),
    final_position TINYINT UNSIGNED,
    number_of_runners TINYINT UNSIGNED,
    time_seconds DECIMAL(6, 2),
    sectional_times JSON,
    reduction_kilometrique DECIMAL(5, 2),
    jockey_name VARCHAR(200),
    trainer_name VARCHAR(200),
    odds DECIMAL(6, 2),
    prize_won DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (horse_id) REFERENCES horses(horse_id) ON DELETE CASCADE,
    INDEX idx_horse_date (horse_id, course_date DESC),
    INDEX idx_course_date (course_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.11 Table pronostics (Pronostics IA)

```sql
CREATE TABLE pronostics (
    pronostic_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    course_id INT UNSIGNED NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    gagnant_predicted JSON,
    place_predicted JSON,
    tierce_predicted JSON,
    quarte_predicted JSON,
    quinte_predicted JSON,
    confidence_score DECIMAL(5, 2),
    value_bet_detected BOOLEAN DEFAULT FALSE,
    shap_values JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE,
    INDEX idx_course_id (course_id),
    INDEX idx_generated_at (generated_at),
    INDEX idx_value_bet (value_bet_detected)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.12 Table partant_predictions (Scores Individuels par Partant)

```sql
CREATE TABLE partant_predictions (
    prediction_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    pronostic_id INT UNSIGNED NOT NULL,
    partant_id INT UNSIGNED NOT NULL,
    predicted_position TINYINT UNSIGNED,
    win_probability DECIMAL(5, 4),
    place_probability DECIMAL(5, 4),
    confidence_score DECIMAL(5, 2),
    shap_contributions JSON,
    top_positive_features JSON,
    top_negative_features JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pronostic_id) REFERENCES pronostics(pronostic_id) ON DELETE CASCADE,
    FOREIGN KEY (partant_id) REFERENCES partants(partant_id) ON DELETE CASCADE,
    INDEX idx_pronostic_id (pronostic_id),
    INDEX idx_partant_id (partant_id),
    UNIQUE KEY unique_prediction (pronostic_id, partant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.13 Table paris_simules (Paris Simulés Utilisateurs)

```sql
CREATE TABLE paris_simules (
    pari_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    course_id INT UNSIGNED NOT NULL,
    bet_type ENUM('gagnant', 'place', 'tierce', 'quarte', 'quinte', 'couple', 'trio') NOT NULL,
    bet_amount DECIMAL(8, 2) NOT NULL,
    selected_horses JSON NOT NULL,
    strategy_used ENUM('kelly', 'flat', 'martingale', 'manual') NOT NULL,
    pronostic_id INT UNSIGNED,
    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result ENUM('pending', 'won', 'lost', 'cancelled') DEFAULT 'pending',
    payout DECIMAL(10, 2) DEFAULT 0.00,
    net_profit DECIMAL(10, 2),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE,
    FOREIGN KEY (pronostic_id) REFERENCES pronostics(pronostic_id) ON DELETE SET NULL,
    INDEX idx_user_course (user_id, course_id),
    INDEX idx_placed_at (placed_at),
    INDEX idx_result (result)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.14 Table bankroll_history (Historique Bankroll)

```sql
CREATE TABLE bankroll_history (
    history_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    transaction_date DATE NOT NULL,
    transaction_type ENUM('bet', 'win', 'loss', 'reset', 'adjustment') NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    balance_after DECIMAL(10, 2) NOT NULL,
    pari_id INT UNSIGNED,
    description VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (pari_id) REFERENCES paris_simules(pari_id) ON DELETE SET NULL,
    INDEX idx_user_date (user_id, transaction_date),
    INDEX idx_pari_id (pari_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.15 Table favoris (Favoris Utilisateurs)

```sql
CREATE TABLE favoris (
    favori_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    entity_type ENUM('horse', 'jockey', 'trainer', 'hippodrome') NOT NULL,
    entity_id INT UNSIGNED NOT NULL,
    alert_enabled BOOLEAN DEFAULT TRUE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_entity (user_id, entity_type, entity_id),
    UNIQUE KEY unique_favori (user_id, entity_type, entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.16 Table notifications (Notifications)

```sql
CREATE TABLE notifications (
    notification_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    notification_type ENUM('pronostic', 'value_bet', 'race_reminder', 'result', 'daily_report') NOT NULL,
    title VARCHAR(300) NOT NULL,
    message TEXT NOT NULL,
    related_course_id INT UNSIGNED,
    related_pronostic_id INT UNSIGNED,
    sent_via ENUM('email', 'telegram', 'in_app') NOT NULL,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP NULL,
    status ENUM('pending', 'sent', 'failed') DEFAULT 'pending',
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (related_course_id) REFERENCES courses(course_id) ON DELETE SET NULL,
    FOREIGN KEY (related_pronostic_id) REFERENCES pronostics(pronostic_id) ON DELETE SET NULL,
    INDEX idx_user_sent (user_id, sent_at),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.17 Table ml_models (Versioning Modèles ML)

```sql
CREATE TABLE ml_models (
    model_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_samples INT UNSIGNED,
    features_used JSON,
    hyperparameters JSON,
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    roc_auc DECIMAL(5, 4),
    file_path VARCHAR(500) NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_version (version),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### 3.2.18 Table training_logs (Logs d'Entraînement ML)

```sql
CREATE TABLE training_logs (
    log_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    model_id INT UNSIGNED NOT NULL,
    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_seconds INT UNSIGNED,
    status ENUM('started', 'completed', 'failed') NOT NULL,
    error_message TEXT,
    metrics JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES ml_models(model_id) ON DELETE CASCADE,
    INDEX idx_model_run (model_id, run_date),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 3.3 Relations et Contraintes d'Intégrité

**Hiérarchie principale des courses :**
```
hippodromes (1) --> (N) reunions (1) --> (N) courses (1) --> (N) partants
```

**Entités liées aux partants :**
```
partants (N) --> (1) horses
partants (N) --> (1) jockeys
partants (N) --> (1) trainers
```

**Pronostics et prédictions :**
```
courses (1) --> (N) pronostics (1) --> (N) partant_predictions
```

**Utilisateurs et paris :**
```
users (1) --> (N) paris_simules
users (1) --> (N) bankroll_history
users (1) --> (N) favoris
users (1) --> (N) notifications
```

**Contraintes CASCADE** : Suppression en cascade pour maintenir l'intégrité référentielle (ex : supprimer un utilisateur supprime ses paris).

**Contraintes UNIQUE** : Empêcher les doublons (ex : un seul partant par numéro de corde par course).

## 4. SYSTÈME DE RÔLES

### 4.1 Rôles Utilisateurs

**Administrateur (admin)**
- Accès complet à toutes les fonctionnalités
- Gestion des utilisateurs
- Déclenchement manuel des tâches (sync programme, génération pronostics, ré-entraînement modèle)
- Accès aux logs et métriques système
- Gestion des modèles ML

**Abonné (subscriber)**
- Accès complet aux fonctionnalités principales
- Consultation illimitée des pronostics
- Gestion complète du bankroll
- Accès au mode entraînement
- Notifications Telegram
- Dashboard analytique
- Favoris et comparateur

**Invité (guest)**
- Accès limité en lecture
- Consultation du programme du jour
- Consultation limitée des pronostics (ex: 3 par jour)
- Pas d'accès au bankroll
- Pas de notifications
- Pas d'historique

## 5. CONFIGURATION DOCKER

### 5.1 Services Docker

L'application sera composée de plusieurs services Docker :

1. **mysql** : Base de données MySQL 8.x
2. **redis** : Cache et message broker pour Celery
3. **backend** : API FastAPI
4. **celery-worker** : Workers Celery pour tâches asynchrones
5. **celery-beat** : Scheduler Celery pour tâches planifiées
6. **frontend** : Application React (build production)
7. **streamlit** : Dashboard Streamlit alternatif
8. **nginx** : Reverse proxy (production)

### 5.2 Volumes Docker

- `mysql_data` : Données persistantes MySQL
- `redis_data` : Données persistantes Redis
- `ml_models` : Modèles ML entraînés
- `logs` : Logs applicatifs

## 6. APIS EXTERNES

### 6.1 Turfinfo API

Procédure d'utilisation détaillée disponible dans `Procédure d'utilisation de TurfInfo.txt` (à ajouter au repository).

**Usage principal :**
- Récupération programme officiel PMU
- Informations avant course
- Données historiques

### 6.2 Open-PMU-API

Procédure d'utilisation détaillée disponible dans `Procédure Open-PMU-API.txt` (à ajouter au repository).

**Usage principal :**
- Vérification arrivées officielles
- Contrôle résultats après course
- Validation paris

### 6.3 AspiTurf API

**Note :** Intégré mais désactivé par défaut.

**Usage potentiel :**
- Détails enrichis des partants
- Performances historiques détaillées

### 6.4 Open-Meteo API

**Endpoint :** https://api.open-meteo.com/v1/forecast

**Usage :**
- Conditions météo en temps réel
- Prévisions pour les hippodromes

## 7. SÉCURITÉ

### 7.1 Authentification et Autorisation

- JWT (JSON Web Tokens) pour authentification
- Tokens signés avec clé secrète robuste (256-bit)
- Access Token : expiration 15 minutes
- Refresh Token : expiration 7 jours
- Middleware de vérification des rôles
- Rate limiting sur les endpoints API

### 7.2 Protection des Données

- Hash bcrypt pour les mots de passe (cost factor 12)
- Validation force mot de passe
- HTTPS obligatoire en production
- CORS configuré strictement
- Variables d'environnement pour secrets
- Conformité RGPD

### 7.3 Disclaimer Légal

**Message obligatoire :**
"Application à but éducatif. Les pronostics fournis sont basés sur des analyses statistiques et ne garantissent aucun gain. Pariez de manière responsable."

## 8. ROADMAP DE DÉVELOPPEMENT

### Phase 1 : MVP (8-10 semaines)
- Setup infrastructure et base de données
- Authentification et gestion utilisateurs
- Récupération programme et affichage
- Moteur IA et génération pronostics
- Gestion bankroll et paris simulés
- Dashboard analytique de base

### Phase 2 : Fonctionnalités Avancées (4-6 semaines)
- Mode entraînement
- Notifications Telegram
- Dashboard Streamlit
- Favoris et comparateur
- Pipeline MLOps

### Phase 3 : Production (2 semaines)
- Optimisations performances
- Tests E2E complets
- Déploiement production
- Documentation complète

## 9. AXES D'AMÉLIORATION FUTURS

- Application mobile native (iOS/Android)
- Support courses internationales
- Fonctionnalités sociales/communauté
- Mode dark
- Support multi-langues
- Intégration bookmakers pour paris réels
- API publique pour développeurs tiers
- Analyse vidéo des courses (computer vision)

---

**Version du document :** 1.0
**Date :** 2025-10-29
**Projet :** PronoTurf - Application de Pronostics Hippiques Intelligents
