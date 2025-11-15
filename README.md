# PronoTurf - Application de Pronostics Hippiques Intelligents

> **⚠️ Licence** : Ce dépôt est soumis à la licence privée PronoTurf AI. Toute copie, diffusion, modification ou création de travaux dérivés est strictement interdite sans autorisation écrite préalable.

## Introduction

La version 1 de PronoTurf est désormais officielle et prête pour la mise en production. L'application couvre l'ensemble de la chaîne de valeur : ingestion de données hippiques multi-sources, scoring prédictif, explicabilité, gestion de bankroll, supervision et diffusion des alertes. Cette livraison consolide les apprentissages des sprints précédents et fournit un socle stable pour les futures évolutions.

## Fonctionnalités Clés

### Intelligence Artificielle & Données
- Modèle Gradient Boosting optimisé pour le classement probabiliste des chevaux.
- Utilisation de SHAP pour expliciter les facteurs influençant chaque pronostic.
- Pipeline de préparation de données multi-sources (Aspiturf, TurfInfo, Open-PMU) avec stratégies de repli.
- Calculs avancés : probabilités de victoire/podium, value bets, suivi de la forme et de la volatilité.

### Expérience Utilisateur & Conseils
- Gestion de bankroll (Kelly Criterion, Flat Betting, Martingale) avec alertes de risque.
- Mode entraînement sur historique pour valider les stratégies.
- Interface web responsive (React + Tailwind) avec dashboards analytiques et filtrage contextuel.
- Notifications temps réel via Telegram et email.

### Opérations & Monitoring
- Tableaux de bord Streamlit pour l'analyse de la précision, du ROI, du drift et de la calibration.
- Suivi synthétique des jalons opérationnels, risques et indicateurs de disponibilité.
- Journalisation centralisée et tâches Celery pour l'orchestration asynchrone.

## Architecture Technique

### Backend
- **Python 3.11** & **FastAPI** pour l'API REST.
- **SQLAlchemy** (async) et **MySQL 8** pour la persistance.
- **Celery** & **Redis** pour les traitements asynchrones et le cache.
- **scikit-learn** & **SHAP** pour la modélisation.
- **APScheduler**, **httpx**, **tenacity** pour les intégrations externes fiables.

### Frontend
- **React 18** avec **TypeScript** et **Vite**.
- **React Router**, **React Query** et **Axios** pour la navigation et les appels API.
- **Tailwind CSS** & **Plotly** pour le rendu graphique et la mise en forme.

### Visualisation Alternative
- **Streamlit** pour les analyses expertes et le pilotage data-driven.

### Infrastructure & Déploiement
- **Docker** & **Docker Compose** pour orchestrer MySQL, Redis, Backend, Frontend, Worker et Streamlit.
- **Nginx** comme reverse proxy (modèle production) et support HTTPS.
- Scripts d'initialisation de base de données (`database/init.sql`, `database/seed.sql`).

### APIs Externes
1. **Aspiturf (source principale)** : fichiers CSV complets (chevaux, jockeys, historiques).
2. **TurfInfo API (fallback)** : programme PMU temps réel sans clé d'authentification.
3. **Open-PMU API (résultats)** : récupération des arrivées officielles et rapports.

Les paramètres d'accès sont configurables via variables d'environnement, avec repli automatique en cas d'indisponibilité d'une source.

## Déploiement & Exécution

### Prérequis
- Docker & Docker Compose
- Accès aux fichiers CSV Aspiturf (optionnel mais recommandé)
- Variables d'environnement secrètes (clé Telegram, SMTP, etc.)

### Démarrage local (mode production simulé)
```bash
docker compose build
docker compose up -d
```
Les services exposés :
- API FastAPI : `http://localhost:8000/docs`
- Frontend React : `http://localhost:3000`
- Streamlit : `http://localhost:8501`

### Configuration essentielle
Créez un fichier `.env` à la racine ou exportez les variables suivantes :
- `MYSQL_ROOT_PASSWORD`, `MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD`
- `SECRET_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES`, `REFRESH_TOKEN_EXPIRE_DAYS`
- `ASPITURF_API_KEY`, `ASPITURF_API_URL`
- `TURFINFO_OFFLINE_URL`, `TURFINFO_ONLINE_URL`
- `OPENPMU_API_URL`
- `TELEGRAM_BOT_TOKEN`, `SMTP_*` pour les notifications

Des valeurs par défaut existent pour l'environnement local ; adaptez-les avant tout déploiement réel.

## Tests & Qualité

### Vérification du backend
```bash
cd backend
poetry install  # si l'environnement virtuel n'est pas déjà configuré
poetry run pytest
```

### Vérification du frontend
```bash
cd frontend
npm install
npm run build
```

### Contrôle de la configuration Docker
```bash
docker compose config
```

### Dépannage MySQL

Si le conteneur `mysql` se bloque avec des messages du type :

```
Cannot create redo log files because data files are corrupt or the database was not shut down cleanly
```

c'est que le volume `mysql_data` contient des fichiers InnoDB corrompus (souvent après un arrêt forcé ou un changement de
machine). Vous pouvez réinitialiser proprement la base locale (les données seront recréées à partir de `database/init.sql` et
`database/seed.sql`) via :

```bash
./scripts/reset_mysql.sh
```

Le script exécute `docker compose down --volumes --remove-orphans`, supprime le volume `mysql_data` puis relance uniquement MySQL.
Une fois le service `mysql` en bonne santé, redémarrez le reste de la stack avec `docker compose up -d`.

## Sécurité

Bonnes pratiques en place :
- Authentification JWT avec refresh tokens.
- Hashage des mots de passe via bcrypt (cost 12).
- Rate limiting, CORS strict, validation Pydantic.
- Segmentation des secrets via variables d'environnement et chiffrement TLS attendu en production.
- Processus de surveillance du drift et alertes sur dérives de données.

**Avertissement légal** : les pronostics sont fournis à titre informatif. Aucun gain n'est garanti et les paris comportent des risques. Respectez la législation en vigueur dans votre juridiction.

## Support

- Email : support@pronoturf.ai
- Documentation : `CDC.md`, `QUICKSTART.md`, `README_API_CONFIGURATION.md`

Pour toute demande d'accès ou d'audit, contacter directement l'équipe produit. Les contributions externes, forks et redistributions sont interdits.

## Licence

Ce projet est protégé par la licence privée **PronoTurf AI Private License v1.0**. L'utilisation est limitée au périmètre défini contractuellement. Voir le fichier `LICENSE` pour les conditions complètes.

## Axes d'Amélioration

### Court Terme
- [ ] Tests unitaires complets (couverture > 80%)
- [ ] Documentation API complète
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

---

**Version** : 1.0.0

**Dernière mise à jour** : 31 Octobre 2025
