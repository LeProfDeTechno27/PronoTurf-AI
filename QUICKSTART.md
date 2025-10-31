# 🚀 Guide de Démarrage Rapide - PronoTurf

Ce guide vous permet de démarrer l'application PronoTurf en quelques minutes.

## Prérequis

- Docker 20.10+ installé
- Docker Compose 2.0+ installé
- 4 GB de RAM disponible minimum
- Ports libres : 3000, 8000, 8501, 3306, 6379

## Installation en 3 étapes

### 1️⃣ Cloner le repository

```bash
git clone https://github.com/LeProfDeTechno27/Prono_Gold.git
cd Prono_Gold
```

### 2️⃣ Configurer les variables d'environnement

Le script de démarrage créera automatiquement les fichiers `.env` manquants à partir des `.env.example`.

**Optionnel** : Si vous voulez personnaliser la configuration :

```bash
# Copier et éditer les fichiers d'exemple
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# Éditer avec votre éditeur préféré
nano .env  # ou vim, code, etc.
```

**Configuration minimale** : Les valeurs par défaut fonctionnent pour le développement local.

### 3️⃣ Démarrer l'application

```bash
./scripts/start.sh
```

Ce script va :
- ✅ Vérifier la configuration
- ✅ Construire les images Docker
- ✅ Démarrer tous les services (MySQL, Redis, Backend, Frontend, Celery, Streamlit)
- ✅ Initialiser la base de données
- ✅ Importer les données de test

**Durée** : 3-5 minutes la première fois (téléchargement des images et build)

## Vérifier que tout fonctionne

Une fois le démarrage terminé, exécutez :

```bash
./scripts/healthcheck.sh
```

Ce script vérifie :
- ✅ État des containers Docker
- ✅ Connectivité MySQL
- ✅ Connectivité Redis
- ✅ Endpoints HTTP (Backend, Frontend, Streamlit)
- ✅ Logs des services

## Accéder à l'application

Une fois tous les services démarrés :

### 🌐 Frontend (Interface principale)
**URL** : http://localhost:3000

Interface utilisateur moderne avec React + TypeScript.

### 🔧 Backend API
**URL** : http://localhost:8000
**Documentation** : http://localhost:8000/api/v1/docs

API REST FastAPI avec documentation Swagger interactive.

### 📊 Dashboard Streamlit
**URL** : http://localhost:8501

Dashboard analytique interactif pour visualiser les statistiques.

## Comptes de test

Utilisez ces comptes pour vous connecter :

| Rôle | Email | Mot de passe | Accès |
|------|-------|--------------|-------|
| **Administrateur** | `admin@pronoturf.ai` | `Password123!` | Accès complet + admin |
| **Abonné** | `subscriber@example.com` | `Password123!` | Accès complet |
| **Invité** | `guest@example.com` | `Password123!` | Accès limité |

## Commandes utiles

### Voir les logs en temps réel

```bash
# Tous les services
docker-compose logs -f

# Service spécifique
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f mysql
```

### Arrêter l'application

```bash
docker-compose down
```

### Redémarrer un service spécifique

```bash
docker-compose restart backend
docker-compose restart frontend
```

### Accéder à un container

```bash
# Backend
docker-compose exec backend bash

# MySQL
docker-compose exec mysql bash

# Redis
docker-compose exec redis sh
```

### Réinitialiser complètement

```bash
# Arrêter et supprimer tous les containers, volumes et réseaux
docker-compose down -v

# Redémarrer from scratch
./scripts/start.sh
```

## Tests de communication inter-services

### Test rapide (externe)

```bash
./scripts/healthcheck.sh
```

### Test complet (interne au backend)

```bash
docker-compose exec backend python scripts/test_connectivity.py
```

### Test manuel des endpoints

```bash
# Health check simple
curl http://localhost:8000/health

# Test de tous les services
curl http://localhost:8000/api/v1/health/all

# Test MySQL
curl http://localhost:8000/api/v1/health/db

# Test Redis
curl http://localhost:8000/api/v1/health/redis
```

## Dépannage

### Les services ne démarrent pas

```bash
# Voir les logs pour identifier le problème
docker-compose logs

# Vérifier l'état des containers
docker-compose ps

# Redémarrer proprement
docker-compose down
docker-compose up -d
```

### Port déjà utilisé

Si vous avez une erreur "port already allocated" :

```bash
# Identifier quel processus utilise le port
# Exemple pour le port 8000
sudo lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Modifier le port dans docker-compose.yml ou arrêter le processus
```

### La base de données n'est pas initialisée

```bash
# Vérifier si les tables existent
docker-compose exec mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! pronoturf -e "SHOW TABLES;"

# Réinitialiser manuellement
docker-compose exec -T mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! pronoturf < database/init.sql
docker-compose exec -T mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! pronoturf < database/seed.sql
```

### Les containers ne peuvent pas communiquer

```bash
# Recréer le réseau Docker
docker-compose down
docker network prune -f
docker-compose up -d

# Vérifier le réseau
docker network inspect pronoturf-network
```

## Prochaines étapes

Maintenant que l'application est opérationnelle :

1. 📖 Consultez la [documentation complète](README.md)
2. 📚 Lisez le [cahier des charges](CDC.md)
3. 🏗️ Explorez l'[architecture de communication](docs/ARCHITECTURE_COMMUNICATION.md)
4. 💻 Commencez à développer de nouvelles fonctionnalités !

## Support

Pour toute question ou problème :

1. Consultez la [documentation d'architecture](docs/ARCHITECTURE_COMMUNICATION.md)
2. Vérifiez les [scripts de test](scripts/README.md)
3. Ouvrez une issue sur GitHub

## Structure du projet

```
Prono_Gold/
├── backend/           # API FastAPI (Python)
├── frontend/          # Interface React (TypeScript)
├── streamlit/         # Dashboard Streamlit (Python)
├── database/          # Scripts SQL d'initialisation
├── docs/              # Documentation technique
├── scripts/           # Scripts utilitaires
├── ml_models/         # Modèles ML (à venir)
├── logs/              # Logs applicatifs
├── docker-compose.yml # Orchestration des services
└── README.md          # Documentation complète
```

## Développement

Pour le développement actif :

```bash
# Backend : mode hot-reload activé automatiquement
docker-compose logs -f backend

# Frontend : mode dev avec Vite hot-reload
docker-compose logs -f frontend

# Pour modifier le code, éditez simplement les fichiers
# Les changements seront appliqués automatiquement
```

---

**PronoTurf v0.1.0** - Application de pronostics hippiques intelligents
