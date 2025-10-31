# Scripts PronoTurf

Ce dossier contient des scripts utilitaires pour gérer et tester l'application PronoTurf.

## Scripts disponibles

### 🚀 start.sh
Script de démarrage complet de l'application.

**Usage:**
```bash
./scripts/start.sh
```

**Actions:**
- Vérifie la configuration (.env)
- Construit les images Docker
- Démarre tous les services
- Initialise la base de données si nécessaire
- Import les données de test

### 🔍 healthcheck.sh
Script de vérification de la santé de tous les services.

**Usage:**
```bash
./scripts/healthcheck.sh
```

**Vérifications:**
- État des containers Docker
- Endpoints HTTP (Backend, Frontend, Streamlit)
- Connectivité MySQL
- Connectivité Redis
- Logs des services

### 🔗 test_connectivity.py
Script Python de test de communication inter-services.

**Usage (depuis le container backend):**
```bash
docker-compose exec backend python /app/scripts/test_connectivity.py
```

**Tests effectués:**
- Connexion MySQL (requêtes, version, nombre de tables)
- Connexion Redis (PING, SET/GET, infos)
- API Backend (endpoints root et health)
- Frontend React (disponibilité)
- Streamlit (disponibilité)

## Ordre d'exécution recommandé

1. **Première installation:**
   ```bash
   ./scripts/start.sh
   # Patientez quelques minutes le temps que tous les services démarrent
   ```

2. **Vérifier que tout fonctionne:**
   ```bash
   ./scripts/healthcheck.sh
   ```

3. **Test de communication avancé (optionnel):**
   ```bash
   docker-compose exec backend python scripts/test_connectivity.py
   ```

## Dépannage

### Les services ne démarrent pas
```bash
# Voir les logs
docker-compose logs -f [service_name]

# Redémarrer un service spécifique
docker-compose restart [service_name]

# Tout arrêter et redémarrer proprement
docker-compose down
./scripts/start.sh
```

### La base de données n'est pas initialisée
```bash
# Réinitialiser la base de données
docker-compose exec -T mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! pronoturf < database/init.sql
docker-compose exec -T mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! pronoturf < database/seed.sql
```

### Les containers ne peuvent pas communiquer
```bash
# Vérifier le réseau Docker
docker network inspect pronoturf-network

# Recréer le réseau
docker-compose down
docker network prune
docker-compose up -d
```

## Variables d'environnement

Les scripts utilisent les variables définies dans les fichiers `.env`:
- `.env` - Configuration globale
- `backend/.env` - Configuration backend
- `frontend/.env` - Configuration frontend

Assurez-vous que ces fichiers existent avant de lancer les scripts.
