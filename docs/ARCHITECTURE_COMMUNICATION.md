# Architecture de Communication Inter-Services

## Vue d'ensemble

L'application PronoTurf utilise Docker Compose pour orchestrer plusieurs services qui communiquent entre eux via un réseau Docker bridge dédié nommé `pronoturf-network`.

## Diagramme de Communication

```
┌─────────────────────────────────────────────────────────────┐
│                     Réseau Docker Bridge                     │
│                   (pronoturf-network)                        │
│                                                              │
│  ┌─────────┐         ┌─────────┐         ┌─────────┐       │
│  │Frontend │◄───────►│ Backend │◄───────►│  MySQL  │       │
│  │  :3000  │         │  :8000  │         │  :3306  │       │
│  └─────────┘         └────┬────┘         └─────────┘       │
│                           │                                 │
│                           │                                 │
│                      ┌────▼────┐                            │
│                      │  Redis  │                            │
│                      │  :6379  │                            │
│                      └────┬────┘                            │
│                           │                                 │
│       ┌───────────────────┼───────────────────┐            │
│       │                   │                   │            │
│  ┌────▼────┐         ┌────▼────┐       ┌─────▼────┐       │
│  │ Celery  │         │ Celery  │       │Streamlit │       │
│  │ Worker  │         │  Beat   │       │  :8501   │       │
│  └─────────┘         └─────────┘       └──────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                         ▲
                         │
                    Ports exposés
                    sur localhost
```

## Services et Communication

### 1. MySQL (Port 3306)
**Rôle:** Base de données relationnelle

**Communique avec:**
- ✅ Backend API (lecture/écriture données)
- ✅ Celery Workers (lecture/écriture via tâches async)
- ✅ Streamlit (lecture pour analytics)

**Hostname dans le réseau:** `mysql`

**URL de connexion:**
```
mysql+aiomysql://pronoturf_user:PronoTurf_DB_2025!@mysql:3306/pronoturf
```

**Healthcheck:**
- Interne: `mysqladmin ping`
- Externe: `http://localhost:8000/api/v1/health/db`

### 2. Redis (Port 6379)
**Rôle:** Cache et message broker pour Celery

**Communique avec:**
- ✅ Backend API (cache, sessions)
- ✅ Celery Workers (message queue)
- ✅ Celery Beat (scheduling)

**Hostname dans le réseau:** `redis`

**URL de connexion:**
```
redis://redis:6379/0
```

**Healthcheck:**
- Interne: `redis-cli ping`
- Externe: `http://localhost:8000/api/v1/health/redis`

### 3. Backend API (Port 8000)
**Rôle:** API REST FastAPI

**Communique avec:**
- ✅ MySQL (ORM SQLAlchemy)
- ✅ Redis (cache et Celery broker)
- ✅ Frontend (réponse aux requêtes HTTP)
- ✅ Celery (déclenche des tâches)

**Hostname dans le réseau:** `backend`

**Endpoints importants:**
- Root: `http://backend:8000/`
- Health: `http://backend:8000/health`
- Health DB: `http://backend:8000/api/v1/health/db`
- Health Redis: `http://backend:8000/api/v1/health/redis`
- Health All: `http://backend:8000/api/v1/health/all`
- API Docs: `http://localhost:8000/api/v1/docs`

**Healthcheck:**
```bash
curl http://localhost:8000/health
```

### 4. Frontend React (Port 3000)
**Rôle:** Interface utilisateur

**Communique avec:**
- ✅ Backend API (requêtes HTTP via axios)

**Hostname dans le réseau:** `frontend`

**Configuration API:**
```env
VITE_API_URL=http://localhost:8000/api/v1
```

**Healthcheck:**
```bash
curl http://localhost:3000/
```

### 5. Celery Worker
**Rôle:** Exécution de tâches asynchrones

**Communique avec:**
- ✅ Redis (récupération des tâches)
- ✅ MySQL (accès données via ORM)
- ✅ APIs externes (Turfinfo, Open-PMU, etc.)

**Hostname dans le réseau:** `celery-worker`

**Tâches principales:**
- Synchronisation programme PMU (quotidien à 6h)
- Génération pronostics IA (quotidien à 7h)
- Vérification arrivées (toutes les heures 12h-23h)
- Envoi notifications Telegram

### 6. Celery Beat
**Rôle:** Planification des tâches périodiques

**Communique avec:**
- ✅ Redis (enregistrement des schedules)

**Hostname dans le réseau:** `celery-beat`

**Schedule configuré:**
- `0 6 * * *` - Sync programme
- `0 7 * * *` - Génération pronostics
- `0 12-23 * * *` - Vérification arrivées
- `0 2 * * 1` - Ré-entraînement ML (lundi 2h)

### 7. Streamlit (Port 8501)
**Rôle:** Dashboard analytique

**Communique avec:**
- ✅ MySQL (lecture directe pour analytics)
- ✅ Backend API (optionnel, via HTTP)

**Hostname dans le réseau:** `streamlit`

**Healthcheck:**
```bash
curl http://localhost:8501/
```

## Réseau Docker

**Nom:** `pronoturf-network`
**Type:** bridge
**Driver:** bridge

### Configuration
```yaml
networks:
  pronoturf-network:
    driver: bridge
    name: pronoturf-network
```

### Avantages
- Isolation des services
- Communication via hostnames
- Résolution DNS automatique
- Isolation du réseau hôte

## Ports Exposés

| Service | Port Interne | Port Externe | Description |
|---------|-------------|--------------|-------------|
| MySQL | 3306 | 3306 | Base de données |
| Redis | 6379 | 6379 | Cache/Broker |
| Backend | 8000 | 8000 | API REST |
| Frontend | 3000 | 3000 | Interface web |
| Streamlit | 8501 | 8501 | Dashboard |

## Volumes Persistants

| Volume | Service | Contenu |
|--------|---------|---------|
| `mysql_data` | MySQL | Données de la base |
| `redis_data` | Redis | Cache persistant |
| `./ml_models` | Backend, Streamlit | Modèles ML |
| `./logs` | Backend, Celery | Logs applicatifs |

## Tests de Connectivité

### 1. Test rapide (externe)
```bash
./scripts/healthcheck.sh
```

### 2. Test complet (interne au backend)
```bash
docker-compose exec backend python scripts/test_connectivity.py
```

### 3. Test manuel MySQL
```bash
docker-compose exec mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! -e "SELECT 1;"
```

### 4. Test manuel Redis
```bash
docker-compose exec redis redis-cli ping
```

### 5. Test manuel Backend
```bash
curl http://localhost:8000/api/v1/health/all
```

## Dépannage Communication

### Problème: Service ne peut pas joindre MySQL

**Symptômes:**
```
sqlalchemy.exc.OperationalError: (pymysql.err.OperationalError) (2003, "Can't connect to MySQL server")
```

**Solutions:**
1. Vérifier que MySQL est démarré: `docker-compose ps mysql`
2. Vérifier le healthcheck: `docker-compose ps | grep mysql`
3. Vérifier les logs: `docker-compose logs mysql`
4. Vérifier le réseau: `docker network inspect pronoturf-network`

### Problème: Service ne peut pas joindre Redis

**Symptômes:**
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solutions:**
1. Vérifier que Redis est démarré: `docker-compose ps redis`
2. Tester manuellement: `docker-compose exec redis redis-cli ping`
3. Vérifier les logs: `docker-compose logs redis`

### Problème: Frontend ne peut pas joindre Backend

**Symptômes:**
- Erreurs CORS dans la console
- Requêtes HTTP échouent

**Solutions:**
1. Vérifier la config CORS dans `backend/.env`
2. Vérifier que Backend est accessible: `curl http://localhost:8000/health`
3. Vérifier la config Frontend: `frontend/.env` → `VITE_API_URL`

### Problème: Les containers ne voient pas les autres

**Solutions:**
1. Recréer le réseau:
   ```bash
   docker-compose down
   docker network prune
   docker-compose up -d
   ```

2. Vérifier le réseau Docker:
   ```bash
   docker network inspect pronoturf-network
   ```

3. Vérifier la résolution DNS:
   ```bash
   docker-compose exec backend ping -c 2 mysql
   docker-compose exec backend ping -c 2 redis
   ```

## Sécurité Réseau

### Bonnes pratiques implémentées

1. **Réseau isolé**: Services sur réseau bridge dédié
2. **Pas d'exposition inutile**: Seuls les ports nécessaires sont exposés sur l'hôte
3. **Communication interne**: Services communiquent via hostnames internes
4. **Healthchecks**: Tous les services critiques ont des healthchecks
5. **Restart policy**: `unless-stopped` pour haute disponibilité

### Recommandations pour production

1. **Ne pas exposer MySQL et Redis** sur l'hôte (enlever les ports dans docker-compose.yml)
2. **Utiliser des secrets Docker** pour les mots de passe
3. **Activer SSL/TLS** pour MySQL et Redis
4. **Utiliser un reverse proxy** (Nginx) pour tout exposer via HTTPS
5. **Limiter les ressources** (CPU, mémoire) pour chaque service

## Monitoring

### Logs en temps réel
```bash
# Tous les services
docker-compose logs -f

# Service spécifique
docker-compose logs -f backend
docker-compose logs -f mysql
docker-compose logs -f redis
```

### État des services
```bash
docker-compose ps
```

### Inspection réseau
```bash
docker network inspect pronoturf-network
```

### Ressources utilisées
```bash
docker stats
```
