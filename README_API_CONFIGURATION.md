# Configuration des APIs PronoTurf

## Stratégie Multi-API

PronoTurf utilise une stratégie multi-API pour maximiser la qualité des données:

### 1. AspiTurf - API PRINCIPALE ⭐
**Status**: À configurer (nécessite clé API)

**Rôle**: Source principale de données complètes
- Informations détaillées des partants
- Statistiques approfondies
- Performances historiques enrichies
- Données en temps réel

**Configuration requise**:
```env
ASPITURF_API_KEY=your-aspiturf-api-key-here
ASPITURF_API_URL=https://api.aspiturf.com
ASPITURF_ENABLED=true
```

**Documentation manquante**:
- 📝 Créer fichier `docs/Procédure Aspiturf.txt` avec:
  - URL base de l'API
  - Endpoints disponibles
  - Format des requêtes/réponses
  - Authentification (headers, params)
  - Exemples d'appels

### 2. TurfInfo API - COMPLÉMENTAIRE ✅
**Status**: Configurée (gratuite, sans clé)

**Rôle**: Complète les données Aspiturf
- Programme officiel PMU
- Partants et cotes
- Performances détaillées
- Rapports PMU

**Endpoints**:
- **OFFLINE**: `https://offline.turfinfo.api.pmu.fr/rest/client/7`
  - Optimisé pour vitesse
  - Données essentielles

- **ONLINE**: `https://online.turfinfo.api.pmu.fr/rest/client/61`
  - Plus de détails
  - Recommandé pour webapp

**Format date**: JJMMAAAA (ex: 30102025)

**Exemples**:
```
GET /programme/{JJMMAAAA}
GET /programme/{JJMMAAAA}/R{num}/C{num}/participants
GET /programme/{JJMMAAAA}/R{num}/C{num}/performances-detaillees/pretty
GET /programme/{JJMMAAAA}/R{num}/C{num}/rapports-definitifs
```

**Documentation**: `docs/Procédure TurfInfo.txt`

### 3. Open-PMU API - COMPLÉMENTAIRE ✅
**Status**: Configurée (gratuite, sans clé)

**Rôle**: Résultats officiels après-course
- Arrivées officielles
- Rapports PMU complets
- Non-partants
- Gains par type de pari

**URL Base**: `https://open-pmu-api.vercel.app/api`

**Format date**: DD/MM/YYYY (ex: 30/10/2025)

**Endpoints**:
```
GET /arrivees?date={DD/MM/YYYY}
GET /arrivees?date={DD/MM/YYYY}&hippo={name}
GET /arrivees?date={DD/MM/YYYY}&prix={name}
```

**Documentation**: `docs/Procédure Open-PMU-API.txt`

## Architecture d'Intégration

```
┌─────────────────────────────────────┐
│      PMUService (Orchestrateur)     │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────┐  API PRINCIPALE   │
│  │  Aspiturf   │─────────────────► │
│  │   Client    │  Données complètes│
│  └─────────────┘                   │
│                                     │
│  ┌─────────────┐  COMPLÉMENTAIRE   │
│  │  TurfInfo   │─────────────────► │
│  │   Client    │  Programme, cotes │
│  └─────────────┘                   │
│                                     │
│  ┌─────────────┐  COMPLÉMENTAIRE   │
│  │  Open-PMU   │─────────────────► │
│  │   Client    │  Résultats        │
│  └─────────────┘                   │
│                                     │
└─────────────────────────────────────┘
```

## Logique de Fallback

```python
def get_race_data(race_id):
    # 1. Essayer Aspiturf (principal)
    if ASPITURF_ENABLED:
        try:
            return aspiturf_client.get_race(race_id)
        except Exception:
            logger.warning("Aspiturf failed, fallback to TurfInfo")

    # 2. Fallback TurfInfo
    if TURFINFO_ENABLED:
        return turfinfo_client.get_partants_course(...)

    raise APIError("No API available")
```

## Configuration Docker

**Fichiers mis à jour**:
- ✅ `.env.example` - Template avec toutes les URLs
- ✅ `backend/.env.example` - Config backend détaillée
- ✅ `docker-compose.yml` - Variables pour tous les services
- ✅ `backend/app/core/config.py` - Settings Python

**Valeurs par défaut**:
```yaml
ASPITURF_ENABLED=true      # API principale
TURFINFO_ENABLED=true      # Complément
OPENPMU_ENABLED=true       # Résultats
```

## Services Implémentés

### Clients API
- ✅ `TurfinfoClient` - `/backend/app/services/turfinfo_client.py`
- ✅ `OpenPMUClient` - `/backend/app/services/open_pmu_client.py`
- ⏳ `AspiturfClient` - **EN ATTENTE DE DOCUMENTATION**

### Service Principal
- ✅ `PMUService` - Orchestrateur multi-API
  - Actuellement utilise TurfInfo et Open-PMU
  - Prêt pour intégration Aspiturf

## Action Requise 🚨

**Pour finaliser l'intégration Aspiturf**:

1. **Obtenir clé API Aspiturf** si pas encore fait

2. **Créer** `docs/Procédure Aspiturf.txt` avec:
   ```
   URL Base API: https://api.aspiturf.com

   Authentification:
   - Type: [API Key / Bearer Token / Basic Auth]
   - Header: [X-API-Key / Authorization]
   - Format: [...]

   Endpoints:
   - Programme jour: GET /programme/{date}
   - Partants: GET /race/{id}/runners
   - Performances: GET /horse/{id}/performances
   - Résultats: GET /race/{id}/results

   Format date: [YYYY-MM-DD / DD-MM-YYYY / timestamp]

   Exemples de réponses JSON:
   {...}
   ```

3. **Configurer** `.env`:
   ```env
   ASPITURF_API_KEY=votre_cle_ici
   ```

4. **Développer** `AspiturfClient` basé sur la procédure

5. **Refactoriser** `PMUService` pour priorité Aspiturf

## Retry & Cache

**Retry automatique** (tenacity):
- 3 tentatives
- Backoff exponentiel (2s, 4s, 8s)
- Sur TimeoutException et NetworkError

**Cache recommandé**:
- TurfInfo: 60 minutes (données avant-course)
- Open-PMU: 24 heures (résultats stables)
- Aspiturf: À définir selon la doc

## Variables d'Environnement Complètes

### Backend
```env
# AspiTurf - API PRINCIPALE
ASPITURF_API_KEY=your-key
ASPITURF_API_URL=https://api.aspiturf.com
ASPITURF_ENABLED=true

# TurfInfo - COMPLÉMENTAIRE (sans clé)
TURFINFO_OFFLINE_URL=https://offline.turfinfo.api.pmu.fr/rest/client/7
TURFINFO_ONLINE_URL=https://online.turfinfo.api.pmu.fr/rest/client/61
TURFINFO_ENABLED=true

# Open-PMU - COMPLÉMENTAIRE (sans clé)
OPENPMU_API_URL=https://open-pmu-api.vercel.app/api
OPENPMU_ENABLED=true

# Notifications
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_ENABLED=false
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email
SMTP_PASSWORD=your-password
SMTP_FROM_EMAIL=noreply@pronoturf.ai
EMAIL_ENABLED=false
FRONTEND_URL=http://localhost:3000
```

### Docker Compose
Toutes les variables sont automatiquement passées aux services:
- backend
- celery-worker
- celery-beat

## Tests

**Une fois Aspiturf configuré**:
```bash
# Test du client
python backend/scripts/test_api_clients.py --test aspiturf

# Test complet
python backend/scripts/test_api_clients.py --test all
```

---

**Dernière mise à jour**: 30 Octobre 2025
**Version**: 0.4.0-beta
