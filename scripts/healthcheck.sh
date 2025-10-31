#!/bin/bash

# PronoTurf - Script de healthcheck complet
# Vérifie que tous les services sont opérationnels et communiquent correctement

set -e

echo "================================================"
echo "🏇 PronoTurf - Healthcheck des services"
echo "================================================"
echo ""

# Couleurs pour l'affichage
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour attendre qu'un service soit prêt
wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=1

    echo -n "⏳ Attente du démarrage de $service..."

    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps | grep -q "$service.*Up"; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e " ${RED}✗ Timeout${NC}"
    return 1
}

# Fonction pour tester un endpoint HTTP
test_endpoint() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}

    echo -n "Testing $name..."

    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")

    if [ "$response" -eq "$expected_code" ]; then
        echo -e " ${GREEN}✓ ($response)${NC}"
        return 0
    else
        echo -e " ${RED}✗ (Got $response, expected $expected_code)${NC}"
        return 1
    fi
}

# Étape 1: Vérifier que docker-compose est installé
echo "1️⃣  Vérification de l'environnement"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗ docker-compose n'est pas installé${NC}"
    exit 1
fi
echo -e "${GREEN}✓ docker-compose est disponible${NC}"
echo ""

# Étape 2: Vérifier que les services sont démarrés
echo "2️⃣  Vérification des services Docker"
services=("mysql" "redis" "backend" "celery-worker" "celery-beat" "frontend" "streamlit")

for service in "${services[@]}"; do
    if docker-compose ps | grep -q "$service.*Up"; then
        echo -e "${GREEN}✓${NC} $service est démarré"
    else
        echo -e "${RED}✗${NC} $service n'est pas démarré"
        echo "   Conseil: Lancez 'docker-compose up -d' pour démarrer les services"
        exit 1
    fi
done
echo ""

# Étape 3: Test des endpoints de santé
echo "3️⃣  Test des endpoints de santé"

# Test root endpoint
test_endpoint "Backend root" "http://localhost:8000/" 200

# Test health endpoint simple
test_endpoint "Backend health" "http://localhost:8000/health" 200

# Test health avec database
test_endpoint "Backend health/db" "http://localhost:8000/api/v1/health/db" 200

# Test health avec Redis
test_endpoint "Backend health/redis" "http://localhost:8000/api/v1/health/redis" 200

# Test health complet
test_endpoint "Backend health/all" "http://localhost:8000/api/v1/health/all" 200

# Test frontend
test_endpoint "Frontend" "http://localhost:3000/" 200

# Test Streamlit
test_endpoint "Streamlit" "http://localhost:8501/" 200

echo ""

# Étape 4: Test de la communication MySQL
echo "4️⃣  Test de la communication avec MySQL"
echo -n "Testing MySQL connection..."
if docker-compose exec -T mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! pronoturf -e "SELECT 1;" &>/dev/null; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗${NC}"
    exit 1
fi

echo -n "Testing MySQL tables creation..."
table_count=$(docker-compose exec -T mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! pronoturf -e "SHOW TABLES;" 2>/dev/null | wc -l)
if [ "$table_count" -gt 1 ]; then
    echo -e " ${GREEN}✓ ($((table_count - 1)) tables)${NC}"
else
    echo -e " ${YELLOW}⚠ Aucune table trouvée${NC}"
fi
echo ""

# Étape 5: Test de la communication Redis
echo "5️⃣  Test de la communication avec Redis"
echo -n "Testing Redis PING..."
if docker-compose exec -T redis redis-cli ping &>/dev/null; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗${NC}"
    exit 1
fi

echo -n "Testing Redis SET/GET..."
docker-compose exec -T redis redis-cli SET healthcheck_test "$(date +%s)" &>/dev/null
value=$(docker-compose exec -T redis redis-cli GET healthcheck_test 2>/dev/null)
if [ -n "$value" ]; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗${NC}"
    exit 1
fi
echo ""

# Étape 6: Test des logs des services
echo "6️⃣  Vérification des logs (dernières erreurs)"
services_to_check=("backend" "celery-worker" "celery-beat")

for service in "${services_to_check[@]}"; do
    echo -n "Checking $service logs..."
    errors=$(docker-compose logs --tail=50 "$service" 2>&1 | grep -i "error" | wc -l)
    if [ "$errors" -eq 0 ]; then
        echo -e " ${GREEN}✓ Aucune erreur${NC}"
    else
        echo -e " ${YELLOW}⚠ $errors erreur(s) trouvée(s)${NC}"
    fi
done
echo ""

# Étape 7: Résumé final
echo "================================================"
echo "✅ Healthcheck terminé avec succès!"
echo "================================================"
echo ""
echo "Services disponibles:"
echo "  - Frontend:         http://localhost:3000"
echo "  - Backend API:      http://localhost:8000"
echo "  - API Docs:         http://localhost:8000/api/v1/docs"
echo "  - Streamlit:        http://localhost:8501"
echo ""
echo "Endpoints de santé:"
echo "  - Health check:     http://localhost:8000/health"
echo "  - Health DB:        http://localhost:8000/api/v1/health/db"
echo "  - Health Redis:     http://localhost:8000/api/v1/health/redis"
echo "  - Health All:       http://localhost:8000/api/v1/health/all"
echo ""
echo "Pour voir les logs en temps réel:"
echo "  docker-compose logs -f [service_name]"
echo ""
