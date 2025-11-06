#!/bin/bash
# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.


# PronoTurf - Script de démarrage complet
# Démarre tous les services avec Docker Compose et vérifie leur santé

set -e

echo "================================================"
echo "🏇 PronoTurf - Démarrage de l'application"
echo "================================================"
echo ""

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Erreur: docker-compose.yml non trouvé"
    echo "   Veuillez exécuter ce script depuis la racine du projet"
    exit 1
fi

# Vérifier que les fichiers .env existent
echo "1️⃣  Vérification de la configuration"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ Fichier .env manquant, copie depuis .env.example${NC}"
    cp .env.example .env
fi

if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}⚠ Fichier backend/.env manquant, copie depuis backend/.env.example${NC}"
    cp backend/.env.example backend/.env
fi

if [ ! -f "frontend/.env" ]; then
    echo -e "${YELLOW}⚠ Fichier frontend/.env manquant, copie depuis frontend/.env.example${NC}"
    cp frontend/.env.example frontend/.env
fi
echo -e "${GREEN}✓ Fichiers de configuration prêts${NC}"
echo ""

# Arrêter les services existants si nécessaire
echo "2️⃣  Nettoyage des services existants"
if docker-compose ps | grep -q "Up"; then
    echo "   Arrêt des services en cours..."
    docker-compose down
fi
echo -e "${GREEN}✓ Nettoyage terminé${NC}"
echo ""

# Construire les images
echo "3️⃣  Construction des images Docker"
echo "   Cela peut prendre quelques minutes la première fois..."
docker-compose build --no-cache
echo -e "${GREEN}✓ Images construites${NC}"
echo ""

# Démarrer les services
echo "4️⃣  Démarrage des services"
docker-compose up -d
echo -e "${GREEN}✓ Services démarrés${NC}"
echo ""

# Attendre que les services soient prêts
echo "5️⃣  Attente que les services soient prêts"
echo "   MySQL..."
timeout=60
counter=0
until docker-compose exec -T mysql mysqladmin ping -h localhost --silent 2>/dev/null || [ $counter -eq $timeout ]; do
    printf '.'
    sleep 2
    counter=$((counter + 2))
done

if [ $counter -eq $timeout ]; then
    echo -e "\n${YELLOW}⚠ MySQL timeout, mais continue...${NC}"
else
    echo -e "\n${GREEN}✓ MySQL prêt${NC}"
fi

echo "   Redis..."
counter=0
until docker-compose exec -T redis redis-cli ping &>/dev/null || [ $counter -eq $timeout ]; do
    printf '.'
    sleep 2
    counter=$((counter + 2))
done

if [ $counter -eq $timeout ]; then
    echo -e "\n${YELLOW}⚠ Redis timeout, mais continue...${NC}"
else
    echo -e "\n${GREEN}✓ Redis prêt${NC}"
fi

echo "   Backend API..."
counter=0
until curl -s http://localhost:8000/health &>/dev/null || [ $counter -eq $timeout ]; do
    printf '.'
    sleep 2
    counter=$((counter + 2))
done

if [ $counter -eq $timeout ]; then
    echo -e "\n${YELLOW}⚠ Backend timeout, mais continue...${NC}"
else
    echo -e "\n${GREEN}✓ Backend prêt${NC}"
fi

echo ""

# Initialiser la base de données si nécessaire
echo "6️⃣  Vérification de la base de données"
table_count=$(docker-compose exec -T mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! pronoturf -e "SHOW TABLES;" 2>/dev/null | wc -l)

if [ "$table_count" -le 1 ]; then
    echo "   Aucune table trouvée, initialisation..."
    docker-compose exec -T mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! pronoturf < database/init.sql
    echo -e "${GREEN}✓ Base de données initialisée${NC}"

    echo "   Import des données de test..."
    docker-compose exec -T mysql mysql -u pronoturf_user -pPronoTurf_DB_2025! pronoturf < database/seed.sql
    echo -e "${GREEN}✓ Données de test importées${NC}"
else
    echo -e "${GREEN}✓ Base de données déjà initialisée ($((table_count - 1)) tables)${NC}"
fi
echo ""

# Afficher les logs des services
echo "7️⃣  État des services"
docker-compose ps
echo ""

# Affichage final
echo "================================================"
echo "✅ Démarrage terminé avec succès!"
echo "================================================"
echo ""
echo -e "${BLUE}Services disponibles:${NC}"
echo "  🌐 Frontend:        http://localhost:3000"
echo "  🔧 Backend API:     http://localhost:8000"
echo "  📚 API Docs:        http://localhost:8000/api/v1/docs"
echo "  📊 Streamlit:       http://localhost:8501"
echo ""
echo -e "${BLUE}Comptes de test (mot de passe: Password123!):${NC}"
echo "  👨‍💼 Admin:           admin@pronoturf.ai"
echo "  👤 Abonné:          subscriber@example.com"
echo "  👁  Invité:          guest@example.com"
echo ""
echo -e "${BLUE}Commandes utiles:${NC}"
echo "  📋 Voir les logs:   docker-compose logs -f [service]"
echo "  🔍 Healthcheck:     ./scripts/healthcheck.sh"
echo "  ⏹  Arrêter:         docker-compose down"
echo "  🔄 Redémarrer:      docker-compose restart [service]"
echo ""
echo "Pour vérifier que tout fonctionne correctement, exécutez:"
echo "  ./scripts/healthcheck.sh"
echo ""