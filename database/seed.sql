-- PronoTurf Database Seed Script
-- Insert test data for development

SET NAMES utf8mb4;

-- ============================
-- Seed Users
-- ============================
-- Password for all test users: "Password123!"
-- Hash generated with bcrypt (cost factor 12)
INSERT INTO users (email, password_hash, first_name, last_name, role, initial_bankroll, current_bankroll, preferred_strategy, is_active) VALUES
('admin@pronoturf.ai', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU7fOr4hQd8C', 'Admin', 'PronoTurf', 'admin', 10000.00, 10000.00, 'kelly', TRUE),
('subscriber@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU7fOr4hQd8C', 'Jean', 'Dupont', 'subscriber', 1000.00, 1000.00, 'flat', TRUE),
('guest@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU7fOr4hQd8C', 'Marie', 'Martin', 'guest', 500.00, 500.00, 'flat', TRUE),
('test@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU7fOr4hQd8C', 'Test', 'User', 'subscriber', 2000.00, 2000.00, 'martingale', TRUE);

-- ============================
-- Seed User Preferences
-- ============================
INSERT INTO user_preferences (user_id, favorite_hippodromes, favorite_disciplines, value_bet_threshold, notification_email, notification_telegram, notification_types) VALUES
(1, '["LONGCHAMP", "VINCENNES", "CHANTILLY"]', '["plat", "trot_attele"]', 10.00, TRUE, TRUE, '["pronostic", "value_bet", "daily_report"]'),
(2, '["LONGCHAMP", "DEAUVILLE"]', '["plat"]', 5.00, TRUE, FALSE, '["pronostic", "value_bet"]'),
(3, '["VINCENNES"]', '["trot_attele"]', 3.00, FALSE, FALSE, '[]'),
(4, '["CHANTILLY", "FONTAINEBLEAU"]', '["plat", "haies"]', 7.50, TRUE, FALSE, '["value_bet", "result"]');

-- ============================
-- Seed Hippodromes (Principaux hippodromes français)
-- ============================
INSERT INTO hippodromes (code, name, city, country, track_type, track_length, track_surface, latitude, longitude) VALUES
-- Paris et région parisienne
('LONGCHAMP', 'Hippodrome de Longchamp', 'Paris', 'France', 'plat', 2400, 'pelouse', 48.8586, 2.2236),
('AUTEUIL', 'Hippodrome d\'Auteuil', 'Paris', 'France', 'obstacles', 2200, 'pelouse', 48.8467, 2.2500),
('VINCENNES', 'Hippodrome de Vincennes', 'Paris', 'France', 'trot', 1600, 'piste', 48.8281, 2.4514),
('MAISONS-LAFFITTE', 'Hippodrome de Maisons-Laffitte', 'Maisons-Laffitte', 'France', 'plat', 1800, 'pelouse', 48.9492, 2.1486),
('SAINT-CLOUD', 'Hippodrome de Saint-Cloud', 'Saint-Cloud', 'France', 'mixte', 2000, 'pelouse', 48.8436, 2.2125),
('CHANTILLY', 'Hippodrome de Chantilly', 'Chantilly', 'France', 'plat', 2400, 'pelouse', 49.1933, 2.4708),
('ENGHIEN', 'Hippodrome d\'Enghien', 'Enghien-les-Bains', 'France', 'trot', 1200, 'piste', 48.9711, 2.3031),

-- Province
('DEAUVILLE', 'Hippodrome de Deauville', 'Deauville', 'France', 'mixte', 2000, 'pelouse', 49.3575, 0.0800),
('CLAIREFONTAINE', 'Hippodrome de Clairefontaine', 'Deauville', 'France', 'obstacles', 1800, 'pelouse', 49.3636, 0.1028),
('LYON-PARILLY', 'Hippodrome de Lyon-Parilly', 'Lyon', 'France', 'mixte', 2000, 'piste', 45.7203, 4.8889),
('CAGNES-SUR-MER', 'Hippodrome de Cagnes-sur-Mer', 'Cagnes-sur-Mer', 'France', 'mixte', 1500, 'piste', 43.6633, 7.1483),
('MARSEILLE-BORELY', 'Hippodrome de Marseille-Borély', 'Marseille', 'France', 'plat', 1600, 'piste', 43.2567, 5.3767),
('BORDEAUX-LE-BOUSCAT', 'Hippodrome de Bordeaux-Le Bouscat', 'Bordeaux', 'France', 'mixte', 1900, 'pelouse', 44.8589, -0.5997),
('TOULOUSE', 'Hippodrome de Toulouse', 'Toulouse', 'France', 'mixte', 1800, 'piste', 43.5789, 1.4086),
('FONTAINEBLEAU', 'Hippodrome de Fontainebleau', 'Fontainebleau', 'France', 'plat', 1600, 'pelouse', 48.4089, 2.7019);

-- ============================
-- Seed Jockeys (Quelques jockeys célèbres)
-- ============================
INSERT INTO jockeys (official_id, first_name, last_name, birth_date, nationality, weight, career_wins, career_places, career_starts) VALUES
('JOCKEY001', 'Christophe', 'Soumillon', '1981-06-04', 'Belgique', 54.0, 3250, 5800, 15000),
('JOCKEY002', 'Mickael', 'Barzalona', '1990-09-26', 'France', 53.5, 1850, 3500, 9000),
('JOCKEY003', 'Pierre-Charles', 'Boudot', '1992-02-03', 'France', 54.5, 1650, 3200, 8500),
('JOCKEY004', 'Olivier', 'Peslier', '1973-01-12', 'France', 55.0, 4200, 7500, 18000),
('JOCKEY005', 'Alexis', 'Badel', '1989-03-15', 'France', 53.0, 1450, 2800, 7500);

-- ============================
-- Seed Trainers (Quelques entraîneurs célèbres)
-- ============================
INSERT INTO trainers (official_id, first_name, last_name, stable_name, nationality, career_wins, career_places, career_starts) VALUES
('TRAINER001', 'André', 'Fabre', 'Écurie André Fabre', 'France', 8500, 15000, 35000),
('TRAINER002', 'Jean-Claude', 'Rouget', 'Écurie Jean-Claude Rouget', 'France', 6200, 12000, 28000),
('TRAINER003', 'Francis-Henri', 'Graffard', 'Écurie Francis-Henri Graffard', 'France', 1800, 3500, 8000),
('TRAINER004', 'Christophe', 'Ferland', 'Écurie Christophe Ferland', 'France', 2200, 4200, 10000),
('TRAINER005', 'Fabrice', 'Chappet', 'Écurie Fabrice Chappet', 'France', 3100, 6000, 14000);

-- ============================
-- Seed Horses (Quelques chevaux de test)
-- ============================
INSERT INTO horses (official_id, name, birth_year, gender, coat_color, breed, sire, dam, owner) VALUES
('HORSE001', 'ÉCLAIR DE FEU', 2020, 'male', 'Alezan', 'Pur-sang', 'Dubawi', 'Flamme d\'Or', 'Ecurie Test 1'),
('HORSE002', 'BELLE ÉTOILE', 2019, 'female', 'Bai', 'Pur-sang', 'Galileo', 'Luna Park', 'Ecurie Test 2'),
('HORSE003', 'VENT D\'OUEST', 2021, 'hongre', 'Gris', 'Pur-sang', 'Sea The Stars', 'Brise Marine', 'Ecurie Test 3'),
('HORSE004', 'ROI DU SOLEIL', 2020, 'male', 'Alezan brûlé', 'Pur-sang', 'Frankel', 'Queen of Light', 'Ecurie Test 4'),
('HORSE005', 'PRINCESSE MAYA', 2019, 'female', 'Bai foncé', 'Pur-sang', 'Kingman', 'Royal Maya', 'Ecurie Test 5');

-- ============================
-- Seed ML Model (Initial dummy model)
-- ============================
INSERT INTO ml_models (version, algorithm, training_samples, features_used, hyperparameters, accuracy, precision_score, recall_score, f1_score, roc_auc, file_path, is_active) VALUES
('v0.1.0-initial', 'GradientBoostingClassifier', 0,
'["recent_form", "handicap_value", "distance", "surface_type", "jockey_win_rate", "trainer_win_rate"]',
'{"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "min_samples_split": 20}',
0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
'/app/ml_models/model_v0.1.0_initial.joblib',
FALSE);

-- ============================
-- Create indexes for performance (if not already created)
-- ============================
-- These are already in init.sql, but included here as reminder

-- ============================
-- Display seed results
-- ============================
SELECT
    'Seed data inserted successfully!' as status,
    (SELECT COUNT(*) FROM users) as users_count,
    (SELECT COUNT(*) FROM hippodromes) as hippodromes_count,
    (SELECT COUNT(*) FROM jockeys) as jockeys_count,
    (SELECT COUNT(*) FROM trainers) as trainers_count,
    (SELECT COUNT(*) FROM horses) as horses_count,
    (SELECT COUNT(*) FROM ml_models) as ml_models_count;

-- ============================
-- Test user accounts information
-- ============================
SELECT '=== TEST USER ACCOUNTS ===' as info;
SELECT
    email,
    role,
    initial_bankroll,
    'Password123!' as password_hint,
    CASE
        WHEN role = 'admin' THEN 'Full access to all features + admin panel'
        WHEN role = 'subscriber' THEN 'Full access to main features'
        WHEN role = 'guest' THEN 'Limited access (3 pronostics/day max)'
    END as access_level
FROM users
ORDER BY
    CASE role
        WHEN 'admin' THEN 1
        WHEN 'subscriber' THEN 2
        WHEN 'guest' THEN 3
    END;
