-- Copyright (c) 2025 PronoTurf AI. All rights reserved.
-- This source code is proprietary and confidential.
-- Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

-- PronoTurf Database Initialization Script
-- MySQL 8.0+
-- This script creates all tables for the PronoTurf application

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ============================
-- Table: users
-- ============================
CREATE TABLE IF NOT EXISTS users (
    user_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role ENUM('admin', 'subscriber', 'guest') DEFAULT 'guest',
    telegram_id VARCHAR(100) UNIQUE,
    telegram_notifications_enabled BOOLEAN DEFAULT FALSE,
    telegram_linked_at TIMESTAMP NULL,
    email_notifications_enabled BOOLEAN DEFAULT TRUE,
    profile_picture_url VARCHAR(500),
    initial_bankroll DECIMAL(10, 2) DEFAULT 1000.00,
    current_bankroll DECIMAL(10, 2) DEFAULT 1000.00,
    preferred_strategy ENUM('kelly', 'flat', 'martingale') DEFAULT 'flat',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_email (email),
    INDEX idx_telegram_id (telegram_id),
    INDEX idx_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================
-- Table: user_preferences
-- ============================
CREATE TABLE IF NOT EXISTS user_preferences (
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

-- ============================
-- Table: hippodromes
-- ============================
CREATE TABLE IF NOT EXISTS hippodromes (
    hippodrome_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    code VARCHAR(50) UNIQUE NOT NULL,
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

-- ============================
-- Table: reunions
-- ============================
CREATE TABLE IF NOT EXISTS reunions (
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

-- ============================
-- Table: courses
-- ============================
CREATE TABLE IF NOT EXISTS courses (
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

-- ============================
-- Table: horses
-- ============================
CREATE TABLE IF NOT EXISTS horses (
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

-- ============================
-- Table: jockeys
-- ============================
CREATE TABLE IF NOT EXISTS jockeys (
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

-- ============================
-- Table: trainers
-- ============================
CREATE TABLE IF NOT EXISTS trainers (
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

-- ============================
-- Table: partants
-- ============================
CREATE TABLE IF NOT EXISTS partants (
    partant_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    course_id INT UNSIGNED NOT NULL,
    horse_id INT UNSIGNED NOT NULL,
    jockey_id INT UNSIGNED,
    trainer_id INT UNSIGNED NOT NULL,
    numero_corde TINYINT UNSIGNED NOT NULL,
    poids_porte DECIMAL(4, 1),
    handicap_value INT,
    equipment JSON,
    aspiturf_stats JSON,
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

-- ============================
-- Table: performances_historiques
-- ============================
CREATE TABLE IF NOT EXISTS performances_historiques (
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

-- ============================
-- Table: pronostics
-- ============================
CREATE TABLE IF NOT EXISTS pronostics (
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

-- ============================
-- Table: partant_predictions
-- ============================
CREATE TABLE IF NOT EXISTS partant_predictions (
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

-- ============================
-- Table: paris_simules
-- ============================
CREATE TABLE IF NOT EXISTS paris_simules (
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

-- ============================
-- Table: bankroll_history
-- ============================
CREATE TABLE IF NOT EXISTS bankroll_history (
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

-- ============================
-- Table: favoris
-- ============================
CREATE TABLE IF NOT EXISTS favoris (
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

-- ============================
-- Table: notifications
-- ============================
CREATE TABLE IF NOT EXISTS notifications (
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

-- ============================
-- Table: ml_models
-- ============================
CREATE TABLE IF NOT EXISTS ml_models (
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

-- ============================
-- Table: training_logs
-- ============================
CREATE TABLE IF NOT EXISTS training_logs (
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

SET FOREIGN_KEY_CHECKS = 1;

-- Database initialization completed successfully
SELECT 'PronoTurf database schema created successfully!' as status;