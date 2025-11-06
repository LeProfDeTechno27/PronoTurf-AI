// Copyright (c) 2025 PronoTurf AI. All rights reserved.
// This source code is proprietary and confidential.
// Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

import { Link } from 'react-router-dom'

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold text-gray-900">
              🏇 PronoTurf Dashboard
            </h1>
            <button className="btn btn-primary">
              Déconnexion
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Welcome Banner */}
        <div className="card mb-8 bg-gradient-to-r from-primary-500 to-primary-700 text-white">
          <h2 className="text-2xl font-bold mb-2">
            Bienvenue sur votre dashboard !
          </h2>
          <p className="text-primary-100">
            Votre bankroll actuel : <span className="font-bold text-2xl">1 000,00 €</span>
          </p>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="card">
            <p className="text-gray-600 text-sm mb-1">Paris du jour</p>
            <p className="text-3xl font-bold text-gray-900">0</p>
          </div>
          <div className="card">
            <p className="text-gray-600 text-sm mb-1">ROI ce mois</p>
            <p className="text-3xl font-bold text-secondary-600">0.00%</p>
          </div>
          <div className="card">
            <p className="text-gray-600 text-sm mb-1">Win rate</p>
            <p className="text-3xl font-bold text-primary-600">0.00%</p>
          </div>
          <div className="card">
            <p className="text-gray-600 text-sm mb-1">Évolution</p>
            <p className="text-3xl font-bold text-gray-900">0.00 €</p>
          </div>
        </div>

        {/* Sections */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="card">
            <h3 className="text-xl font-bold mb-4">Courses du jour</h3>
            <p className="text-gray-600">Aucune course disponible pour le moment.</p>
          </div>

          <div className="card">
            <h3 className="text-xl font-bold mb-4">Pronostics récents</h3>
            <p className="text-gray-600">Aucun pronostic pour le moment.</p>
          </div>
        </div>

        <div className="card">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h3 className="text-xl font-bold text-gray-900">Analytics Aspiturf</h3>
              <p className="text-gray-600">
                Accédez au laboratoire de statistiques pour explorer les performances cheval, jockey et entraîneur.
              </p>
            </div>
            <Link to="/analytics" className="btn btn-primary w-full md:w-auto">
              Explorer les analytics
            </Link>
          </div>
        </div>
      </main>
    </div>
  )
}