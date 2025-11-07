// Copyright (c) 2025 PronoTurf AI. All rights reserved.
// This source code is proprietary and confidential.
// Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

import { Link } from 'react-router-dom'

const quickStats = [
  { label: 'Paris du jour', value: '0', accent: 'text-white' },
  { label: 'ROI ce mois', value: '0.00 %', accent: 'text-emerald-300' },
  { label: 'Win rate', value: '0.00 %', accent: 'text-primary-200' },
  { label: 'Évolution', value: '0.00 €', accent: 'text-slate-200' },
]

export default function Dashboard() {
  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-40 right-0 h-96 w-96 rounded-full bg-primary-500/30 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-[28rem] w-[28rem] rounded-full bg-secondary-500/25 blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto flex min-h-screen max-w-7xl flex-col px-6 pb-16">
        <header className="flex flex-col gap-6 py-10 md:flex-row md:items-center md:justify-between">
          <div className="space-y-2">
            <p className="badge">Espace membre</p>
            <h1 className="text-4xl font-heading text-white">Tableau de bord PronoTurf</h1>
            <p className="text-slate-300">
              Visualisez vos performances, accédez aux analytics et pilotez vos stratégies de mise en toute confiance.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link to="/analytics" className="btn btn-ghost px-4">
              Analytics avancés
            </Link>
            <button className="btn btn-secondary px-4">Déconnexion</button>
          </div>
        </header>

        <main className="flex flex-1 flex-col gap-10">
          <section className="surface bg-gradient-to-r from-primary-500/30 via-secondary-500/20 to-transparent p-8">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-sm uppercase tracking-[0.3em] text-slate-300">Votre bankroll</p>
                <p className="mt-3 text-4xl font-semibold text-white">1 000,00 €</p>
                <p className="mt-2 max-w-xl text-sm text-slate-300">
                  Connectez vos opérateurs favoris pour synchroniser vos mises et suivre l’évolution de votre capital en temps réel.
                </p>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
                  <p className="text-sm text-slate-300">Stratégie actuelle</p>
                  <p className="mt-2 text-lg font-semibold text-white">Kelly adaptatif</p>
                  <p className="mt-2 text-xs text-slate-400">Pilotage automatique selon votre profil de risque.</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
                  <p className="text-sm text-slate-300">Variance observée</p>
                  <p className="mt-2 text-lg font-semibold text-white">Faible</p>
                  <p className="mt-2 text-xs text-slate-400">Stabilité sur les 15 derniers jours.</p>
                </div>
              </div>
            </div>
          </section>

          <section className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
            {quickStats.map((stat) => (
              <div key={stat.label} className="card">
                <p className="text-sm text-slate-300">{stat.label}</p>
                <p className={`mt-3 text-3xl font-semibold ${stat.accent}`}>{stat.value}</p>
              </div>
            ))}
          </section>

          <section className="grid gap-6 lg:grid-cols-2">
            <div className="card">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-heading text-white">Courses du jour</h2>
                <span className="badge">Live bientôt</span>
              </div>
              <p className="mt-4 text-sm text-slate-300">Aucune course disponible pour le moment.</p>
              <div className="mt-6 rounded-2xl border border-dashed border-white/20 bg-white/5 p-6 text-sm text-slate-400">
                Connectez une source de données pour voir apparaître les courses ici.
              </div>
            </div>

            <div className="card">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-heading text-white">Pronostics récents</h2>
                <span className="badge">Historique</span>
              </div>
              <p className="mt-4 text-sm text-slate-300">Aucun pronostic pour le moment.</p>
              <div className="mt-6 rounded-2xl border border-dashed border-white/20 bg-white/5 p-6 text-sm text-slate-400">
                Une fois les modèles synchronisés, vos tickets et explications SHAP apparaîtront ici.
              </div>
            </div>
          </section>

          <section className="card">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <h2 className="text-2xl font-heading text-white">Analytics Aspiturf</h2>
                <p className="mt-2 text-sm text-slate-300">
                  Accédez au laboratoire de statistiques pour explorer les performances cheval, jockey et entraîneur.
                </p>
              </div>
              <Link to="/analytics" className="btn btn-primary w-full md:w-auto">
                Explorer les analytics
              </Link>
            </div>
          </section>
        </main>
      </div>
    </div>
  )
}
