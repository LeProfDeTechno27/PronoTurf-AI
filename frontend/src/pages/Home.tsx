// Copyright (c) 2025 PronoTurf AI. All rights reserved.
// This source code is proprietary and confidential.
// Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

import { Link } from 'react-router-dom'

const featureHighlights = [
  {
    title: 'Prédictions IA temps réel',
    description: 'Pipeline de machine learning supervisé, calibré sur des milliers de courses et mis à jour quotidiennement.',
    icon: '🤖',
  },
  {
    title: 'Pilotage de bankroll',
    description: 'Stratégies Kelly, Flat Betting ou Martingale pour sécuriser et maximiser votre capital.',
    icon: '📊',
  },
  {
    title: 'Transparence totale',
    description: 'Explicabilité SHAP et fiches détaillées pour comprendre chaque recommandation.',
    icon: '🔍',
  },
]

const valuePropositions = [
  {
    title: 'Pronostics IA',
    description: 'Les modèles Gradient Boosting sélectionnent les chevaux à plus fort potentiel en intégrant météo, historique et cote.',
  },
  {
    title: 'Gestion de bankroll',
    description: 'Simulez vos scénarios, définissez votre risque et laissez l’IA proposer la mise optimale pour chaque pari.',
  },
  {
    title: 'Mode entraînement',
    description: 'Analysez vos performances passées, identifiez les biais et construisez une stratégie gagnante sur la durée.',
  },
]

const roadmap = [
  {
    title: 'Analyse pré-course',
    description: 'Tableaux de bord dynamiques, scoring de forme et détection d’outsiders à forte valeur.',
  },
  {
    title: 'Suivi live & alertes',
    description: 'Notifications sur les variations de cotes, forfaits et conditions de piste pour ajuster vos paris.',
  },
  {
    title: 'Intégrations avancées',
    description: 'Synchronisation WebSocket en temps réel et export API pour vos propres modèles.',
  },
]

export default function Home() {
  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-48 -right-32 h-96 w-96 rounded-full bg-primary-500/40 blur-3xl" />
        <div className="absolute bottom-0 left-1/2 h-[28rem] w-[28rem] -translate-x-1/2 rounded-full bg-secondary-500/30 blur-3xl" />
        <div className="absolute -bottom-40 -left-32 h-96 w-96 rounded-full bg-accent-500/20 blur-3xl" />
      </div>

      <div className="relative z-10">
        <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-8">
          <span className="flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.35em] text-slate-300">
            <span className="h-2 w-2 rounded-full bg-accent" />
            PronoTurf AI
          </span>
          <nav className="hidden items-center gap-3 text-sm text-slate-300 md:flex">
            <Link to="/login" className="btn btn-ghost px-4 py-2 text-sm">
              Se connecter
            </Link>
            <Link to="/register" className="btn btn-secondary px-4 py-2 text-sm">
              Créer un compte
            </Link>
          </nav>
        </header>

        <main className="mx-auto flex max-w-6xl flex-col gap-20 px-6 pb-24">
          <section className="grid gap-12 lg:grid-cols-[1.25fr_1fr] lg:items-center">
            <div className="flex flex-col gap-8">
              <div className="inline-flex w-fit items-center gap-2 rounded-full border border-white/10 bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.35em] text-slate-200">
                Intelligence artificielle hippique
              </div>
              <div className="space-y-6">
                <h1 className="text-4xl font-heading leading-tight text-white md:text-5xl lg:text-6xl">
                  L’IA qui accélère vos décisions sur le turf
                </h1>
                <p className="max-w-xl text-lg text-slate-300 md:text-xl">
                  Explorez les facteurs de performance, optimisez votre bankroll et prenez une longueur d’avance sur les paris hippiques grâce à une plateforme pensée pour les turfistes ambitieux.
                </p>
              </div>
              <div className="flex flex-col gap-4 sm:flex-row">
                <Link to="/register" className="btn btn-primary w-full sm:w-auto">
                  Commencer gratuitement
                </Link>
                <Link to="/login" className="btn btn-ghost w-full sm:w-auto">
                  Déjà membre ? Se connecter
                </Link>
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                {featureHighlights.map((feature) => (
                  <div key={feature.title} className="surface h-full px-5 py-6">
                    <div className="text-3xl">{feature.icon}</div>
                    <h3 className="mt-4 text-lg font-semibold text-white">{feature.title}</h3>
                    <p className="mt-2 text-sm text-slate-300">{feature.description}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="surface relative overflow-hidden px-8 py-10">
              <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-white/5 to-transparent" />
              <div className="relative space-y-8">
                <div>
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Score PronoTurf</p>
                  <p className="mt-2 text-5xl font-bold text-white">87.4</p>
                  <p className="mt-2 text-sm text-slate-300">
                    Indice de confiance consolidé sur les 30 derniers jours, basé sur plus de 5 000 courses.
                  </p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-400">ROI simulé</span>
                    <span className="text-sm text-emerald-300">+12,4 %</span>
                  </div>
                  <div className="mt-3 h-24 rounded-xl bg-gradient-to-tr from-primary-500/40 via-secondary-500/30 to-accent-500/20" />
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                  <p className="text-sm text-slate-400">Stratégie active</p>
                  <p className="mt-1 text-lg font-semibold text-white">Kelly adaptatif</p>
                  <p className="mt-2 text-sm text-slate-300">
                    Ajuste automatiquement la mise recommandée selon la variance et votre tolérance au risque.
                  </p>
                </div>
              </div>
            </div>
          </section>

          <section className="space-y-10">
            <div className="flex flex-col items-start gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="badge">Pourquoi PronoTurf ?</p>
                <h2 className="mt-4 text-3xl font-heading text-white md:text-4xl">
                  Trois piliers pour booster vos pronostics
                </h2>
              </div>
              <p className="max-w-xl text-base text-slate-300">
                Une combinaison unique d’algorithmes, de signaux explicables et d’outils de pilotage pour sécuriser chaque prise de décision.
              </p>
            </div>
            <div className="grid gap-6 md:grid-cols-3">
              {valuePropositions.map((item) => (
                <div key={item.title} className="card h-full">
                  <h3 className="text-xl font-semibold text-white">{item.title}</h3>
                  <p className="mt-3 text-sm leading-relaxed text-slate-200">{item.description}</p>
                </div>
              ))}
            </div>
          </section>

          <section className="grid gap-10 lg:grid-cols-[1fr_1.2fr] lg:items-center">
            <div className="card space-y-6">
              <p className="badge">Plan d’amélioration continue</p>
              <h2 className="text-3xl font-heading text-white md:text-4xl">
                Toujours plus de précision et de réactivité
              </h2>
              <p className="text-base text-slate-200">
                Nos prochaines itérations mettent l’accent sur la donnée en direct, l’automatisation des alertes et l’ouverture des API pour créer vos propres analyses.
              </p>
              <div className="flex flex-wrap gap-3">
                <span className="badge bg-primary/20 text-primary-100">Live odds tracking</span>
                <span className="badge bg-secondary/20 text-secondary-100">API premium</span>
                <span className="badge bg-accent/20 text-emerald-200">Backtesting avancé</span>
              </div>
            </div>
            <div className="grid gap-6">
              {roadmap.map((item, index) => (
                <div key={item.title} className="surface flex items-start gap-5 p-6">
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-white/15 bg-white/5 text-lg font-semibold text-primary-200">
                    {index + 1}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">{item.title}</h3>
                    <p className="mt-2 text-sm text-slate-300">{item.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </main>

        <footer className="border-t border-white/10 bg-white/5">
          <div className="mx-auto flex max-w-6xl flex-col gap-2 px-6 py-8 text-sm text-slate-400 md:flex-row md:items-center md:justify-between">
            <p>© {new Date().getFullYear()} PronoTurf. Plateforme éducative et analytique.</p>
            <p>Construite avec amour pour les turfistes qui misent sur la data.</p>
          </div>
        </footer>
      </div>
    </div>
  )
}
