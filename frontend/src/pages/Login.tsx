// Copyright (c) 2025 PronoTurf AI. All rights reserved.
// This source code is proprietary and confidential.
// Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

import { Link } from 'react-router-dom'

export default function Login() {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden px-6 py-16">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-32 left-1/2 h-96 w-96 -translate-x-1/2 rounded-full bg-primary-500/30 blur-3xl" />
        <div className="absolute bottom-0 right-0 h-[22rem] w-[22rem] rounded-full bg-secondary-500/20 blur-3xl" />
      </div>

      <div className="relative z-10 grid w-full max-w-5xl gap-12 lg:grid-cols-[1.1fr_1fr]">
        <div className="surface hidden flex-col justify-between p-10 lg:flex">
          <div className="space-y-6">
            <p className="badge">Plateforme PronoTurf</p>
            <h1 className="text-4xl font-heading leading-tight text-white">
              L’analytique IA qui comprend vos paris
            </h1>
            <p className="text-slate-300">
              Accédez aux recommandations calibrées, à l’explicabilité SHAP et à un suivi de bankroll intelligent.
            </p>
          </div>
          <div className="space-y-4">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
              <p className="text-sm text-slate-300">Historique 30 derniers jours</p>
              <p className="mt-2 text-3xl font-semibold text-white">+12,4 % ROI</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
              <p className="text-sm text-slate-300">Alertes générées</p>
              <p className="mt-2 text-3xl font-semibold text-white">58</p>
            </div>
          </div>
        </div>

        <div className="card backdrop-blur-xl">
          <div className="mb-8 text-center">
            <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.3em] text-slate-300">
              PronoTurf
            </span>
            <h2 className="mt-6 text-3xl font-heading text-white">Connectez-vous</h2>
            <p className="mt-2 text-sm text-slate-300">Ravi de vous revoir sur la plateforme.</p>
          </div>
          <form className="space-y-6" action="#" method="POST">
            <div className="space-y-4">
              <div className="text-left">
                <label htmlFor="email-address" className="mb-2 block text-sm font-semibold text-slate-200">
                  Email
                </label>
                <input
                  id="email-address"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="input"
                  placeholder="vous@exemple.com"
                />
              </div>
              <div className="text-left">
                <label htmlFor="password" className="mb-2 block text-sm font-semibold text-slate-200">
                  Mot de passe
                </label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  className="input"
                  placeholder="••••••••"
                />
              </div>
            </div>

            <div className="flex items-center justify-between text-sm text-slate-300">
              <label className="flex items-center gap-2">
                <input
                  id="remember-me"
                  name="remember-me"
                  type="checkbox"
                  className="h-4 w-4 rounded border-white/20 bg-white/5 text-primary-500 focus:ring-primary-400"
                />
                Se souvenir de moi
              </label>
              <a href="#" className="text-primary-200 transition hover:text-primary-100">
                Mot de passe oublié ?
              </a>
            </div>

            <button type="submit" className="btn btn-primary w-full py-3 text-lg">
              Se connecter
            </button>
          </form>

          <div className="mt-8 text-center text-sm text-slate-300">
            Pas encore de compte ?{' '}
            <Link to="/register" className="font-semibold text-primary-200 hover:text-primary-100">
              Rejoindre PronoTurf
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
