// Copyright (c) 2025 PronoTurf AI. All rights reserved.
// This source code is proprietary and confidential.
// Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden px-6 py-16">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-24 left-1/3 h-80 w-80 rounded-full bg-primary-500/30 blur-3xl" />
        <div className="absolute bottom-0 right-1/4 h-96 w-96 rounded-full bg-secondary-500/25 blur-3xl" />
      </div>

      <div className="card relative z-10 max-w-xl text-center">
        <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.3em] text-slate-300">
          Erreur 404
        </span>
        <h1 className="mt-6 text-6xl font-heading text-white">Page introuvable</h1>
        <p className="mt-4 text-base text-slate-300">
          La page que vous recherchez n'existe pas ou a été déplacée. Revenez à l’accueil pour continuer votre exploration.
        </p>
        <Link to="/" className="btn btn-primary mt-8 inline-flex px-8 py-3 text-lg">
          Retour à l'accueil
        </Link>
      </div>
    </div>
  )
}
