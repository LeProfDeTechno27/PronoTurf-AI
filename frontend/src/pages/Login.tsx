// Copyright (c) 2025 PronoTurf AI. All rights reserved.
// This source code is proprietary and confidential.
// Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

import { ChangeEvent, FormEvent, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { isAxiosError } from 'axios'

import { loginUser } from '../services/auth'

type LoginFormState = {
  email: string
  password: string
  rememberMe: boolean
}

const initialState: LoginFormState = {
  email: '',
  password: '',
  rememberMe: false,
}

export default function Login() {
  const navigate = useNavigate()
  const [formState, setFormState] = useState<LoginFormState>(initialState)
  const [statusMessage, setStatusMessage] = useState<
    { type: 'success' | 'error'; text: string } | null
  >(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = event.target
    setFormState((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }))
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setStatusMessage(null)
    setIsSubmitting(true)

    try {
      const response = await loginUser({
        email: formState.email.trim(),
        password: formState.password,
      })

      const storage = formState.rememberMe ? window.localStorage : window.sessionStorage
      storage.setItem('access_token', response.access_token)
      storage.setItem('refresh_token', response.refresh_token)
      storage.setItem('token_type', response.token_type)

      setStatusMessage({ type: 'success', text: 'Connexion réussie. Redirection en cours…' })

      setTimeout(() => {
        navigate('/dashboard')
      }, 800)
    } catch (error) {
      let message = 'Impossible de vous connecter. Veuillez vérifier vos identifiants.'

      if (isAxiosError(error)) {
        const detail = error.response?.data?.detail
        if (typeof detail === 'string' && detail.trim().length > 0) {
          message = detail
        }
      }

      setStatusMessage({ type: 'error', text: message })
    } finally {
      setIsSubmitting(false)
    }
  }

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
          <form className="space-y-6" onSubmit={handleSubmit} noValidate>
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
                  value={formState.email}
                  onChange={handleChange}
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
                  value={formState.password}
                  onChange={handleChange}
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
                  checked={formState.rememberMe}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      rememberMe: event.target.checked,
                    }))
                  }
                />
                Se souvenir de moi
              </label>
              <a href="#" className="text-primary-200 transition hover:text-primary-100">
                Mot de passe oublié ?
              </a>
            </div>

            {statusMessage && (
              <div
                className={`rounded-xl border px-4 py-3 text-sm ${
                  statusMessage.type === 'success'
                    ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-100'
                    : 'border-rose-500/40 bg-rose-500/10 text-rose-100'
                }`}
              >
                {statusMessage.text}
              </div>
            )}

            <button type="submit" className="btn btn-primary w-full py-3 text-lg" disabled={isSubmitting}>
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
