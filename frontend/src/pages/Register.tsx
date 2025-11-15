// Copyright (c) 2025 PronoTurf AI. All rights reserved.
// This source code is proprietary and confidential.
// Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

import { ChangeEvent, FormEvent, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { isAxiosError } from 'axios'

import { registerUser } from '../services/auth'

type RegisterFormState = {
  firstName: string
  lastName: string
  email: string
  password: string
  confirmPassword: string
  acceptTerms: boolean
}

const initialState: RegisterFormState = {
  firstName: '',
  lastName: '',
  email: '',
  password: '',
  confirmPassword: '',
  acceptTerms: false,
}

export default function Register() {
  const navigate = useNavigate()
  const [formState, setFormState] = useState<RegisterFormState>(initialState)
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
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

    if (formState.password !== formState.confirmPassword) {
      setStatusMessage({ type: 'error', text: 'Les mots de passe ne correspondent pas.' })
      return
    }

    if (!formState.acceptTerms) {
      setStatusMessage({
        type: 'error',
        text: 'Vous devez accepter les conditions d’utilisation pour continuer.',
      })
      return
    }

    setIsSubmitting(true)
    try {
      await registerUser({
        email: formState.email.trim(),
        password: formState.password,
        first_name: formState.firstName.trim() || undefined,
        last_name: formState.lastName.trim() || undefined,
      })

      setStatusMessage({
        type: 'success',
        text: 'Votre compte a été créé avec succès. Vous allez être redirigé vers la connexion.',
      })

      setTimeout(() => {
        navigate('/login')
      }, 1600)
    } catch (error) {
      let message = "Impossible de finaliser l'inscription. Veuillez réessayer."

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
        <div className="absolute -top-36 right-1/4 h-[26rem] w-[26rem] rounded-full bg-secondary-500/30 blur-3xl" />
        <div className="absolute bottom-0 left-0 h-96 w-96 rounded-full bg-primary-500/20 blur-3xl" />
      </div>

      <div className="relative z-10 grid w-full max-w-5xl gap-12 lg:grid-cols-[1fr_1.1fr]">
        <div className="surface hidden flex-col justify-between p-10 lg:flex">
          <div className="space-y-6">
            <p className="badge">Rejoindre la communauté</p>
            <h1 className="text-4xl font-heading leading-tight text-white">
              Construisez votre avantage data-driven
            </h1>
            <p className="text-slate-300">
              Accédez aux analyses premium, configurez vos alertes personnalisées et laissez l’IA vous suggérer la mise idéale.
            </p>
          </div>
          <div className="grid gap-3">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
              <p className="text-sm text-slate-300">Nouveaux signaux</p>
              <p className="mt-2 text-3xl font-semibold text-white">+18 / mois</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
              <p className="text-sm text-slate-300">Taux de satisfaction</p>
              <p className="mt-2 text-3xl font-semibold text-white">94 %</p>
            </div>
          </div>
        </div>

        <div className="card backdrop-blur-xl">
          <div className="mb-8 text-center">
            <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.3em] text-slate-300">
              Commencez gratuitement
            </span>
            <h2 className="mt-6 text-3xl font-heading text-white">Créer un compte</h2>
            <p className="mt-2 text-sm text-slate-300">
              Activez votre espace personnel et synchronisez vos préférences.
            </p>
          </div>
          <form className="space-y-6" onSubmit={handleSubmit} noValidate>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="text-left">
                <label htmlFor="first-name" className="mb-2 block text-sm font-semibold text-slate-200">
                  Prénom
                </label>
                <input
                  id="first-name"
                  name="firstName"
                  type="text"
                  autoComplete="given-name"
                  className="input"
                  placeholder="Jean"
                  value={formState.firstName}
                  onChange={handleChange}
                />
              </div>
              <div className="text-left">
                <label htmlFor="last-name" className="mb-2 block text-sm font-semibold text-slate-200">
                  Nom
                </label>
                <input
                  id="last-name"
                  name="lastName"
                  type="text"
                  autoComplete="family-name"
                  className="input"
                  placeholder="Dupont"
                  value={formState.lastName}
                  onChange={handleChange}
                />
              </div>
              <div className="sm:col-span-2 text-left">
                <label htmlFor="email" className="mb-2 block text-sm font-semibold text-slate-200">
                  Email
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="input"
                  placeholder="jean.dupont@example.com"
                  value={formState.email}
                  onChange={handleChange}
                />
              </div>
              <div className="sm:col-span-2 grid gap-4 sm:grid-cols-2">
                <div className="text-left">
                  <label htmlFor="password" className="mb-2 block text-sm font-semibold text-slate-200">
                    Mot de passe
                  </label>
                  <input
                    id="password"
                    name="password"
                    type="password"
                    autoComplete="new-password"
                    required
                    className="input"
                    placeholder="••••••••"
                    value={formState.password}
                    onChange={handleChange}
                  />
                </div>
                <div className="text-left">
                  <label htmlFor="password-confirm" className="mb-2 block text-sm font-semibold text-slate-200">
                    Confirmer le mot de passe
                  </label>
                  <input
                    id="password-confirm"
                    name="confirmPassword"
                    type="password"
                    autoComplete="new-password"
                    required
                    className="input"
                    placeholder="••••••••"
                    value={formState.confirmPassword}
                    onChange={handleChange}
                  />
                </div>
              </div>
            </div>

            <label className="flex items-start gap-3 text-sm text-slate-300">
              <input
                id="terms"
                name="acceptTerms"
                type="checkbox"
                required
                className="mt-1 h-4 w-4 rounded border-white/20 bg-white/5 text-primary-500 focus:ring-primary-400"
                checked={formState.acceptTerms}
                onChange={handleChange}
              />
              <span>
                J’accepte les{' '}
                <a href="#" className="font-semibold text-primary-200 hover:text-primary-100">
                  Conditions d’utilisation
                </a>
              </span>
            </label>

            {statusMessage && (
              <p
                className={`rounded-xl border px-4 py-3 text-sm ${
                  statusMessage.type === 'error'
                    ? 'border-red-500/40 bg-red-500/10 text-red-100'
                    : 'border-emerald-400/40 bg-emerald-500/10 text-emerald-100'
                }`}
                role="status"
                aria-live="polite"
              >
                {statusMessage.text}
              </p>
            )}

            <button
              type="submit"
              className="btn btn-primary w-full py-3 text-lg disabled:cursor-not-allowed disabled:opacity-70"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Création du compte…' : "S'inscrire"}
            </button>
          </form>

          <div className="mt-8 text-center text-sm text-slate-300">
            Vous avez déjà un compte ?{' '}
            <Link to="/login" className="font-semibold text-primary-200 hover:text-primary-100">
              Se connecter
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
