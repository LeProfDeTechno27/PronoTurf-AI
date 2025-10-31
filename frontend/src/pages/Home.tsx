import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-primary-600 to-primary-900 text-white">
      <div className="max-w-4xl mx-auto px-6 py-12 text-center">
        <h1 className="text-6xl font-bold mb-6">
          🏇 PronoTurf
        </h1>
        <p className="text-2xl mb-4 text-primary-100">
          L'intelligence artificielle au service de vos paris hippiques
        </p>
        <p className="text-lg mb-12 text-primary-200">
          Pronostics intelligents • Explicabilité SHAP • Gestion de bankroll optimisée
        </p>

        <div className="flex gap-4 justify-center">
          <Link
            to="/register"
            className="btn btn-secondary text-lg px-8 py-4"
          >
            Commencer gratuitement
          </Link>
          <Link
            to="/login"
            className="btn bg-white text-primary-700 hover:bg-gray-100 text-lg px-8 py-4"
          >
            Se connecter
          </Link>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
            <div className="text-4xl mb-3">🤖</div>
            <h3 className="text-xl font-semibold mb-2">IA Avancée</h3>
            <p className="text-primary-100">
              Modèle Gradient Boosting entraîné sur des milliers de courses
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
            <div className="text-4xl mb-3">💡</div>
            <h3 className="text-xl font-semibold mb-2">Explicabilité SHAP</h3>
            <p className="text-primary-100">
              Comprenez les facteurs clés derrière chaque pronostic
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
            <div className="text-4xl mb-3">💰</div>
            <h3 className="text-xl font-semibold mb-2">Bankroll Optimisée</h3>
            <p className="text-primary-100">
              Kelly, Flat Betting, Martingale : choisissez votre stratégie
            </p>
          </div>
        </div>
      </div>

      <footer className="mt-auto py-6 text-center text-primary-200">
        <p>PronoTurf v0.1.0 - À but éducatif uniquement</p>
      </footer>
    </div>
  )
}
