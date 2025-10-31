import { FormEvent, useMemo, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import analyticsService from '../services/analytics'
import type {
  CourseAnalyticsResponse,
  CoupleAnalyticsResponse,
  HorseAnalyticsResponse,
  JockeyAnalyticsResponse,
  PerformanceBreakdown,
  RecentRace,
  TrainerAnalyticsResponse,
} from '../types/analytics'

const formatPercent = (value?: number | null, digits = 1) =>
  value == null ? '—' : `${(value * 100).toFixed(digits)} %`

const formatNumber = (value?: number | null) =>
  value == null ? '—' : value.toLocaleString('fr-FR')

const formatAverage = (value?: number | null, digits = 2) =>
  value == null ? '—' : value.toFixed(digits)

const formatDate = (value?: string | null) =>
  value ? new Date(value).toLocaleDateString('fr-FR') : '—'

type HorseSearch = { id: string; hippodrome?: string }
type PersonSearch = { id: string; hippodrome?: string }
type CoupleSearch = { horseId: string; jockeyId: string; hippodrome?: string }
type CourseSearch = { date: string; hippodrome: string; courseNumber: number }

const isJockeyResponse = (
  data: JockeyAnalyticsResponse | TrainerAnalyticsResponse,
): data is JockeyAnalyticsResponse => 'jockey_id' in data

function SectionCard({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <section className="card space-y-6">
      <div className="space-y-1">
        <h2 className="text-2xl font-semibold text-gray-900">{title}</h2>
        <p className="text-gray-600">{description}</p>
      </div>
      {children}
    </section>
  )
}

function BreakdownTable({ data, emptyLabel }: { data: PerformanceBreakdown[]; emptyLabel: string }) {
  if (!data.length) {
    return <p className="text-gray-500">{emptyLabel}</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Catégorie</th>
            <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Courses</th>
            <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Victoires</th>
            <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Podiums</th>
            <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Taux de victoire</th>
            <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Taux de podium</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.map((item) => (
            <tr key={item.label}>
              <td className="px-4 py-2 text-sm text-gray-900">{item.label}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(item.total)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(item.wins)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(item.podiums)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatPercent(item.win_rate)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatPercent(item.podium_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RecentResults({ races }: { races: RecentRace[] }) {
  if (!races.length) {
    return <p className="text-gray-500">Aucun historique récent disponible.</p>
  }

  return (
    <ul className="space-y-2">
      {races.map((race, index) => (
        <li key={`${race.date}-${race.course_number}-${index}`} className="flex items-center justify-between rounded-lg border border-gray-200 px-4 py-2">
          <div>
            <p className="text-sm font-medium text-gray-900">
              {race.hippodrome ?? 'Hippodrome inconnu'} • Course {race.course_number ?? '—'}
            </p>
            <p className="text-xs text-gray-500">{formatDate(race.date)} • {race.distance ? `${race.distance} m` : 'Distance inconnue'}</p>
          </div>
          <div className="text-right">
            <p className="text-sm font-semibold text-gray-900">Position finale : {race.final_position ?? '—'}</p>
            <p className="text-xs text-gray-500">Cote : {race.odds ?? '—'} • {race.is_win ? 'Victoire' : race.is_podium ? 'Podium' : 'Hors podium'}</p>
          </div>
        </li>
      ))}
    </ul>
  )
}

function SummaryStats({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-3">
      <p className="text-xs uppercase tracking-wide text-gray-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-gray-900">{value}</p>
    </div>
  )
}

function HorseAnalyticsPanel({ data }: { data: HorseAnalyticsResponse }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Nom du cheval" value={data.horse_name ?? data.horse_id} />
        <SummaryStats label="Courses analysées" value={formatNumber(data.sample_size)} />
        <SummaryStats label="Victoires" value={`${formatNumber(data.wins)} (${formatPercent(data.win_rate)})`} />
        <SummaryStats label="Podiums" value={`${formatNumber(data.podiums)} (${formatPercent(data.podium_rate)})`} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats label="Position moyenne" value={formatAverage(data.average_finish)} />
        <SummaryStats label="Cote moyenne" value={formatAverage(data.average_odds)} />
        <SummaryStats label="Plage de dates" value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-3 text-lg font-semibold text-gray-900">Résultats récents</h3>
          <RecentResults races={data.recent_results} />
        </div>
        <div>
          <h3 className="mb-3 text-lg font-semibold text-gray-900">Répartition par hippodrome</h3>
          <BreakdownTable data={data.hippodrome_breakdown} emptyLabel="Pas de répartition disponible." />
        </div>
      </div>

      <div>
        <h3 className="mb-3 text-lg font-semibold text-gray-900">Répartition par distance</h3>
        <BreakdownTable data={data.distance_breakdown} emptyLabel="Pas de répartition disponible." />
      </div>
    </div>
  )
}

function PersonAnalyticsPanel({
  data,
  label,
}: {
  data: JockeyAnalyticsResponse | TrainerAnalyticsResponse
  label: string
}) {
  const displayName = isJockeyResponse(data)
    ? data.jockey_name ?? data.jockey_id
    : data.trainer_name ?? data.trainer_id

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label={label} value={displayName} />
        <SummaryStats label="Courses analysées" value={formatNumber(data.sample_size)} />
        <SummaryStats label="Victoires" value={`${formatNumber(data.wins)} (${formatPercent(data.win_rate)})`} />
        <SummaryStats label="Podiums" value={`${formatNumber(data.podiums)} (${formatPercent(data.podium_rate)})`} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats label="Position moyenne" value={formatAverage(data.average_finish)} />
        <SummaryStats label="Plage de dates" value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`} />
        <SummaryStats label="Hippodrome filtré" value={data.metadata.hippodrome_filter ?? 'Tous'} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-3 text-lg font-semibold text-gray-900">Résultats récents</h3>
          <RecentResults races={data.recent_results} />
        </div>
        <div>
          <h3 className="mb-3 text-lg font-semibold text-gray-900">Top chevaux</h3>
          <BreakdownTable data={'horse_breakdown' in data ? data.horse_breakdown : []} emptyLabel="Pas de répartition disponible." />
        </div>
      </div>

      <div>
        <h3 className="mb-3 text-lg font-semibold text-gray-900">Répartition par hippodrome</h3>
        <BreakdownTable data={data.hippodrome_breakdown} emptyLabel="Pas de répartition disponible." />
      </div>
    </div>
  )
}

function CoupleAnalyticsPanel({ data }: { data: CoupleAnalyticsResponse }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Cheval" value={data.horse_name ?? data.horse_id} />
        <SummaryStats label="Jockey" value={data.jockey_name ?? data.jockey_id} />
        <SummaryStats label="Courses ensemble" value={formatNumber(data.sample_size)} />
        <SummaryStats label="Victoires" value={`${formatNumber(data.wins)} (${formatPercent(data.win_rate)})`} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats label="Podiums" value={`${formatNumber(data.podiums)} (${formatPercent(data.podium_rate)})`} />
        <SummaryStats label="Position moyenne" value={formatAverage(data.average_finish)} />
        <SummaryStats label="Plage de dates" value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`} />
      </div>

      <div>
        <h3 className="mb-3 text-lg font-semibold text-gray-900">Résultats récents</h3>
        <RecentResults races={data.recent_results} />
      </div>
    </div>
  )
}

function CourseAnalyticsPanel({ data }: { data: CourseAnalyticsResponse }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-6">
        <SummaryStats label="Date" value={formatDate(data.date)} />
        <SummaryStats label="Hippodrome" value={data.hippodrome} />
        <SummaryStats label="Course" value={`Course n°${data.course_number}`} />
        <SummaryStats label="Distance" value={data.distance ? `${data.distance} m` : '—'} />
        <SummaryStats label="Allocation" value={data.allocation ? `${formatNumber(data.allocation)} ${data.currency ?? '€'}` : '—'} />
        <SummaryStats label="Discipline" value={data.discipline ?? '—'} />
      </div>

      <div>
        <h3 className="mb-3 text-lg font-semibold text-gray-900">Partants ({data.partants.length})</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">N°</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Cheval</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Jockey</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Entraîneur</th>
                <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Cote</th>
                <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Forme</th>
                <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Jours depuis</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data.partants.map((partant) => (
                <tr key={`${partant.numero}-${partant.horse_id}`}> 
                  <td className="px-4 py-2 text-sm text-gray-700">{partant.numero ?? '—'}</td>
                  <td className="px-4 py-2 text-sm text-gray-900">{partant.horse_name ?? partant.horse_id ?? 'Cheval inconnu'}</td>
                  <td className="px-4 py-2 text-sm text-gray-700">{partant.jockey_name ?? partant.jockey_id ?? '—'}</td>
                  <td className="px-4 py-2 text-sm text-gray-700">{partant.trainer_name ?? partant.trainer_id ?? '—'}</td>
                  <td className="px-4 py-2 text-sm text-right text-gray-700">{partant.odds ?? partant.probable_odds ?? '—'}</td>
                  <td className="px-4 py-2 text-sm text-right text-gray-700">{partant.recent_form ?? '—'}</td>
                  <td className="px-4 py-2 text-sm text-right text-gray-700">{partant.days_since_last_race ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default function AnalyticsPage() {
  const [horseIdInput, setHorseIdInput] = useState('')
  const [horseHippoInput, setHorseHippoInput] = useState('')
  const [horseSearch, setHorseSearch] = useState<HorseSearch | null>(null)
  const [horseError, setHorseError] = useState<string | null>(null)

  const [jockeyIdInput, setJockeyIdInput] = useState('')
  const [jockeyHippoInput, setJockeyHippoInput] = useState('')
  const [jockeySearch, setJockeySearch] = useState<PersonSearch | null>(null)
  const [jockeyError, setJockeyError] = useState<string | null>(null)

  const [trainerIdInput, setTrainerIdInput] = useState('')
  const [trainerHippoInput, setTrainerHippoInput] = useState('')
  const [trainerSearch, setTrainerSearch] = useState<PersonSearch | null>(null)
  const [trainerError, setTrainerError] = useState<string | null>(null)

  const [coupleHorseInput, setCoupleHorseInput] = useState('')
  const [coupleJockeyInput, setCoupleJockeyInput] = useState('')
  const [coupleHippoInput, setCoupleHippoInput] = useState('')
  const [coupleSearch, setCoupleSearch] = useState<CoupleSearch | null>(null)
  const [coupleError, setCoupleError] = useState<string | null>(null)

  const [courseDateInput, setCourseDateInput] = useState('')
  const [courseHippoInput, setCourseHippoInput] = useState('')
  const [courseNumberInput, setCourseNumberInput] = useState('')
  const [courseSearch, setCourseSearch] = useState<CourseSearch | null>(null)
  const [courseError, setCourseError] = useState<string | null>(null)

  const horseQueryKey = useMemo(() => (
    horseSearch ? ['analytics', 'horse', horseSearch.id, horseSearch.hippodrome ?? ''] : ['analytics', 'horse', 'idle']
  ), [horseSearch])

  const horseQuery = useQuery({
    queryKey: horseQueryKey,
    queryFn: () => analyticsService.getHorseAnalytics(horseSearch!.id, horseSearch?.hippodrome),
    enabled: Boolean(horseSearch?.id),
  })

  const jockeyQueryKey = useMemo(() => (
    jockeySearch ? ['analytics', 'jockey', jockeySearch.id, jockeySearch.hippodrome ?? ''] : ['analytics', 'jockey', 'idle']
  ), [jockeySearch])

  const jockeyQuery = useQuery({
    queryKey: jockeyQueryKey,
    queryFn: () => analyticsService.getJockeyAnalytics(jockeySearch!.id, jockeySearch?.hippodrome),
    enabled: Boolean(jockeySearch?.id),
  })

  const trainerQueryKey = useMemo(() => (
    trainerSearch ? ['analytics', 'trainer', trainerSearch.id, trainerSearch.hippodrome ?? ''] : ['analytics', 'trainer', 'idle']
  ), [trainerSearch])

  const trainerQuery = useQuery({
    queryKey: trainerQueryKey,
    queryFn: () => analyticsService.getTrainerAnalytics(trainerSearch!.id, trainerSearch?.hippodrome),
    enabled: Boolean(trainerSearch?.id),
  })

  const coupleQueryKey = useMemo(() => (
    coupleSearch
      ? ['analytics', 'couple', coupleSearch.horseId, coupleSearch.jockeyId, coupleSearch.hippodrome ?? '']
      : ['analytics', 'couple', 'idle']
  ), [coupleSearch])

  const coupleQuery = useQuery({
    queryKey: coupleQueryKey,
    queryFn: () => analyticsService.getCoupleAnalytics(
      coupleSearch!.horseId,
      coupleSearch!.jockeyId,
      coupleSearch?.hippodrome,
    ),
    enabled: Boolean(coupleSearch?.horseId && coupleSearch?.jockeyId),
  })

  const courseQueryKey = useMemo(() => (
    courseSearch
      ? ['analytics', 'course', courseSearch.date, courseSearch.hippodrome, courseSearch.courseNumber]
      : ['analytics', 'course', 'idle']
  ), [courseSearch])

  const courseQuery = useQuery({
    queryKey: courseQueryKey,
    queryFn: () => analyticsService.getCourseAnalytics(
      courseSearch!.date,
      courseSearch!.hippodrome,
      courseSearch!.courseNumber,
    ),
    enabled: Boolean(courseSearch?.date && courseSearch?.hippodrome && courseSearch?.courseNumber),
  })

  const handleHorseSubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = horseIdInput.trim()
    if (!id) {
      setHorseError('Veuillez saisir un identifiant cheval (idChe).')
      setHorseSearch(null)
      return
    }

    setHorseError(null)
    setHorseSearch({ id, hippodrome: horseHippoInput.trim() || undefined })
  }

  const handleJockeySubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = jockeyIdInput.trim()
    if (!id) {
      setJockeyError('Veuillez saisir un identifiant jockey (idJockey).')
      setJockeySearch(null)
      return
    }

    setJockeyError(null)
    setJockeySearch({ id, hippodrome: jockeyHippoInput.trim() || undefined })
  }

  const handleTrainerSubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = trainerIdInput.trim()
    if (!id) {
      setTrainerError('Veuillez saisir un identifiant entraîneur (idEntraineur).')
      setTrainerSearch(null)
      return
    }

    setTrainerError(null)
    setTrainerSearch({ id, hippodrome: trainerHippoInput.trim() || undefined })
  }

  const handleCoupleSubmit = (event: FormEvent) => {
    event.preventDefault()
    const horseId = coupleHorseInput.trim()
    const jockeyId = coupleJockeyInput.trim()

    if (!horseId || !jockeyId) {
      setCoupleError('Veuillez saisir un identifiant cheval et jockey.')
      setCoupleSearch(null)
      return
    }

    setCoupleError(null)
    setCoupleSearch({ horseId, jockeyId, hippodrome: coupleHippoInput.trim() || undefined })
  }

  const handleCourseSubmit = (event: FormEvent) => {
    event.preventDefault()
    const date = courseDateInput.trim()
    const hippodrome = courseHippoInput.trim()
    const courseNumber = Number(courseNumberInput.trim())

    if (!date || !hippodrome || Number.isNaN(courseNumber) || courseNumber < 1) {
      setCourseError('Veuillez renseigner une date (YYYY-MM-DD), un hippodrome et un numéro de course valide.')
      setCourseSearch(null)
      return
    }

    setCourseError(null)
    setCourseSearch({ date, hippodrome, courseNumber })
  }

  return (
    <div className="min-h-screen bg-gray-50 py-10">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 px-4">
        <header className="space-y-2">
          <h1 className="text-4xl font-bold text-gray-900">Analytics Aspiturf</h1>
          <p className="text-gray-600">
            Explorez les statistiques avancées issues des fichiers Aspiturf pour éclairer vos décisions avant-course.
          </p>
        </header>

        <SectionCard
          title="Cheval"
          description="Obtenez la fiche complète d'un cheval à partir de son identifiant Aspiturf (idChe)."
        >
          <form onSubmit={handleHorseSubmit} className="grid gap-4 md:grid-cols-[2fr,2fr,auto]">
            <input
              value={horseIdInput}
              onChange={(event) => setHorseIdInput(event.target.value)}
              placeholder="Identifiant cheval (idChe)"
              className="input"
            />
            <input
              value={horseHippoInput}
              onChange={(event) => setHorseHippoInput(event.target.value)}
              placeholder="Filtrer par hippodrome (optionnel)"
              className="input"
            />
            <button type="submit" className="btn btn-primary">
              Analyser
            </button>
          </form>
          {horseError && <p className="text-sm text-red-600">{horseError}</p>}
          {horseQuery.isPending && <p className="text-sm text-gray-500">Chargement des statistiques cheval…</p>}
          {horseQuery.isError && (
            <p className="text-sm text-red-600">Erreur: {(horseQuery.error as Error).message}</p>
          )}
          {horseQuery.data && <HorseAnalyticsPanel data={horseQuery.data} />}
        </SectionCard>

        <SectionCard
          title="Jockey"
          description="Analysez la forme d'un jockey sur l'ensemble de ses montes enregistrées."
        >
          <form onSubmit={handleJockeySubmit} className="grid gap-4 md:grid-cols-[2fr,2fr,auto]">
            <input
              value={jockeyIdInput}
              onChange={(event) => setJockeyIdInput(event.target.value)}
              placeholder="Identifiant jockey (idJockey)"
              className="input"
            />
            <input
              value={jockeyHippoInput}
              onChange={(event) => setJockeyHippoInput(event.target.value)}
              placeholder="Filtrer par hippodrome (optionnel)"
              className="input"
            />
            <button type="submit" className="btn btn-primary">
              Analyser
            </button>
          </form>
          {jockeyError && <p className="text-sm text-red-600">{jockeyError}</p>}
          {jockeyQuery.isPending && <p className="text-sm text-gray-500">Chargement des statistiques jockey…</p>}
          {jockeyQuery.isError && (
            <p className="text-sm text-red-600">Erreur: {(jockeyQuery.error as Error).message}</p>
          )}
          {jockeyQuery.data && <PersonAnalyticsPanel data={jockeyQuery.data} label="Nom du jockey" />}
        </SectionCard>

        <SectionCard
          title="Entraîneur"
          description="Mesurez la performance récente d'un entraîneur Aspiturf."
        >
          <form onSubmit={handleTrainerSubmit} className="grid gap-4 md:grid-cols-[2fr,2fr,auto]">
            <input
              value={trainerIdInput}
              onChange={(event) => setTrainerIdInput(event.target.value)}
              placeholder="Identifiant entraîneur (idEntraineur)"
              className="input"
            />
            <input
              value={trainerHippoInput}
              onChange={(event) => setTrainerHippoInput(event.target.value)}
              placeholder="Filtrer par hippodrome (optionnel)"
              className="input"
            />
            <button type="submit" className="btn btn-primary">
              Analyser
            </button>
          </form>
          {trainerError && <p className="text-sm text-red-600">{trainerError}</p>}
          {trainerQuery.isPending && <p className="text-sm text-gray-500">Chargement des statistiques entraîneur…</p>}
          {trainerQuery.isError && (
            <p className="text-sm text-red-600">Erreur: {(trainerQuery.error as Error).message}</p>
          )}
          {trainerQuery.data && <PersonAnalyticsPanel data={trainerQuery.data} label="Nom de l'entraîneur" />}
        </SectionCard>

        <SectionCard
          title="Couple cheval / jockey"
          description="Visualisez l'alchimie d'un couple spécifique, avec historique commun."
        >
          <form onSubmit={handleCoupleSubmit} className="grid gap-4 md:grid-cols-[1.5fr,1.5fr,1.5fr,auto]">
            <input
              value={coupleHorseInput}
              onChange={(event) => setCoupleHorseInput(event.target.value)}
              placeholder="Identifiant cheval (idChe)"
              className="input"
            />
            <input
              value={coupleJockeyInput}
              onChange={(event) => setCoupleJockeyInput(event.target.value)}
              placeholder="Identifiant jockey (idJockey)"
              className="input"
            />
            <input
              value={coupleHippoInput}
              onChange={(event) => setCoupleHippoInput(event.target.value)}
              placeholder="Filtrer par hippodrome (optionnel)"
              className="input"
            />
            <button type="submit" className="btn btn-primary">
              Analyser
            </button>
          </form>
          {coupleError && <p className="text-sm text-red-600">{coupleError}</p>}
          {coupleQuery.isPending && <p className="text-sm text-gray-500">Chargement des statistiques du couple…</p>}
          {coupleQuery.isError && (
            <p className="text-sm text-red-600">Erreur: {(coupleQuery.error as Error).message}</p>
          )}
          {coupleQuery.data && <CoupleAnalyticsPanel data={coupleQuery.data} />}
        </SectionCard>

        <SectionCard
          title="Course Aspiturf"
          description="Accédez au tableau des partants et aux métriques pré-course d'une réunion Aspiturf."
        >
          <form onSubmit={handleCourseSubmit} className="grid gap-4 md:grid-cols-[1.5fr,1.5fr,1fr,auto]">
            <input
              value={courseDateInput}
              onChange={(event) => setCourseDateInput(event.target.value)}
              placeholder="Date (YYYY-MM-DD)"
              className="input"
            />
            <input
              value={courseHippoInput}
              onChange={(event) => setCourseHippoInput(event.target.value)}
              placeholder="Nom de l'hippodrome"
              className="input"
            />
            <input
              value={courseNumberInput}
              onChange={(event) => setCourseNumberInput(event.target.value)}
              placeholder="Numéro de course"
              className="input"
            />
            <button type="submit" className="btn btn-primary">
              Analyser
            </button>
          </form>
          {courseError && <p className="text-sm text-red-600">{courseError}</p>}
          {courseQuery.isPending && <p className="text-sm text-gray-500">Chargement des informations course…</p>}
          {courseQuery.isError && (
            <p className="text-sm text-red-600">Erreur: {(courseQuery.error as Error).message}</p>
          )}
          {courseQuery.data && <CourseAnalyticsPanel data={courseQuery.data} />}
        </SectionCard>
      </div>
    </div>
  )
}
