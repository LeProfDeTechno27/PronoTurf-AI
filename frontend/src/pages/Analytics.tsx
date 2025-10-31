import { FormEvent, useMemo, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import analyticsService from '../services/analytics'
import type {
  AnalyticsSearchResult,
  CourseAnalyticsResponse,
  CoupleAnalyticsResponse,
  HorseAnalyticsResponse,
  JockeyAnalyticsResponse,
  AnalyticsInsightsResponse,
  AnalyticsStreakResponse,
  AnalyticsFormResponse,
  LeaderboardEntry,
  PerformanceBreakdown,
  RecentRace,
  FormRace,
  TrainerAnalyticsResponse,
  PerformanceTrendResponse,
  PerformanceTrendPoint,
  PerformanceStreak,
  PerformanceDistributionResponse,
  DistributionBucket,
  DistributionDimension,
  TrendEntityType,
  TrendGranularity,
  AnalyticsComparisonResponse,
  ComparisonEntitySummary,
  HeadToHeadBreakdown,
} from '../types/analytics'

const formatPercent = (value?: number | null, digits = 1) =>
  value == null ? '—' : `${(value * 100).toFixed(digits)} %`

const formatNumber = (value?: number | null) =>
  value == null ? '—' : value.toLocaleString('fr-FR')

const formatAverage = (value?: number | null, digits = 2) =>
  value == null ? '—' : value.toFixed(digits)

const formatScore = (value?: number | null, digits = 1) =>
  value == null ? '—' : value.toFixed(digits)

const formatDate = (value?: string | null) =>
  value ? new Date(value).toLocaleDateString('fr-FR') : '—'

const formatList = (values?: string[] | null, max = 3) =>
  values?.length ? values.slice(0, max).join(', ') : '—'

const formatStreakType = (type: PerformanceStreak['type']) =>
  type === 'win' ? 'Victoires' : 'Podiums'

const formatStreakSummary = (streak?: PerformanceStreak | null) => {
  if (!streak) {
    return '—'
  }

  const period = `${formatDate(streak.start_date)} → ${formatDate(streak.end_date)}`
  const status = streak.is_active ? 'en cours' : 'terminée'
  const unit = streak.length > 1 ? 'courses' : 'course'

  return `${streak.length} ${unit} • ${period} (${status})`
}

const toError = (value: unknown): Error | null => {
  if (value instanceof Error) {
    return value
  }

  if (value) {
    return new Error('Impossible de charger les suggestions. Réessayez.')
  }

  return null
}

type HorseSearch = { id: string; hippodrome?: string }
type PersonSearch = { id: string; hippodrome?: string }
type CoupleSearch = { horseId: string; jockeyId: string; hippodrome?: string }
type CourseSearch = { date: string; hippodrome: string; courseNumber: number }
type InsightsFilters = {
  hippodrome?: string
  startDate?: string
  endDate?: string
  limit: number
}

type TrendFilters = {
  entityType: TrendEntityType
  entityId: string
  hippodrome?: string
  startDate?: string
  endDate?: string
  granularity: TrendGranularity
}

type FormFilters = {
  entityType: TrendEntityType
  entityId: string
  window: number
  hippodrome?: string
  startDate?: string
  endDate?: string
}

type StreakFilters = {
  entityType: TrendEntityType
  entityId: string
  hippodrome?: string
  startDate?: string
  endDate?: string
}

type DistributionFilters = {
  entityType: TrendEntityType
  entityId: string
  dimension: DistributionDimension
  hippodrome?: string
  startDate?: string
  endDate?: string
  distanceStep?: number
}

type ComparisonFilters = {
  entityType: TrendEntityType
  entityIds: string[]
  hippodrome?: string
  startDate?: string
  endDate?: string
}

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

function SuggestionMetadata({ result }: { result: AnalyticsSearchResult }) {
  const { metadata } = result
  const segments: string[] = []

  if (metadata.total_races != null) {
    segments.push(`${metadata.total_races} courses`)
  } else if (metadata.course_count != null) {
    segments.push(`${metadata.course_count} courses`)
  }

  if (metadata.hippodromes?.length) {
    segments.push(`Hippos: ${formatList(metadata.hippodromes)}`)
  }

  if (metadata.disciplines?.length) {
    segments.push(`Disciplines: ${formatList(metadata.disciplines)}`)
  }

  const lastDate = metadata.last_seen ?? metadata.last_meeting
  if (lastDate) {
    segments.push(`Dernière : ${formatDate(lastDate)}`)
  }

  if (!segments.length) {
    return null
  }

  return <p className="text-xs text-gray-500">{segments.join(' • ')}</p>
}

function SuggestionsList({
  results,
  isLoading,
  error,
  onSelect,
  emptyLabel,
}: {
  results: AnalyticsSearchResult[] | undefined
  isLoading: boolean
  error: Error | null
  onSelect: (result: AnalyticsSearchResult) => void
  emptyLabel: string
}) {
  if (isLoading) {
    return <p className="text-sm text-gray-500">Chargement des suggestions…</p>
  }

  if (error) {
    return <p className="text-sm text-red-600">Erreur: {error.message}</p>
  }

  if (!results?.length) {
    return <p className="text-sm text-gray-500">{emptyLabel}</p>
  }

  return (
    <ul className="space-y-2">
      {results.map((item) => (
        <li key={`${item.type}-${item.id}`}>
          <button
            type="button"
            onClick={() => onSelect(item)}
            className="flex w-full flex-col rounded-lg border border-gray-200 bg-white px-4 py-3 text-left transition hover:border-indigo-300 hover:shadow-sm"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm font-semibold text-gray-900">{item.label}</span>
              <span className="text-xs font-medium uppercase tracking-wide text-indigo-600">{item.id}</span>
            </div>
            <SuggestionMetadata result={item} />
          </button>
        </li>
      ))}
    </ul>
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

function LeaderboardTable({
  entries,
  emptyLabel,
}: {
  entries: LeaderboardEntry[]
  emptyLabel: string
}) {
  // Simplifie l'affichage des classements trans-entités pour l'explorateur analytics.
  if (!entries.length) {
    return <p className="text-gray-500">{emptyLabel}</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-gray-500">#</th>
            <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-gray-500">Entité</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Courses</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Victoires</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Podiums</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Taux V</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Taux P</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Moy. place</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Dernière course</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {entries.map((item, index) => (
            <tr key={item.entity_id}>
              <td className="px-4 py-2 text-sm text-gray-700">#{index + 1}</td>
              <td className="px-4 py-2 text-sm font-medium text-gray-900">{item.label}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(item.sample_size)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(item.wins)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(item.podiums)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatPercent(item.win_rate)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatPercent(item.podium_rate)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(item.average_finish)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatDate(item.last_seen)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TrendTable({ points }: { points: PerformanceTrendPoint[] }) {
  // Met en forme l'évolution d'une entité période par période.
  if (!points.length) {
    return <p className="text-gray-500">Aucune course sur la période sélectionnée.</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-gray-500">Période</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Courses</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Victoires</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Podiums</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Taux victoire</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Taux podium</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Arrivée moy.</th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Cote moy.</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {points.map((point) => (
            <tr key={point.label}>
              <td className="px-4 py-2 text-sm text-gray-900">{point.label}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(point.races)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(point.wins)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(point.podiums)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatPercent(point.win_rate)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatPercent(point.podium_rate)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(point.average_finish)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(point.average_odds)}</td>
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

function HeadToHeadTableView({ rows }: { rows: HeadToHeadBreakdown[] }) {
  // Rend lisible le bilan des confrontations directes entité par entité.
  if (!rows.length) {
    return <p className="text-gray-500">Pas de duel direct enregistré.</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 text-sm">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left font-medium uppercase tracking-wide text-gray-500">Adversaire</th>
            <th className="px-4 py-2 text-right font-medium uppercase tracking-wide text-gray-500">Courses</th>
            <th className="px-4 py-2 text-right font-medium uppercase tracking-wide text-gray-500">Devant</th>
            <th className="px-4 py-2 text-right font-medium uppercase tracking-wide text-gray-500">Derrière</th>
            <th className="px-4 py-2 text-right font-medium uppercase tracking-wide text-gray-500">Indécis</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {rows.map((row) => (
            <tr key={row.opponent_id}>
              <td className="px-4 py-2 text-gray-900">{row.opponent_id}</td>
              <td className="px-4 py-2 text-right text-gray-700">{formatNumber(row.meetings)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{formatNumber(row.ahead)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{formatNumber(row.behind)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{formatNumber(row.ties)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ComparisonEntityCard({ entity }: { entity: ComparisonEntitySummary }) {
  // Présente les statistiques clés d'une entité comparée.
  return (
    <div className="space-y-4 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h4 className="text-lg font-semibold text-gray-900">{entity.label ?? entity.entity_id}</h4>
          <p className="text-xs uppercase tracking-wide text-gray-500">{entity.entity_id}</p>
        </div>
        <SummaryStats label="Courses" value={formatNumber(entity.sample_size)} />
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <SummaryStats
          label="Victoires"
          value={`${formatNumber(entity.wins)} (${formatPercent(entity.win_rate)})`}
        />
        <SummaryStats
          label="Podiums"
          value={`${formatNumber(entity.podiums)} (${formatPercent(entity.podium_rate)})`}
        />
        <SummaryStats label="Moyenne arrivée" value={formatAverage(entity.average_finish)} />
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <SummaryStats label="Meilleure place" value={entity.best_finish ?? '—'} />
        <SummaryStats label="Dernière apparition" value={formatDate(entity.last_seen ?? null)} />
      </div>

      <div>
        <h5 className="mb-2 text-sm font-semibold text-gray-900">Duels directs</h5>
        <HeadToHeadTableView rows={entity.head_to_head} />
      </div>
    </div>
  )
}

function ComparisonPanel({ data }: { data: AnalyticsComparisonResponse }) {
  // Assemble les indicateurs globaux d'une comparaison multi-entités.
  const typeLabel =
    data.entity_type === 'horse'
      ? 'Chevaux'
      : data.entity_type === 'jockey'
        ? 'Jockeys'
        : 'Entraîneurs'

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Type analysé" value={typeLabel} />
        <SummaryStats label="Entités comparées" value={formatNumber(data.entities.length)} />
        <SummaryStats label="Courses communes" value={formatNumber(data.shared_races)} />
        <SummaryStats
          label="Plage de dates"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {data.entities.map((entity) => (
          <ComparisonEntityCard key={entity.entity_id} entity={entity} />
        ))}
      </div>
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

function InsightsPanel({ data }: { data: AnalyticsInsightsResponse }) {
  // Offre une vue synthétique sur les entités dominantes de la période choisie.
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats
          label="Hippodrome filtré"
          value={data.metadata.hippodrome_filter ?? 'Tous'}
        />
        <SummaryStats
          label="Période analysée"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
        <SummaryStats
          label="Total classements"
          value={`${formatNumber(data.top_horses.length)} chevaux / ${formatNumber(data.top_jockeys.length)} jockeys / ${formatNumber(data.top_trainers.length)} entraîneurs`}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-gray-900">Top Chevaux</h3>
          <LeaderboardTable entries={data.top_horses} emptyLabel="Aucun cheval identifié sur la période." />
        </div>
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-gray-900">Top Jockeys</h3>
          <LeaderboardTable entries={data.top_jockeys} emptyLabel="Aucun jockey identifié sur la période." />
        </div>
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-gray-900">Top Entraîneurs</h3>
          <LeaderboardTable entries={data.top_trainers} emptyLabel="Aucun entraîneur identifié sur la période." />
        </div>
      </div>
    </div>
  )
}

function TrendPanel({ data }: { data: PerformanceTrendResponse }) {
  // Résume la tendance agrégée avant d'afficher le détail période par période.
  const entityDisplay = data.entity_label
    ? `${data.entity_label} (${data.entity_id})`
    : data.entity_id

  const granularityLabel = data.granularity === 'week' ? 'Semaine' : 'Mois'

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats label="Entité analysée" value={entityDisplay} />
        <SummaryStats
          label="Granularité"
          value={granularityLabel}
        />
        <SummaryStats
          label="Période"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <SummaryStats
          label="Hippodrome filtré"
          value={data.metadata.hippodrome_filter ?? 'Tous'}
        />
        <SummaryStats
          label="Nombre de périodes"
          value={`${formatNumber(data.points.length)} intervalles`}
        />
      </div>

      <div>
        <h3 className="mb-3 text-lg font-semibold text-gray-900">Historique des performances</h3>
        <TrendTable points={data.points} />
      </div>
    </div>
  )
}

function StreakHistoryTable({ items }: { items: PerformanceStreak[] }) {
  // Table simplifiée listant les meilleures séries détectées.
  if (!items.length) {
    return <p className="text-gray-500">Aucune série détectée sur la période étudiée.</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
              Type
            </th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
              Longueur
            </th>
            <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
              Période
            </th>
            <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
              Statut
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {items.map((item, index) => (
            <tr key={`${item.type}-${item.start_date ?? 'unknown'}-${index}`}>
              <td className="px-4 py-2 text-sm text-gray-900">{formatStreakType(item.type)}</td>
              <td className="px-4 py-2 text-right text-sm text-gray-700">{item.length}</td>
              <td className="px-4 py-2 text-sm text-gray-700">
                {`${formatDate(item.start_date)} → ${formatDate(item.end_date)}`}
              </td>
              <td className="px-4 py-2 text-sm text-gray-700">
                {item.is_active ? 'En cours' : 'Terminée'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function StreakPanel({ data }: { data: AnalyticsStreakResponse }) {
  // Fournit une vue dédiée sur les séries consécutives (wins/podiums) d'une entité.
  const entityDisplay = data.entity_label
    ? `${data.entity_label} (${data.entity_id})`
    : data.entity_id

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Entité analysée" value={entityDisplay} />
        <SummaryStats label="Courses analysées" value={formatNumber(data.total_races)} />
        <SummaryStats label="Victoires" value={formatNumber(data.wins)} />
        <SummaryStats label="Podiums" value={formatNumber(data.podiums)} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats
          label="Plage de dates"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
        <SummaryStats
          label="Meilleure série de victoires"
          value={formatStreakSummary(data.best_win_streak)}
        />
        <SummaryStats
          label="Meilleure série de podiums"
          value={formatStreakSummary(data.best_podium_streak)}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats
          label="Série de victoires en cours"
          value={formatStreakSummary(data.current_win_streak)}
        />
        <SummaryStats
          label="Série de podiums en cours"
          value={formatStreakSummary(data.current_podium_streak)}
        />
        <SummaryStats
          label="Hippodrome filtré"
          value={data.metadata.hippodrome_filter ?? 'Tous'}
        />
      </div>

      <div>
        <h3 className="mb-3 text-lg font-semibold text-gray-900">Historique des séries</h3>
        <StreakHistoryTable items={data.streak_history} />
      </div>
    </div>
  )
}

function DistributionTable({ buckets }: { buckets: DistributionBucket[] }) {
  // Table dédiée pour visualiser la répartition des résultats par segment.
  if (!buckets.length) {
    return <p className="text-gray-500">Aucune distribution disponible pour ces filtres.</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
              Segment
            </th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
              Courses
            </th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
              Victoires
            </th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
              Podiums
            </th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
              Taux de victoire
            </th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
              Taux de podium
            </th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
              Moy. arrivée
            </th>
            <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
              Cote moyenne
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 bg-white">
          {buckets.map((bucket) => (
            <tr key={bucket.label}>
              <td className="px-4 py-2 text-sm font-medium text-gray-900">{bucket.label}</td>
              <td className="px-4 py-2 text-right text-sm text-gray-700">{formatNumber(bucket.races)}</td>
              <td className="px-4 py-2 text-right text-sm text-gray-700">{formatNumber(bucket.wins)}</td>
              <td className="px-4 py-2 text-right text-sm text-gray-700">{formatNumber(bucket.podiums)}</td>
              <td className="px-4 py-2 text-right text-sm text-gray-700">{formatPercent(bucket.win_rate)}</td>
              <td className="px-4 py-2 text-right text-sm text-gray-700">{formatPercent(bucket.podium_rate)}</td>
              <td className="px-4 py-2 text-right text-sm text-gray-700">{formatAverage(bucket.average_finish)}</td>
              <td className="px-4 py-2 text-right text-sm text-gray-700">{formatAverage(bucket.average_odds)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function FormRaceTimeline({ races }: { races: FormRace[] }) {
  // Restitue la séquence des courses utilisées pour le calcul de la forme.
  if (!races.length) {
    return <p className="text-gray-500">Aucune course récente disponible sur la période.</p>
  }

  return (
    <ul className="space-y-2">
      {races.map((race, index) => (
        <li
          key={`${race.date ?? 'inconnue'}-${race.course_number ?? index}`}
          className="flex items-center justify-between rounded-lg border border-indigo-100 bg-white px-4 py-3 shadow-sm"
        >
          <div>
            <p className="text-sm font-semibold text-gray-900">
              {race.hippodrome ?? 'Hippodrome inconnu'} • Course {race.course_number ?? '—'}
            </p>
            <p className="text-xs text-gray-500">
              {formatDate(race.date)} • {race.distance ? `${race.distance} m` : 'Distance inconnue'}
            </p>
            <p className="text-xs text-gray-500">
              {race.is_win ? 'Victoire' : race.is_podium ? 'Podium' : 'Hors podium'} • Position {race.final_position ?? '—'}
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm font-semibold text-indigo-600">Score: {race.score} / 5</p>
            <p className="text-xs text-gray-500">Cote : {race.odds ?? '—'}</p>
          </div>
        </li>
      ))}
    </ul>
  )
}

function FormPanel({ data }: { data: AnalyticsFormResponse }) {
  // Vue synthétique sur la forme récente et les indicateurs associés.
  const entityDisplay = data.entity_label ? `${data.entity_label} (${data.entity_id})` : data.entity_id

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Entité analysée" value={entityDisplay} />
        <SummaryStats label="Fenêtre analysée" value={`${data.window} courses`} />
        <SummaryStats label="Courses retenues" value={formatNumber(data.total_races)} />
        <SummaryStats label="Score moyen" value={`${formatScore(data.form_score, 2)} / 5`} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Taux de victoire" value={formatPercent(data.win_rate)} />
        <SummaryStats label="Taux de podium" value={formatPercent(data.podium_rate)} />
        <SummaryStats label="Indice de constance" value={formatPercent(data.consistency_index)} />
        <SummaryStats label="Position moyenne" value={formatAverage(data.average_finish)} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats label="Cote moyenne" value={formatAverage(data.average_odds)} />
        <SummaryStats label="Cote médiane" value={formatAverage(data.median_odds)} />
        <SummaryStats label="Meilleure place" value={formatNumber(data.best_position)} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats
          label="Plage de dates"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
        <SummaryStats
          label="Filtre hippodrome"
          value={data.metadata.hippodrome_filter ? data.metadata.hippodrome_filter : 'Tous'}
        />
        <SummaryStats label="Victoires / Podiums" value={`${formatNumber(data.wins)} • ${formatNumber(data.podiums)}`} />
      </div>

      <div className="space-y-3">
        <h3 className="text-lg font-semibold text-gray-900">Courses prises en compte</h3>
        <FormRaceTimeline races={data.races} />
      </div>
    </div>
  )
}

function DistributionPanel({ data }: { data: PerformanceDistributionResponse }) {
  // Synthèse complète de la distribution calculée côté backend.
  const entityDisplay = data.entity_label
    ? `${data.entity_label} (${data.entity_id})`
    : data.entity_id

  const dimensionLabels: Record<DistributionDimension, string> = {
    distance: 'Distance (m)',
    draw: 'Numéro de corde',
    hippodrome: 'Hippodrome',
    discipline: 'Discipline',
  }

  const totalRaces = data.buckets.reduce((sum, bucket) => sum + bucket.races, 0)
  const dominant = data.buckets[0]

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Entité analysée" value={entityDisplay} />
        <SummaryStats label="Dimension étudiée" value={dimensionLabels[data.dimension]} />
        <SummaryStats
          label="Plage de dates"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
        <SummaryStats label="Hippodrome filtré" value={data.metadata.hippodrome_filter ?? 'Tous'} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats label="Segments analysés" value={formatNumber(data.buckets.length)} />
        <SummaryStats label="Total courses" value={formatNumber(totalRaces)} />
        {dominant ? (
          <SummaryStats
            label="Segment dominant"
            value={`${dominant.label} (${formatNumber(dominant.races)} courses)`}
          />
        ) : (
          <SummaryStats label="Segment dominant" value="—" />
        )}
      </div>

      <DistributionTable buckets={data.buckets} />
    </div>
  )
}

export default function AnalyticsPage() {
  // États dédiés aux classements transverses.
  const [insightHippoInput, setInsightHippoInput] = useState('')
  const [insightStartInput, setInsightStartInput] = useState('')
  const [insightEndInput, setInsightEndInput] = useState('')
  const [insightLimitInput, setInsightLimitInput] = useState('5')
  const [insightError, setInsightError] = useState<string | null>(null)
  const [insightFilters, setInsightFilters] = useState<InsightsFilters>({ limit: 5 })

  const [trendTypeInput, setTrendTypeInput] = useState<TrendEntityType>('horse')
  const [trendIdInput, setTrendIdInput] = useState('')
  const [trendHippoInput, setTrendHippoInput] = useState('')
  const [trendStartInput, setTrendStartInput] = useState('')
  const [trendEndInput, setTrendEndInput] = useState('')
  const [trendGranularityInput, setTrendGranularityInput] = useState<TrendGranularity>('month')
  const [trendError, setTrendError] = useState<string | null>(null)
  const [trendFilters, setTrendFilters] = useState<TrendFilters | null>(null)

  const [formTypeInput, setFormTypeInput] = useState<TrendEntityType>('horse')
  const [formIdInput, setFormIdInput] = useState('')
  const [formWindowInput, setFormWindowInput] = useState('5')
  const [formHippoInput, setFormHippoInput] = useState('')
  const [formStartInput, setFormStartInput] = useState('')
  const [formEndInput, setFormEndInput] = useState('')
  const [formError, setFormError] = useState<string | null>(null)
  const [formFilters, setFormFilters] = useState<FormFilters | null>(null)

  const [streakTypeInput, setStreakTypeInput] = useState<TrendEntityType>('horse')
  const [streakIdInput, setStreakIdInput] = useState('')
  const [streakHippoInput, setStreakHippoInput] = useState('')
  const [streakStartInput, setStreakStartInput] = useState('')
  const [streakEndInput, setStreakEndInput] = useState('')
  const [streakError, setStreakError] = useState<string | null>(null)
  const [streakFilters, setStreakFilters] = useState<StreakFilters | null>(null)

  const [distributionTypeInput, setDistributionTypeInput] = useState<TrendEntityType>('horse')
  const [distributionIdInput, setDistributionIdInput] = useState('')
  const [distributionDimensionInput, setDistributionDimensionInput] =
    useState<DistributionDimension>('distance')
  const [distributionHippoInput, setDistributionHippoInput] = useState('')
  const [distributionStartInput, setDistributionStartInput] = useState('')
  const [distributionEndInput, setDistributionEndInput] = useState('')
  const [distributionStepInput, setDistributionStepInput] = useState('200')
  const [distributionError, setDistributionError] = useState<string | null>(null)
  const [distributionFilters, setDistributionFilters] = useState<DistributionFilters | null>(null)

  const [comparisonTypeInput, setComparisonTypeInput] = useState<TrendEntityType>('horse')
  const [comparisonIdInput, setComparisonIdInput] = useState('')
  const [comparisonHippoInput, setComparisonHippoInput] = useState('')
  const [comparisonStartInput, setComparisonStartInput] = useState('')
  const [comparisonEndInput, setComparisonEndInput] = useState('')
  const [comparisonQuery, setComparisonQuery] = useState('')
  const [comparisonSelections, setComparisonSelections] = useState<AnalyticsSearchResult[]>([])
  const [comparisonError, setComparisonError] = useState<string | null>(null)
  const [comparisonFilters, setComparisonFilters] = useState<ComparisonFilters | null>(null)

  const [horseIdInput, setHorseIdInput] = useState('')
  const [horseHippoInput, setHorseHippoInput] = useState('')
  const [horseSearch, setHorseSearch] = useState<HorseSearch | null>(null)
  const [horseError, setHorseError] = useState<string | null>(null)
  const [horseNameQuery, setHorseNameQuery] = useState('')

  const [jockeyIdInput, setJockeyIdInput] = useState('')
  const [jockeyHippoInput, setJockeyHippoInput] = useState('')
  const [jockeySearch, setJockeySearch] = useState<PersonSearch | null>(null)
  const [jockeyError, setJockeyError] = useState<string | null>(null)
  const [jockeyNameQuery, setJockeyNameQuery] = useState('')

  const [trainerIdInput, setTrainerIdInput] = useState('')
  const [trainerHippoInput, setTrainerHippoInput] = useState('')
  const [trainerSearch, setTrainerSearch] = useState<PersonSearch | null>(null)
  const [trainerError, setTrainerError] = useState<string | null>(null)
  const [trainerNameQuery, setTrainerNameQuery] = useState('')

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
  const [courseHippoQuery, setCourseHippoQuery] = useState('')

  const insightsQueryKey = useMemo(
    () => [
      'analytics',
      'insights',
      insightFilters.hippodrome ?? '',
      insightFilters.startDate ?? '',
      insightFilters.endDate ?? '',
      insightFilters.limit,
    ],
    [insightFilters],
  )

  const insightsQuery = useQuery({
    queryKey: insightsQueryKey,
    queryFn: () =>
      analyticsService.getInsights({
        hippodrome: insightFilters.hippodrome,
        startDate: insightFilters.startDate,
        endDate: insightFilters.endDate,
        limit: insightFilters.limit,
      }),
  })

  const trendQueryKey = useMemo(
    () =>
      trendFilters
        ? [
            'analytics',
            'trends',
            trendFilters.entityType,
            trendFilters.entityId,
            trendFilters.hippodrome ?? '',
            trendFilters.startDate ?? '',
            trendFilters.endDate ?? '',
            trendFilters.granularity,
          ]
        : ['analytics', 'trends', 'idle'],
    [trendFilters],
  )

  const trendQuery = useQuery({
    queryKey: trendQueryKey,
    queryFn: () =>
      analyticsService.getPerformanceTrend({
        entityType: trendFilters!.entityType,
        entityId: trendFilters!.entityId,
        granularity: trendFilters!.granularity,
        hippodrome: trendFilters?.hippodrome,
        startDate: trendFilters?.startDate,
        endDate: trendFilters?.endDate,
      }),
    enabled: Boolean(trendFilters?.entityId),
  })

  const formQueryKey = useMemo(
    () =>
      formFilters
        ? [
            'analytics',
            'form',
            formFilters.entityType,
            formFilters.entityId,
            formFilters.window,
            formFilters.hippodrome ?? '',
            formFilters.startDate ?? '',
            formFilters.endDate ?? '',
          ]
        : ['analytics', 'form', 'idle'],
    [formFilters],
  )

  const formQuery = useQuery({
    queryKey: formQueryKey,
    queryFn: () =>
      analyticsService.getFormSnapshot({
        entityType: formFilters!.entityType,
        entityId: formFilters!.entityId,
        window: formFilters!.window,
        hippodrome: formFilters?.hippodrome,
        startDate: formFilters?.startDate,
        endDate: formFilters?.endDate,
      }),
    enabled: Boolean(formFilters?.entityId),
  })

  const streakQueryKey = useMemo(
    () =>
      streakFilters
        ? [
            'analytics',
            'streaks',
            streakFilters.entityType,
            streakFilters.entityId,
            streakFilters.hippodrome ?? '',
            streakFilters.startDate ?? '',
            streakFilters.endDate ?? '',
          ]
        : ['analytics', 'streaks', 'idle'],
    [streakFilters],
  )

  const streakQuery = useQuery({
    queryKey: streakQueryKey,
    queryFn: () =>
      analyticsService.getPerformanceStreaks({
        entityType: streakFilters!.entityType,
        entityId: streakFilters!.entityId,
        hippodrome: streakFilters?.hippodrome,
        startDate: streakFilters?.startDate,
        endDate: streakFilters?.endDate,
      }),
    enabled: Boolean(streakFilters?.entityId),
  })

  const distributionQueryKey = useMemo(
    () =>
      distributionFilters
        ? [
            'analytics',
            'distribution',
            distributionFilters.entityType,
            distributionFilters.entityId,
            distributionFilters.dimension,
            distributionFilters.hippodrome ?? '',
            distributionFilters.startDate ?? '',
            distributionFilters.endDate ?? '',
            distributionFilters.distanceStep ?? '',
          ]
        : ['analytics', 'distribution', 'idle'],
    [distributionFilters],
  )

  const distributionQuery = useQuery({
    queryKey: distributionQueryKey,
    queryFn: () =>
      analyticsService.getPerformanceDistribution({
        entityType: distributionFilters!.entityType,
        entityId: distributionFilters!.entityId,
        dimension: distributionFilters!.dimension,
        hippodrome: distributionFilters?.hippodrome,
        startDate: distributionFilters?.startDate,
        endDate: distributionFilters?.endDate,
        distanceStep: distributionFilters?.distanceStep,
      }),
    enabled: Boolean(distributionFilters?.entityId),
  })

  const comparisonQueryKey = useMemo(
    () =>
      comparisonFilters
        ? [
            'analytics',
            'comparisons',
            comparisonFilters.entityType,
            comparisonFilters.entityIds.join(','),
            comparisonFilters.hippodrome ?? '',
            comparisonFilters.startDate ?? '',
            comparisonFilters.endDate ?? '',
          ]
        : ['analytics', 'comparisons', 'idle'],
    [comparisonFilters],
  )

  const comparisonQueryResult = useQuery({
    queryKey: comparisonQueryKey,
    queryFn: () =>
      analyticsService.getComparisons({
        entityType: comparisonFilters!.entityType,
        entityIds: comparisonFilters!.entityIds,
        hippodrome: comparisonFilters?.hippodrome,
        startDate: comparisonFilters?.startDate,
        endDate: comparisonFilters?.endDate,
      }),
    enabled: Boolean(comparisonFilters?.entityIds?.length),
  })

  const horseQueryKey = useMemo(() => (
    horseSearch ? ['analytics', 'horse', horseSearch.id, horseSearch.hippodrome ?? ''] : ['analytics', 'horse', 'idle']
  ), [horseSearch])

  const horseQuery = useQuery({
    queryKey: horseQueryKey,
    queryFn: () => analyticsService.getHorseAnalytics(horseSearch!.id, horseSearch?.hippodrome),
    enabled: Boolean(horseSearch?.id),
  })

  const horseSuggestionsQuery = useQuery<AnalyticsSearchResult[]>({
    queryKey: ['analytics', 'suggestions', 'horse', horseNameQuery],
    queryFn: () => analyticsService.searchEntities('horse', horseNameQuery),
    enabled: horseNameQuery.trim().length >= 2,
    staleTime: 60_000,
  })

  const jockeyQueryKey = useMemo(() => (
    jockeySearch ? ['analytics', 'jockey', jockeySearch.id, jockeySearch.hippodrome ?? ''] : ['analytics', 'jockey', 'idle']
  ), [jockeySearch])

  const jockeyQuery = useQuery({
    queryKey: jockeyQueryKey,
    queryFn: () => analyticsService.getJockeyAnalytics(jockeySearch!.id, jockeySearch?.hippodrome),
    enabled: Boolean(jockeySearch?.id),
  })

  const jockeySuggestionsQuery = useQuery<AnalyticsSearchResult[]>({
    queryKey: ['analytics', 'suggestions', 'jockey', jockeyNameQuery],
    queryFn: () => analyticsService.searchEntities('jockey', jockeyNameQuery),
    enabled: jockeyNameQuery.trim().length >= 2,
    staleTime: 60_000,
  })

  const comparisonSuggestionsQuery = useQuery<AnalyticsSearchResult[]>({
    queryKey: ['analytics', 'suggestions', 'comparison', comparisonTypeInput, comparisonQuery],
    queryFn: () => analyticsService.searchEntities(comparisonTypeInput, comparisonQuery),
    enabled: comparisonQuery.trim().length >= 2,
    staleTime: 60_000,
  })

  const trainerQueryKey = useMemo(() => (
    trainerSearch ? ['analytics', 'trainer', trainerSearch.id, trainerSearch.hippodrome ?? ''] : ['analytics', 'trainer', 'idle']
  ), [trainerSearch])

  const trainerQuery = useQuery({
    queryKey: trainerQueryKey,
    queryFn: () => analyticsService.getTrainerAnalytics(trainerSearch!.id, trainerSearch?.hippodrome),
    enabled: Boolean(trainerSearch?.id),
  })

  const trainerSuggestionsQuery = useQuery<AnalyticsSearchResult[]>({
    queryKey: ['analytics', 'suggestions', 'trainer', trainerNameQuery],
    queryFn: () => analyticsService.searchEntities('trainer', trainerNameQuery),
    enabled: trainerNameQuery.trim().length >= 2,
    staleTime: 60_000,
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

  const hippodromeSuggestionsQuery = useQuery<AnalyticsSearchResult[]>({
    queryKey: ['analytics', 'suggestions', 'hippodrome', courseHippoQuery],
    queryFn: () => analyticsService.searchEntities('hippodrome', courseHippoQuery),
    enabled: courseHippoQuery.trim().length >= 2,
    staleTime: 60_000,
  })

  const handleInsightsSubmit = (event: FormEvent) => {
    event.preventDefault()
    const limitValue = Number(insightLimitInput.trim() || '5')

    if (Number.isNaN(limitValue) || limitValue < 1 || limitValue > 20) {
      setInsightError('Le nombre d\'entrées doit être compris entre 1 et 20.')
      return
    }

    if (insightStartInput && insightEndInput && insightStartInput > insightEndInput) {
      setInsightError('La date de début doit précéder la date de fin.')
      return
    }

    setInsightError(null)
    const nextFilters: InsightsFilters = {
      hippodrome: insightHippoInput.trim() || undefined,
      startDate: insightStartInput || undefined,
      endDate: insightEndInput || undefined,
      limit: limitValue,
    }

    setInsightFilters(nextFilters)
  }

  const handleTrendSubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = trendIdInput.trim()

    if (!id) {
      setTrendError("Veuillez saisir un identifiant Aspiturf valide.")
      setTrendFilters(null)
      return
    }

    if (trendStartInput && trendEndInput && trendStartInput > trendEndInput) {
      setTrendError('La date de début doit précéder la date de fin.')
      setTrendFilters(null)
      return
    }

    setTrendError(null)
    setTrendFilters({
      entityType: trendTypeInput,
      entityId: id,
      hippodrome: trendHippoInput.trim() || undefined,
      startDate: trendStartInput || undefined,
      endDate: trendEndInput || undefined,
      granularity: trendGranularityInput,
    })
  }

  const handleFormSubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = formIdInput.trim()
    const windowValue = Number(formWindowInput.trim() || '5')

    if (!id) {
      setFormError("Veuillez saisir un identifiant Aspiturf valide.")
      setFormFilters(null)
      return
    }

    if (Number.isNaN(windowValue) || windowValue < 1 || windowValue > 30) {
      setFormError('La fenêtre doit être comprise entre 1 et 30 courses.')
      setFormFilters(null)
      return
    }

    if (formStartInput && formEndInput && formStartInput > formEndInput) {
      setFormError('La date de début doit précéder la date de fin.')
      setFormFilters(null)
      return
    }

    setFormError(null)
    setFormFilters({
      entityType: formTypeInput,
      entityId: id,
      window: windowValue,
      hippodrome: formHippoInput.trim() || undefined,
      startDate: formStartInput || undefined,
      endDate: formEndInput || undefined,
    })
  }

  const handleDistributionSubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = distributionIdInput.trim()

    if (!id) {
      setDistributionError("Veuillez saisir un identifiant Aspiturf valide.")
      setDistributionFilters(null)
      return
    }

    if (distributionStartInput && distributionEndInput && distributionStartInput > distributionEndInput) {
      setDistributionError('La date de début doit précéder la date de fin.')
      setDistributionFilters(null)
      return
    }

    let parsedStep: number | undefined
    if (distributionDimensionInput === 'distance') {
      parsedStep = Number(distributionStepInput.trim() || '200')
      if (Number.isNaN(parsedStep) || parsedStep < 50 || parsedStep > 1000) {
        setDistributionError('Le pas de distance doit être compris entre 50 et 1000 mètres.')
        setDistributionFilters(null)
        return
      }
    }

    setDistributionError(null)
    setDistributionFilters({
      entityType: distributionTypeInput,
      entityId: id,
      dimension: distributionDimensionInput,
      hippodrome: distributionHippoInput.trim() || undefined,
      startDate: distributionStartInput || undefined,
      endDate: distributionEndInput || undefined,
      distanceStep: parsedStep,
    })
  }

  const handleComparisonAddManual = () => {
    const trimmed = comparisonIdInput.trim()

    if (!trimmed) {
      setComparisonError("Saisissez un identifiant Aspiturf à ajouter.")
      return
    }

    if (comparisonSelections.some((item) => item.id === trimmed)) {
      setComparisonError('Cet identifiant est déjà présent dans la sélection.')
      return
    }

    if (comparisonSelections.length >= 6) {
      setComparisonError('Limitez la comparaison à six entités maximum pour conserver une lecture claire.')
      return
    }

    setComparisonSelections((previous) => [
      ...previous,
      { id: trimmed, label: trimmed, type: comparisonTypeInput, metadata: {} },
    ])
    setComparisonIdInput('')
    setComparisonError(null)
  }

  const handleComparisonSuggestionSelect = (item: AnalyticsSearchResult) => {
    if (comparisonSelections.some((entry) => entry.id === item.id)) {
      setComparisonError('Cet identifiant est déjà présent dans la sélection.')
      return
    }

    if (comparisonSelections.length >= 6) {
      setComparisonError('Limitez la comparaison à six entités maximum pour conserver une lecture claire.')
      return
    }

    setComparisonSelections((previous) => [...previous, item])
    setComparisonError(null)
    setComparisonQuery('')
  }

  const handleComparisonRemove = (identifier: string) => {
    setComparisonSelections((previous) => {
      const next = previous.filter((item) => item.id !== identifier)
      if (next.length < 2) {
        setComparisonFilters(null)
      }
      return next
    })
    setComparisonError(null)
  }

  const handleComparisonSubmit = (event: FormEvent) => {
    event.preventDefault()

    if (comparisonSelections.length < 2) {
      setComparisonError('Sélectionnez au moins deux entités à comparer.')
      setComparisonFilters(null)
      return
    }

    if (comparisonStartInput && comparisonEndInput && comparisonStartInput > comparisonEndInput) {
      setComparisonError('La date de début doit précéder la date de fin.')
      setComparisonFilters(null)
      return
    }

    setComparisonError(null)
    setComparisonFilters({
      entityType: comparisonTypeInput,
      entityIds: comparisonSelections.map((item) => item.id),
      hippodrome: comparisonHippoInput.trim() || undefined,
      startDate: comparisonStartInput || undefined,
      endDate: comparisonEndInput || undefined,
    })
  }

  const handleStreakSubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = streakIdInput.trim()

    if (!id) {
      setStreakError("Veuillez saisir un identifiant Aspiturf valide.")
      setStreakFilters(null)
      return
    }

    if (streakStartInput && streakEndInput && streakStartInput > streakEndInput) {
      setStreakError('La date de début doit précéder la date de fin.')
      setStreakFilters(null)
      return
    }

    setStreakError(null)
    setStreakFilters({
      entityType: streakTypeInput,
      entityId: id,
      hippodrome: streakHippoInput.trim() || undefined,
      startDate: streakStartInput || undefined,
      endDate: streakEndInput || undefined,
    })
  }

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

  const handleHorseSuggestionSelect = (suggestion: AnalyticsSearchResult) => {
    setHorseIdInput(suggestion.id)
    if (!horseHippoInput && suggestion.metadata.hippodromes?.length) {
      setHorseHippoInput(suggestion.metadata.hippodromes[0])
    }
    setHorseNameQuery('')
  }

  const handleJockeySuggestionSelect = (suggestion: AnalyticsSearchResult) => {
    setJockeyIdInput(suggestion.id)
    if (!jockeyHippoInput && suggestion.metadata.hippodromes?.length) {
      setJockeyHippoInput(suggestion.metadata.hippodromes[0])
    }
    setJockeyNameQuery('')
  }

  const handleTrainerSuggestionSelect = (suggestion: AnalyticsSearchResult) => {
    setTrainerIdInput(suggestion.id)
    if (!trainerHippoInput && suggestion.metadata.hippodromes?.length) {
      setTrainerHippoInput(suggestion.metadata.hippodromes[0])
    }
    setTrainerNameQuery('')
  }

  const handleHippodromeSuggestionSelect = (suggestion: AnalyticsSearchResult) => {
    setCourseHippoInput(suggestion.label)
    if (!horseHippoInput) setHorseHippoInput(suggestion.label)
    if (!jockeyHippoInput) setJockeyHippoInput(suggestion.label)
    if (!trainerHippoInput) setTrainerHippoInput(suggestion.label)
    if (!coupleHippoInput) setCoupleHippoInput(suggestion.label)
    setCourseHippoQuery('')
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
          title="Classements express"
          description="Identifiez en un clin d'œil les chevaux, jockeys et entraîneurs les plus performants sur une période donnée."
        >
          <form
            onSubmit={handleInsightsSubmit}
            className="grid gap-4 md:grid-cols-[repeat(4,minmax(0,1fr)),auto]"
          >
            <input
              value={insightHippoInput}
              onChange={(event) => setInsightHippoInput(event.target.value)}
              placeholder="Filtrer par hippodrome (optionnel)"
              className="input"
            />
            <input
              type="date"
              value={insightStartInput}
              onChange={(event) => setInsightStartInput(event.target.value)}
              placeholder="Date de début"
              className="input"
            />
            <input
              type="date"
              value={insightEndInput}
              onChange={(event) => setInsightEndInput(event.target.value)}
              placeholder="Date de fin"
              className="input"
            />
            <input
              type="number"
              min={1}
              max={20}
              value={insightLimitInput}
              onChange={(event) => setInsightLimitInput(event.target.value)}
              placeholder="Nombre d'entrées"
              className="input"
            />
            <button type="submit" className="btn btn-primary">
              Actualiser
            </button>
          </form>
          {insightError && <p className="text-sm text-red-600">{insightError}</p>}
          {insightsQuery.isLoading && (
            <p className="text-sm text-gray-500">Calcul des classements en cours…</p>
          )}
        {insightsQuery.isError && (
          <p className="text-sm text-red-600">Erreur: {(insightsQuery.error as Error).message}</p>
        )}
        {insightsQuery.data && <InsightsPanel data={insightsQuery.data} />}
      </SectionCard>

      <SectionCard
        title="Tendances de performance"
        description="Mesurez l'évolution d'un cheval, d'un jockey ou d'un entraîneur grâce à une agrégation hebdomadaire ou mensuelle."
      >
        <form
          onSubmit={handleTrendSubmit}
          className="grid gap-4 md:grid-cols-[repeat(6,minmax(0,1fr)),auto]"
        >
          <select
            value={trendTypeInput}
            onChange={(event) => setTrendTypeInput(event.target.value as TrendEntityType)}
            className="input"
          >
            <option value="horse">Cheval</option>
            <option value="jockey">Jockey</option>
            <option value="trainer">Entraîneur</option>
          </select>
          <input
            value={trendIdInput}
            onChange={(event) => setTrendIdInput(event.target.value)}
            placeholder="Identifiant Aspiturf (id)"
            className="input"
          />
          <input
            value={trendHippoInput}
            onChange={(event) => setTrendHippoInput(event.target.value)}
            placeholder="Filtrer par hippodrome (optionnel)"
            className="input"
          />
          <input
            type="date"
            value={trendStartInput}
            onChange={(event) => setTrendStartInput(event.target.value)}
            className="input"
          />
          <input
            type="date"
            value={trendEndInput}
            onChange={(event) => setTrendEndInput(event.target.value)}
            className="input"
          />
          <select
            value={trendGranularityInput}
            onChange={(event) => setTrendGranularityInput(event.target.value as TrendGranularity)}
            className="input"
          >
            <option value="month">Mois</option>
            <option value="week">Semaine</option>
          </select>
          <button type="submit" className="btn btn-primary">
            Générer
          </button>
        </form>
        {trendError && <p className="text-sm text-red-600">{trendError}</p>}
        {trendQuery.isPending && (
          <p className="text-sm text-gray-500">Calcul des tendances en cours…</p>
        )}
        {trendQuery.isError && (
          <p className="text-sm text-red-600">Erreur: {(trendQuery.error as Error).message}</p>
        )}
        {trendQuery.data && <TrendPanel data={trendQuery.data} />}
      </SectionCard>

      <SectionCard
        title="Indice de forme"
        description="Analysez les N dernières courses d'un cheval, d'un jockey ou d'un entraîneur pour connaître sa dynamique actuelle."
      >
        <form
          onSubmit={handleFormSubmit}
          className="grid gap-4 md:grid-cols-[repeat(6,minmax(0,1fr)),auto]"
        >
          <select
            value={formTypeInput}
            onChange={(event) => setFormTypeInput(event.target.value as TrendEntityType)}
            className="input"
          >
            <option value="horse">Cheval</option>
            <option value="jockey">Jockey</option>
            <option value="trainer">Entraîneur</option>
          </select>
          <input
            value={formIdInput}
            onChange={(event) => setFormIdInput(event.target.value)}
            placeholder="Identifiant Aspiturf (id)"
            className="input"
          />
          <input
            type="number"
            min={1}
            max={30}
            value={formWindowInput}
            onChange={(event) => setFormWindowInput(event.target.value)}
            placeholder="Fenêtre (courses)"
            className="input"
          />
          <input
            value={formHippoInput}
            onChange={(event) => setFormHippoInput(event.target.value)}
            placeholder="Filtrer par hippodrome (optionnel)"
            className="input"
          />
          <input
            type="date"
            value={formStartInput}
            onChange={(event) => setFormStartInput(event.target.value)}
            className="input"
          />
          <input
            type="date"
            value={formEndInput}
            onChange={(event) => setFormEndInput(event.target.value)}
            className="input"
          />
          <button type="submit" className="btn btn-primary">
            Calculer
          </button>
        </form>
        {formError && <p className="text-sm text-red-600">{formError}</p>}
        {formQuery.isPending && (
          <p className="text-sm text-gray-500">Calcul de l'indice de forme en cours…</p>
        )}
        {formQuery.isError && (
          <p className="text-sm text-red-600">Erreur: {(formQuery.error as Error).message}</p>
        )}
      {formQuery.data && <FormPanel data={formQuery.data} />}
    </SectionCard>

    <SectionCard
      title="Comparaison multi-entités"
      description="Contrastez les statistiques de plusieurs chevaux, jockeys ou entraîneurs et analysez leurs confrontations directes."
    >
      <form
        onSubmit={handleComparisonSubmit}
        className="grid gap-4 md:grid-cols-[repeat(5,minmax(0,1fr)),auto]"
      >
        <select
          value={comparisonTypeInput}
          onChange={(event) => {
            const nextType = event.target.value as TrendEntityType
            setComparisonTypeInput(nextType)
            setComparisonSelections([])
            setComparisonFilters(null)
          }}
          className="input"
        >
          <option value="horse">Cheval</option>
          <option value="jockey">Jockey</option>
          <option value="trainer">Entraîneur</option>
        </select>
        <div className="flex gap-2">
          <input
            value={comparisonIdInput}
            onChange={(event) => setComparisonIdInput(event.target.value)}
            placeholder="Ajouter un identifiant Aspiturf"
            className="input flex-1"
          />
          <button type="button" className="btn btn-secondary" onClick={handleComparisonAddManual}>
            Ajouter
          </button>
        </div>
        <input
          value={comparisonHippoInput}
          onChange={(event) => setComparisonHippoInput(event.target.value)}
          placeholder="Filtrer par hippodrome (optionnel)"
          className="input"
        />
        <input
          type="date"
          value={comparisonStartInput}
          onChange={(event) => setComparisonStartInput(event.target.value)}
          className="input"
        />
        <input
          type="date"
          value={comparisonEndInput}
          onChange={(event) => setComparisonEndInput(event.target.value)}
          className="input"
        />
        <button type="submit" className="btn btn-primary">
          Comparer
        </button>
      </form>

      {comparisonError && <p className="text-sm text-red-600">{comparisonError}</p>}

      <div className="mt-4 flex flex-wrap gap-2">
        {comparisonSelections.length ? (
          comparisonSelections.map((item) => (
            <span
              key={item.id}
              className="inline-flex items-center gap-2 rounded-full bg-indigo-50 px-3 py-1 text-sm text-indigo-700"
            >
              {item.label}
              <button
                type="button"
                className="text-xs font-semibold uppercase tracking-wide"
                onClick={() => handleComparisonRemove(item.id)}
              >
                Retirer
              </button>
            </span>
          ))
        ) : (
          <p className="text-sm text-gray-500">Sélectionnez au moins deux entités pour lancer la comparaison.</p>
        )}
      </div>

      <div className="mt-4 space-y-2 rounded-lg border border-dashed border-gray-200 bg-gray-50 p-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Recherche assistée</span>
          <span className="text-xs text-gray-500">2 lettres minimum</span>
        </div>
        <input
          value={comparisonQuery}
          onChange={(event) => setComparisonQuery(event.target.value)}
          placeholder="Rechercher une entité par nom ou identifiant"
          className="input"
        />
        {comparisonQuery.trim().length >= 2 ? (
          <SuggestionsList
            results={comparisonSuggestionsQuery.data}
            isLoading={comparisonSuggestionsQuery.isFetching}
            error={toError(comparisonSuggestionsQuery.error)}
            onSelect={handleComparisonSuggestionSelect}
            emptyLabel="Aucune entité ne correspond à cette recherche."
          />
        ) : (
          <p className="text-xs text-gray-500">
            Combinez l'autocomplétion et l'ajout manuel pour constituer votre liste de comparaisons.
          </p>
        )}
      </div>

      {comparisonQueryResult.isPending && (
        <p className="text-sm text-gray-500">Analyse comparative en cours…</p>
      )}
      {comparisonQueryResult.isError && (
        <p className="text-sm text-red-600">Erreur: {(comparisonQueryResult.error as Error).message}</p>
      )}
      {comparisonQueryResult.data && <ComparisonPanel data={comparisonQueryResult.data} />}
    </SectionCard>

    <SectionCard
      title="Répartition des performances"
        description="Décomposez les résultats d'un cheval, d'un jockey ou d'un entraîneur par distance, corde, hippodrome ou discipline."
      >
        <form
          onSubmit={handleDistributionSubmit}
          className="grid gap-4 md:grid-cols-[repeat(7,minmax(0,1fr)),auto]"
        >
          <select
            value={distributionTypeInput}
            onChange={(event) => setDistributionTypeInput(event.target.value as TrendEntityType)}
            className="input"
          >
            <option value="horse">Cheval</option>
            <option value="jockey">Jockey</option>
            <option value="trainer">Entraîneur</option>
          </select>
          <input
            value={distributionIdInput}
            onChange={(event) => setDistributionIdInput(event.target.value)}
            placeholder="Identifiant Aspiturf"
            className="input"
          />
          <select
            value={distributionDimensionInput}
            onChange={(event) => setDistributionDimensionInput(event.target.value as DistributionDimension)}
            className="input"
          >
            <option value="distance">Distance</option>
            <option value="draw">Numéro de corde</option>
            <option value="hippodrome">Hippodrome</option>
            <option value="discipline">Discipline</option>
          </select>
          <input
            value={distributionHippoInput}
            onChange={(event) => setDistributionHippoInput(event.target.value)}
            placeholder="Filtrer par hippodrome (optionnel)"
            className="input"
          />
          <input
            type="date"
            value={distributionStartInput}
            onChange={(event) => setDistributionStartInput(event.target.value)}
            className="input"
          />
          <input
            type="date"
            value={distributionEndInput}
            onChange={(event) => setDistributionEndInput(event.target.value)}
            className="input"
          />
          <input
            type="number"
            min={50}
            max={1000}
            step={50}
            value={distributionStepInput}
            onChange={(event) => setDistributionStepInput(event.target.value)}
            placeholder="Pas distance (m)"
            className="input"
            disabled={distributionDimensionInput !== 'distance'}
          />
          <button type="submit" className="btn btn-primary">
            Analyser
          </button>
        </form>
        {distributionError && <p className="text-sm text-red-600">{distributionError}</p>}
        {distributionQuery.isPending && (
          <p className="text-sm text-gray-500">Calcul de la distribution en cours…</p>
        )}
        {distributionQuery.isError && (
          <p className="text-sm text-red-600">Erreur: {(distributionQuery.error as Error).message}</p>
        )}
        {distributionQuery.data && <DistributionPanel data={distributionQuery.data} />}
      </SectionCard>

      <SectionCard
        title="Séries de résultats"
        description="Identifiez les séries de victoires et de podiums consécutifs pour un cheval, un jockey ou un entraîneur."
      >
        <form
          onSubmit={handleStreakSubmit}
          className="grid gap-4 md:grid-cols-[repeat(5,minmax(0,1fr)),auto]"
        >
          <select
            value={streakTypeInput}
            onChange={(event) => setStreakTypeInput(event.target.value as TrendEntityType)}
            className="input"
          >
            <option value="horse">Cheval</option>
            <option value="jockey">Jockey</option>
            <option value="trainer">Entraîneur</option>
          </select>
          <input
            value={streakIdInput}
            onChange={(event) => setStreakIdInput(event.target.value)}
            placeholder="Identifiant Aspiturf (id)"
            className="input"
          />
          <input
            value={streakHippoInput}
            onChange={(event) => setStreakHippoInput(event.target.value)}
            placeholder="Filtrer par hippodrome (optionnel)"
            className="input"
          />
          <input
            type="date"
            value={streakStartInput}
            onChange={(event) => setStreakStartInput(event.target.value)}
            className="input"
          />
          <input
            type="date"
            value={streakEndInput}
            onChange={(event) => setStreakEndInput(event.target.value)}
            className="input"
          />
          <button type="submit" className="btn btn-primary">
            Analyser
          </button>
        </form>
        {streakError && <p className="text-sm text-red-600">{streakError}</p>}
        {streakQuery.isPending && (
          <p className="text-sm text-gray-500">Calcul des séries en cours…</p>
        )}
        {streakQuery.isError && (
          <p className="text-sm text-red-600">Erreur: {(streakQuery.error as Error).message}</p>
        )}
        {streakQuery.data && <StreakPanel data={streakQuery.data} />}
      </SectionCard>

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
          <div className="space-y-2 rounded-lg border border-dashed border-gray-200 bg-gray-50 p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Recherche assistée</span>
              <span className="text-xs text-gray-500">2 lettres minimum</span>
            </div>
            <input
              value={horseNameQuery}
              onChange={(event) => setHorseNameQuery(event.target.value)}
              placeholder="Rechercher un cheval par nom ou identifiant"
              className="input"
            />
            {horseNameQuery.trim().length >= 2 ? (
              <SuggestionsList
                results={horseSuggestionsQuery.data}
                isLoading={horseSuggestionsQuery.isFetching}
                error={toError(horseSuggestionsQuery.error)}
                onSelect={handleHorseSuggestionSelect}
                emptyLabel="Aucune correspondance trouvée pour cette recherche."
              />
            ) : (
              <p className="text-xs text-gray-500">
                Tapez au moins deux lettres pour lister les chevaux correspondants dans le CSV Aspiturf.
              </p>
            )}
          </div>
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
          <div className="space-y-2 rounded-lg border border-dashed border-gray-200 bg-gray-50 p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Recherche assistée</span>
              <span className="text-xs text-gray-500">2 lettres minimum</span>
            </div>
            <input
              value={jockeyNameQuery}
              onChange={(event) => setJockeyNameQuery(event.target.value)}
              placeholder="Rechercher un jockey par nom ou identifiant"
              className="input"
            />
            {jockeyNameQuery.trim().length >= 2 ? (
              <SuggestionsList
                results={jockeySuggestionsQuery.data}
                isLoading={jockeySuggestionsQuery.isFetching}
                error={toError(jockeySuggestionsQuery.error)}
                onSelect={handleJockeySuggestionSelect}
                emptyLabel="Aucun jockey ne correspond à cette recherche."
              />
            ) : (
              <p className="text-xs text-gray-500">
                Utilisez la recherche pour récupérer rapidement l'identifiant Aspiturf d'un jockey.
              </p>
            )}
          </div>
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
          <div className="space-y-2 rounded-lg border border-dashed border-gray-200 bg-gray-50 p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Recherche assistée</span>
              <span className="text-xs text-gray-500">2 lettres minimum</span>
            </div>
            <input
              value={trainerNameQuery}
              onChange={(event) => setTrainerNameQuery(event.target.value)}
              placeholder="Rechercher un entraîneur par nom ou identifiant"
              className="input"
            />
            {trainerNameQuery.trim().length >= 2 ? (
              <SuggestionsList
                results={trainerSuggestionsQuery.data}
                isLoading={trainerSuggestionsQuery.isFetching}
                error={toError(trainerSuggestionsQuery.error)}
                onSelect={handleTrainerSuggestionSelect}
                emptyLabel="Aucun entraîneur ne correspond à cette recherche."
              />
            ) : (
              <p className="text-xs text-gray-500">
                Trouvez instantanément l'identifiant Aspiturf d'un entraîneur pour l'analyse détaillée.
              </p>
            )}
          </div>
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
          <div className="space-y-2 rounded-lg border border-dashed border-gray-200 bg-gray-50 p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Rechercher un hippodrome</span>
              <span className="text-xs text-gray-500">2 lettres minimum</span>
            </div>
            <input
              value={courseHippoQuery}
              onChange={(event) => setCourseHippoQuery(event.target.value)}
              placeholder="Saisir le nom d'un hippodrome Aspiturf"
              className="input"
            />
            {courseHippoQuery.trim().length >= 2 ? (
              <SuggestionsList
                results={hippodromeSuggestionsQuery.data}
                isLoading={hippodromeSuggestionsQuery.isFetching}
                error={toError(hippodromeSuggestionsQuery.error)}
                onSelect={handleHippodromeSuggestionSelect}
                emptyLabel="Aucun hippodrome trouvé avec ce terme."
              />
            ) : (
              <p className="text-xs text-gray-500">
                Sélectionnez un hippodrome pour pré-remplir les filtres des différents formulaires.
              </p>
            )}
          </div>
          {courseQuery.data && <CourseAnalyticsPanel data={courseQuery.data} />}
        </SectionCard>
      </div>
    </div>
  )
}
