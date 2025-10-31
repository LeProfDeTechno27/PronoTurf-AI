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
  AnalyticsValueResponse,
  AnalyticsVolatilityResponse,
  AnalyticsEfficiencyResponse,
  AnalyticsWorkloadResponse,
  AnalyticsMomentumResponse,
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
  AnalyticsCalendarResponse,
  CalendarDaySummary,
  CalendarRaceDetail,
  ValueOpportunitySample,
  VolatilityRaceSample,
  EfficiencySample,
  WorkloadTimelineEntry,
  MomentumSlice,
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

type CalendarFilters = {
  entityType: TrendEntityType
  entityId: string
  hippodrome?: string
  startDate?: string
  endDate?: string
}

type ValueFilters = {
  entityType: TrendEntityType
  entityId: string
  hippodrome?: string
  startDate?: string
  endDate?: string
  minEdge?: number
  limit?: number
}

type VolatilityFilters = {
  entityType: TrendEntityType
  entityId: string
  hippodrome?: string
  startDate?: string
  endDate?: string
}

type EfficiencyFilters = {
  entityType: TrendEntityType
  entityId: string
  hippodrome?: string
  startDate?: string
  endDate?: string
}

type WorkloadFilters = {
  entityType: TrendEntityType
  entityId: string
  hippodrome?: string
  startDate?: string
  endDate?: string
}

type MomentumFilters = {
  entityType: TrendEntityType
  entityId: string
  hippodrome?: string
  startDate?: string
  endDate?: string
  window?: number
  baselineWindow?: number
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


function CalendarRaceTable({ details }: { details: CalendarRaceDetail[] }) {
  if (!details.length) {
    return null
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
              Hippodrome
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
              Course
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
              Distance
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
              Position
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
              Cote
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {details.map((detail, index) => {
            const key = `${detail.hippodrome ?? 'hippo'}-${detail.course_number ?? index}`
            return (
              <tr key={key}>
                <td className="px-4 py-2 text-sm text-gray-900">{detail.hippodrome ?? '—'}</td>
                <td className="px-4 py-2 text-sm text-gray-900">
                  {detail.course_number != null ? `N° ${detail.course_number}` : '—'}
                </td>
                <td className="px-4 py-2 text-sm text-gray-900">
                  {detail.distance != null ? `${formatNumber(detail.distance)} m` : '—'}
                </td>
                <td className="px-4 py-2 text-sm text-gray-900">
                  {detail.final_position != null ? detail.final_position : '—'}
                </td>
                <td className="px-4 py-2 text-sm text-gray-900">
                  {detail.odds != null ? detail.odds.toFixed(2) : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}


function CalendarDayCard({ day }: { day: CalendarDaySummary }) {
  const hippoLabel = day.hippodromes.length ? day.hippodromes.join(', ') : 'Tous hippodromes'

  return (
    <div className="space-y-4 rounded-lg border border-gray-200 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-lg font-semibold text-gray-900">{formatDate(day.date)}</h4>
        <span className="text-sm text-gray-500">{hippoLabel}</span>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <SummaryStats label="Courses" value={formatNumber(day.races)} />
        <SummaryStats label="Victoires" value={formatNumber(day.wins)} />
        <SummaryStats label="Podiums" value={formatNumber(day.podiums)} />
        <SummaryStats label="Position moyenne" value={formatAverage(day.average_finish)} />
        <SummaryStats label="Cote moyenne" value={formatAverage(day.average_odds)} />
        <SummaryStats label="Lignes détaillées" value={`${day.race_details.length} courses`} />
      </div>

      <CalendarRaceTable details={day.race_details} />
    </div>
  )
}


function CalendarPanel({ data }: { data: AnalyticsCalendarResponse }) {
  const entityDisplay = data.entity_label ? `${data.entity_label} (${data.entity_id})` : data.entity_id
  const winRate = data.total_races ? data.total_wins / data.total_races : null
  const podiumRate = data.total_races ? data.total_podiums / data.total_races : null

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Entité analysée" value={entityDisplay} />
        <SummaryStats label="Total courses" value={formatNumber(data.total_races)} />
        <SummaryStats label="Victoires" value={`${formatNumber(data.total_wins)} (${formatPercent(winRate)})`} />
        <SummaryStats label="Podiums" value={`${formatNumber(data.total_podiums)} (${formatPercent(podiumRate)})`} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats
          label="Plage de dates"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
        <SummaryStats
          label="Filtre hippodrome"
          value={data.metadata.hippodrome_filter ? data.metadata.hippodrome_filter.toUpperCase() : 'Tous'}
        />
        <SummaryStats label="Journées analysées" value={formatNumber(data.days.length)} />
      </div>

      <div className="space-y-4">
        {data.days.map((day) => (
          <CalendarDayCard key={day.date} day={day} />
        ))}
      </div>
    </div>
  )
}

function ValueOpportunitiesTable({ samples }: { samples: ValueOpportunitySample[] }) {
  if (!samples.length) {
    return <p className="text-sm text-gray-500">Aucune opportunité retenue.</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 rounded-lg border">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">Date</th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
              Hippodrome
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Course
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Distance
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Position
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Cote observée
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Cote probable
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Écart
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Gain unitaire
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {samples.map((sample, index) => (
            <tr key={`${sample.date}-${sample.course_number}-${index}`}>
              <td className="px-4 py-2 text-sm text-gray-700">{formatDate(sample.date ?? null)}</td>
              <td className="px-4 py-2 text-sm text-gray-700">{sample.hippodrome ?? '—'}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(sample.course_number)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(sample.distance)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(sample.final_position)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(sample.odds_actual)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(sample.odds_implied)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(sample.edge)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(sample.profit)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ValuePanel({ data }: { data: AnalyticsValueResponse }) {
  const entityDisplay = data.entity_label ? `${data.entity_label} (${data.entity_id})` : data.entity_id
  const winRate = data.win_rate ?? (data.sample_size ? data.wins / data.sample_size : null)
  const positiveRatio = data.sample_size ? data.positive_edges / data.sample_size : null

  return (
    <div className="space-y-6">
      {/* Bloc de synthèse pour visualiser d'un coup d'œil le potentiel value bet. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Entité analysée" value={entityDisplay} />
        <SummaryStats label="Courses étudiées" value={formatNumber(data.sample_size)} />
        <SummaryStats label="Victoires" value={`${formatNumber(data.wins)} (${formatPercent(winRate)})`} />
        <SummaryStats
          label="Opportunités positives"
          value={`${formatNumber(data.positive_edges)} (${formatPercent(positiveRatio)})`}
        />
      </div>

      {/* Information financière et probabiliste pour piloter la prise de décision. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Écart moyen" value={formatAverage(data.average_edge)} />
        <SummaryStats label="Cote moyenne" value={formatAverage(data.average_odds)} />
        <SummaryStats label="ROI théorique" value={data.roi != null ? `${formatAverage(data.roi, 2)}x` : '—'} />
        <SummaryStats label="Gain cumulé" value={formatAverage(data.profit)} />
      </div>

      {/* Rappels contextuels pour vérifier la cohérence des filtres appliqués. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats
          label="Plage de dates"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
        <SummaryStats
          label="Filtre hippodrome"
          value={data.metadata.hippodrome_filter ? data.metadata.hippodrome_filter.toUpperCase() : 'Tous'}
        />
        <SummaryStats label="Hippodromes couverts" value={formatList(data.hippodromes, 5)} />
      </div>

      <ValueOpportunitiesTable samples={data.samples} />
    </div>
  )
}

function VolatilityTable({ races }: { races: VolatilityRaceSample[] }) {
  if (!races.length) {
    return <p className="text-sm text-gray-500">Aucune course disponible pour calculer la volatilité.</p>
  }

  return (
    <div className="overflow-x-auto rounded-lg shadow ring-1 ring-black/5">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
              Date
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
              Hippodrome
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Course
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Distance
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Arrivée
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Cote observée
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Cote probable
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Écart
            </th>
            <th className="px-4 py-2 text-center text-xs font-semibold uppercase tracking-wider text-gray-500">
              Victoire
            </th>
            <th className="px-4 py-2 text-center text-xs font-semibold uppercase tracking-wider text-gray-500">
              Podium
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {races.map((race, index) => (
            <tr key={`${race.date}-${race.course_number}-${index}`}>
              <td className="px-4 py-2 text-sm text-gray-700">{formatDate(race.date ?? null)}</td>
              <td className="px-4 py-2 text-sm text-gray-700">{race.hippodrome ?? '—'}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(race.course_number)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">
                {race.distance ? `${race.distance.toLocaleString('fr-FR')} m` : '—'}
              </td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(race.final_position)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(race.odds_actual)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(race.odds_implied)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(race.edge)}</td>
              <td className="px-4 py-2 text-sm text-center text-gray-700">{race.is_win ? 'Oui' : 'Non'}</td>
              <td className="px-4 py-2 text-sm text-center text-gray-700">{race.is_podium ? 'Oui' : 'Non'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function VolatilityPanel({ data }: { data: AnalyticsVolatilityResponse }) {
  const entityDisplay = data.entity_label ? `${data.entity_label} (${data.entity_id})` : data.entity_id
  const podiumRate = data.metrics.podium_rate ??
    (data.metrics.sample_size ? data.metrics.podiums / data.metrics.sample_size : null)

  return (
    <div className="space-y-6">
      {/* Résumé chiffré pour comprendre instantanément la régularité de l'entité. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Entité analysée" value={entityDisplay} />
        <SummaryStats label="Courses retenues" value={formatNumber(data.metrics.sample_size)} />
        <SummaryStats
          label="Taux de victoire"
          value={`${formatNumber(data.metrics.wins)} (${formatPercent(data.metrics.win_rate)})`}
        />
        <SummaryStats
          label="Taux de podium"
          value={`${formatNumber(data.metrics.podiums)} (${formatPercent(podiumRate)})`}
        />
      </div>

      {/* Mesures de dispersion pour détecter les profils irréguliers. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Position moyenne" value={formatAverage(data.metrics.average_finish)} />
        <SummaryStats label="Écart-type position" value={formatAverage(data.metrics.position_std_dev, 3)} />
        <SummaryStats label="Cote moyenne" value={formatAverage(data.metrics.average_odds)} />
        <SummaryStats label="Dispersion cote" value={formatAverage(data.metrics.odds_std_dev, 3)} />
      </div>

      {/* Synthèse des écarts de cote et rappel des filtres appliqués. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Écart moyen" value={formatAverage(data.metrics.average_edge)} />
        <SummaryStats
          label="Indice de constance"
          value={formatPercent(data.metrics.consistency_index, 1)}
        />
        <SummaryStats
          label="Plage de dates"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
        <SummaryStats
          label="Filtre hippodrome"
          value={data.metadata.hippodrome_filter ? data.metadata.hippodrome_filter.toUpperCase() : 'Tous'}
        />
      </div>

      <VolatilityTable races={data.races} />
    </div>
  )
}

function EfficiencyTable({ samples }: { samples: EfficiencySample[] }) {
  if (!samples.length) {
    return <p className="text-sm text-gray-500">Aucune course disposant d'une cote exploitable.</p>
  }

  return (
    <div className="overflow-x-auto rounded-lg shadow ring-1 ring-black/5">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
              Date
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
              Hippodrome
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Course
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Cote
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Proba victoire
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Proba podium
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Arrivée
            </th>
            <th className="px-4 py-2 text-center text-xs font-semibold uppercase tracking-wider text-gray-500">
              Victoire
            </th>
            <th className="px-4 py-2 text-center text-xs font-semibold uppercase tracking-wider text-gray-500">
              Podium
            </th>
            <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
              Edge
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {samples.map((sample, index) => (
            <tr key={`${sample.date}-${sample.course_number}-${index}`}>
              <td className="px-4 py-2 text-sm text-gray-700">{formatDate(sample.date ?? null)}</td>
              <td className="px-4 py-2 text-sm text-gray-700">{sample.hippodrome ?? '—'}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(sample.course_number)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(sample.odds)}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">
                {formatPercent(sample.expected_win_probability, 1)}
              </td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">
                {formatPercent(sample.expected_podium_probability, 1)}
              </td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatNumber(sample.finish_position)}</td>
              <td className="px-4 py-2 text-sm text-center text-gray-700">{sample.is_win ? 'Oui' : 'Non'}</td>
              <td className="px-4 py-2 text-sm text-center text-gray-700">{sample.is_podium ? 'Oui' : 'Non'}</td>
              <td className="px-4 py-2 text-sm text-right text-gray-700">{formatAverage(sample.edge, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function EfficiencyPanel({ data }: { data: AnalyticsEfficiencyResponse }) {
  const entityDisplay = data.entity_label ? `${data.entity_label} (${data.entity_id})` : data.entity_id
  const winDelta = data.metrics.win_delta ?? null
  const podiumDelta = data.metrics.podium_delta ?? null
  const formatDelta = (value: number | null) => {
    if (value == null) {
      return '—'
    }
    const prefix = value > 0 ? '+' : ''
    return `${prefix}${value.toFixed(2)}`
  }

  return (
    <div className="space-y-6">
      {/* Cartouche synthétique pour visualiser d'un coup d'œil les écarts attendus/observés. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Entité analysée" value={entityDisplay} />
        <SummaryStats label="Courses analysées" value={formatNumber(data.metrics.sample_size)} />
        <SummaryStats
          label="Victoires observées"
          value={`${formatNumber(data.metrics.wins)} (Δ ${formatDelta(winDelta)})`}
        />
        <SummaryStats
          label="Podiums observés"
          value={`${formatNumber(data.metrics.podiums)} (Δ ${formatDelta(podiumDelta)})`}
        />
      </div>

      {/* Indicateurs probabilistes et financiers pour qualifier l'efficacité du profil. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Victoires attendues" value={formatAverage(data.metrics.expected_wins)} />
        <SummaryStats label="Podiums attendus" value={formatAverage(data.metrics.expected_podiums)} />
        <SummaryStats label="Cote moyenne" value={formatAverage(data.metrics.average_odds)} />
        <SummaryStats
          label="Probabilité implicite"
          value={formatPercent(data.metrics.average_expected_win_probability, 1)}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Courses avec cote" value={formatNumber(data.metrics.stake_count)} />
        <SummaryStats label="Profit théorique" value={formatAverage(data.metrics.profit)} />
        <SummaryStats label="ROI théorique" value={formatAverage(data.metrics.roi)} />
        <SummaryStats
          label="Filtre hippodrome"
          value={data.metadata.hippodrome_filter ? data.metadata.hippodrome_filter.toUpperCase() : 'Tous'}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats
          label="Plage de dates"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
        <SummaryStats label="Identifiant" value={data.entity_id} />
        <SummaryStats label="Libellé" value={data.entity_label ?? '—'} />
      </div>

      {/* Tableau détaillé pour identifier les courses responsables des écarts majeurs. */}
      <EfficiencyTable samples={data.samples} />
    </div>
  )
}

function WorkloadPanel({ data }: { data: AnalyticsWorkloadResponse }) {
  const entityDisplay = data.entity_label ? `${data.entity_label} (${data.entity_id})` : data.entity_id
  const { summary, timeline } = data
  const distributionEntries = Object.entries(summary.rest_distribution)

  const formatRest = (value?: number | null, digits = 1) =>
    value == null ? '—' : `${value.toFixed(digits)} j`

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Entité analysée" value={entityDisplay} />
        <SummaryStats
          label="Plage des données"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
        <SummaryStats label="Courses analysées" value={formatNumber(summary.sample_size)} />
        <SummaryStats label="Repos moyen" value={formatRest(summary.average_rest_days)} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats
          label="Victoires / Podiums"
          value={`${formatNumber(summary.wins)} / ${formatNumber(summary.podiums)}`}
        />
        <SummaryStats label="Taux de victoire" value={formatPercent(summary.win_rate)} />
        <SummaryStats label="Repos médian" value={formatRest(summary.median_rest_days)} />
        <SummaryStats
          label="Activité 30j / 90j"
          value={`${formatNumber(summary.races_last_30_days)} / ${formatNumber(summary.races_last_90_days)}`}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats label="Repos min" value={formatRest(summary.shortest_rest_days, 0)} />
        <SummaryStats label="Repos max" value={formatRest(summary.longest_rest_days, 0)} />
        <SummaryStats
          label="Rythme mensuel"
          value={summary.average_monthly_races != null ? `${summary.average_monthly_races.toFixed(2)} courses` : '—'}
        />
      </div>

      {distributionEntries.length ? (
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="text-sm font-semibold text-gray-700">Répartition des repos</h4>
          <dl className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-3">
            {distributionEntries.map(([label, count]) => (
              <div key={label} className="rounded-md bg-slate-50 p-3">
                <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</dt>
                <dd className="text-lg font-semibold text-slate-800">{formatNumber(count)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : null}

      <div className="overflow-hidden rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                Date
              </th>
              <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                Hippodrome
              </th>
              <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                Course
              </th>
              <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                Distance
              </th>
              <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                Arrivée
              </th>
              <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                Repos
              </th>
              <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                Cote
              </th>
              <th className="px-3 py-2 text-center text-xs font-semibold uppercase tracking-wider text-gray-500">
                Victoire
              </th>
              <th className="px-3 py-2 text-center text-xs font-semibold uppercase tracking-wider text-gray-500">
                Podium
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {timeline.map((race: WorkloadTimelineEntry, index) => (
              <tr key={`${race.date}-${race.course_number}-${index}`}>
                <td className="px-3 py-2 text-sm text-gray-700">{formatDate(race.date ?? null)}</td>
                <td className="px-3 py-2 text-sm text-gray-700">{race.hippodrome ?? '—'}</td>
                <td className="px-3 py-2 text-sm text-right text-gray-700">{formatNumber(race.course_number)}</td>
                <td className="px-3 py-2 text-sm text-right text-gray-700">
                  {race.distance ? `${race.distance.toLocaleString('fr-FR')} m` : '—'}
                </td>
                <td className="px-3 py-2 text-sm text-right text-gray-700">{formatNumber(race.final_position)}</td>
                <td className="px-3 py-2 text-sm text-right text-gray-700">
                  {race.rest_days != null ? `${race.rest_days} j` : '—'}
                </td>
                <td className="px-3 py-2 text-sm text-right text-gray-700">{formatAverage(race.odds)}</td>
                <td className="px-3 py-2 text-sm text-center text-gray-700">{race.is_win ? 'Oui' : 'Non'}</td>
                <td className="px-3 py-2 text-sm text-center text-gray-700">{race.is_podium ? 'Oui' : 'Non'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function MomentumSliceTable({ title, slice }: { title: string; slice: MomentumSlice }) {
  return (
    <div className="space-y-4 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-lg font-semibold text-gray-900">{title}</h4>
          <p className="text-sm text-gray-500">
            {`${formatDate(slice.start_date ?? null)} → ${formatDate(slice.end_date ?? null)}`}
          </p>
        </div>
        <span className="rounded-full bg-indigo-50 px-3 py-1 text-sm font-medium text-indigo-700">
          {slice.label}
        </span>
      </div>

      {/* Synthèse compacte pour visualiser rapidement les indicateurs clefs de la fenêtre. */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <SummaryStats label="Courses" value={formatNumber(slice.race_count)} />
        <SummaryStats label="Victoires" value={formatNumber(slice.wins)} />
        <SummaryStats label="Podiums" value={formatNumber(slice.podiums)} />
        <SummaryStats label="Taux de victoire" value={formatPercent(slice.win_rate, 1)} />
        <SummaryStats label="Taux de podium" value={formatPercent(slice.podium_rate, 1)} />
        <SummaryStats label="Position moyenne" value={formatAverage(slice.average_finish)} />
        <SummaryStats label="Cote moyenne" value={formatAverage(slice.average_odds)} />
        <SummaryStats label="ROI théorique" value={formatAverage(slice.roi)} />
      </div>

      {slice.races.length ? (
        <div className="overflow-x-auto rounded-lg ring-1 ring-black/5">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                  Date
                </th>
                <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                  Hippodrome
                </th>
                <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                  Course
                </th>
                <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                  Distance
                </th>
                <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                  Arrivée
                </th>
                <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                  Cote
                </th>
                <th className="px-3 py-2 text-center text-xs font-semibold uppercase tracking-wider text-gray-500">
                  Victoire
                </th>
                <th className="px-3 py-2 text-center text-xs font-semibold uppercase tracking-wider text-gray-500">
                  Podium
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {slice.races.map((race, index) => (
                <tr key={`${race.date}-${race.course_number}-${index}`}>
                  <td className="px-3 py-2 text-sm text-gray-700">{formatDate(race.date ?? null)}</td>
                  <td className="px-3 py-2 text-sm text-gray-700">{race.hippodrome ?? '—'}</td>
                  <td className="px-3 py-2 text-sm text-right text-gray-700">{formatNumber(race.course_number)}</td>
                  <td className="px-3 py-2 text-sm text-right text-gray-700">
                    {race.distance ? `${race.distance.toLocaleString('fr-FR')} m` : '—'}
                  </td>
                  <td className="px-3 py-2 text-sm text-right text-gray-700">{formatNumber(race.final_position)}</td>
                  <td className="px-3 py-2 text-sm text-right text-gray-700">{formatAverage(race.odds)}</td>
                  <td className="px-3 py-2 text-sm text-center text-gray-700">{race.is_win ? 'Oui' : 'Non'}</td>
                  <td className="px-3 py-2 text-sm text-center text-gray-700">{race.is_podium ? 'Oui' : 'Non'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-gray-500">
          Aucune course ne correspond aux filtres pour cette fenêtre temporelle.
        </p>
      )}
    </div>
  )
}

function MomentumPanel({ data }: { data: AnalyticsMomentumResponse }) {
  const entityDisplay = data.entity_label ? `${data.entity_label} (${data.entity_id})` : data.entity_id

  return (
    <div className="space-y-6">
      {/* Comparaison synthétique entre la fenêtre récente et la fenêtre de référence. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <SummaryStats label="Entité analysée" value={entityDisplay} />
        <SummaryStats label="Fenêtre récente" value={data.recent_window.label} />
        <SummaryStats
          label="Fenêtre de référence"
          value={data.reference_window ? data.reference_window.label : 'Insuffisant'}
        />
        <SummaryStats
          label="Filtre hippodrome"
          value={data.metadata.hippodrome_filter ? data.metadata.hippodrome_filter.toUpperCase() : 'Tous'}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryStats
          label="Plage des données"
          value={`${formatDate(data.metadata.date_start)} → ${formatDate(data.metadata.date_end)}`}
        />
        <SummaryStats label="Évolution du podium" value={formatPercent(data.deltas.podium_rate, 1)} />
        <SummaryStats label="Variation ROI" value={formatAverage(data.deltas.roi)} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <MomentumSliceTable title="Fenêtre récente" slice={data.recent_window} />
        {data.reference_window ? (
          <MomentumSliceTable title="Fenêtre de référence" slice={data.reference_window} />
        ) : (
          <div className="rounded-lg border border-dashed border-gray-300 p-6 text-sm text-gray-500">
            <p>
              Ajoutez davantage de courses (paramètre « baseline ») pour comparer la dynamique à une période
              précédente.
            </p>
          </div>
        )}
      </div>
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

  const [calendarTypeInput, setCalendarTypeInput] = useState<TrendEntityType>('horse')
  const [calendarIdInput, setCalendarIdInput] = useState('')
  const [calendarHippoInput, setCalendarHippoInput] = useState('')
  const [calendarStartInput, setCalendarStartInput] = useState('')
  const [calendarEndInput, setCalendarEndInput] = useState('')
  const [calendarError, setCalendarError] = useState<string | null>(null)
  const [calendarFilters, setCalendarFilters] = useState<CalendarFilters | null>(null)

  const [valueTypeInput, setValueTypeInput] = useState<TrendEntityType>('horse')
  const [valueIdInput, setValueIdInput] = useState('')
  const [valueHippoInput, setValueHippoInput] = useState('')
  const [valueStartInput, setValueStartInput] = useState('')
  const [valueEndInput, setValueEndInput] = useState('')
  const [valueMinEdgeInput, setValueMinEdgeInput] = useState('0.0')
  const [valueLimitInput, setValueLimitInput] = useState('25')
  const [valueError, setValueError] = useState<string | null>(null)
  const [valueFilters, setValueFilters] = useState<ValueFilters | null>(null)

  const [volatilityTypeInput, setVolatilityTypeInput] = useState<TrendEntityType>('horse')
  const [volatilityIdInput, setVolatilityIdInput] = useState('')
  const [volatilityHippoInput, setVolatilityHippoInput] = useState('')
  const [volatilityStartInput, setVolatilityStartInput] = useState('')
  const [volatilityEndInput, setVolatilityEndInput] = useState('')
  const [volatilityError, setVolatilityError] = useState<string | null>(null)
  const [volatilityFilters, setVolatilityFilters] = useState<VolatilityFilters | null>(null)

  const [efficiencyTypeInput, setEfficiencyTypeInput] = useState<TrendEntityType>('horse')
  const [efficiencyIdInput, setEfficiencyIdInput] = useState('')
  const [efficiencyHippoInput, setEfficiencyHippoInput] = useState('')
  const [efficiencyStartInput, setEfficiencyStartInput] = useState('')
  const [efficiencyEndInput, setEfficiencyEndInput] = useState('')
  const [efficiencyError, setEfficiencyError] = useState<string | null>(null)
  const [efficiencyFilters, setEfficiencyFilters] = useState<EfficiencyFilters | null>(null)

  const [workloadTypeInput, setWorkloadTypeInput] = useState<TrendEntityType>('horse')
  const [workloadIdInput, setWorkloadIdInput] = useState('')
  const [workloadHippoInput, setWorkloadHippoInput] = useState('')
  const [workloadStartInput, setWorkloadStartInput] = useState('')
  const [workloadEndInput, setWorkloadEndInput] = useState('')
  const [workloadError, setWorkloadError] = useState<string | null>(null)
  const [workloadFilters, setWorkloadFilters] = useState<WorkloadFilters | null>(null)

  const [momentumTypeInput, setMomentumTypeInput] = useState<TrendEntityType>('horse')
  const [momentumIdInput, setMomentumIdInput] = useState('')
  const [momentumHippoInput, setMomentumHippoInput] = useState('')
  const [momentumStartInput, setMomentumStartInput] = useState('')
  const [momentumEndInput, setMomentumEndInput] = useState('')
  const [momentumWindowInput, setMomentumWindowInput] = useState('5')
  const [momentumBaselineInput, setMomentumBaselineInput] = useState('5')
  const [momentumError, setMomentumError] = useState<string | null>(null)
  const [momentumFilters, setMomentumFilters] = useState<MomentumFilters | null>(null)

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

  const calendarQueryKey = useMemo(
    () =>
      calendarFilters
        ? [
            'analytics',
            'calendar',
            calendarFilters.entityType,
            calendarFilters.entityId,
            calendarFilters.hippodrome ?? '',
            calendarFilters.startDate ?? '',
            calendarFilters.endDate ?? '',
          ]
        : ['analytics', 'calendar', 'idle'],
    [calendarFilters],
  )

  const calendarQuery = useQuery({
    queryKey: calendarQueryKey,
    queryFn: () =>
      analyticsService.getPerformanceCalendar({
        entityType: calendarFilters!.entityType,
        entityId: calendarFilters!.entityId,
        hippodrome: calendarFilters?.hippodrome,
        startDate: calendarFilters?.startDate,
        endDate: calendarFilters?.endDate,
      }),
    enabled: Boolean(calendarFilters?.entityId),
  })

  const valueQueryKey = useMemo(
    () =>
      valueFilters
        ? [
            'analytics',
            'value',
            valueFilters.entityType,
            valueFilters.entityId,
            valueFilters.hippodrome ?? '',
            valueFilters.startDate ?? '',
            valueFilters.endDate ?? '',
            valueFilters.minEdge ?? '',
            valueFilters.limit ?? '',
          ]
        : ['analytics', 'value', 'idle'],
    [valueFilters],
  )

  const volatilityQueryKey = useMemo(
    () =>
      volatilityFilters
        ? [
            'analytics',
            'volatility',
            volatilityFilters.entityType,
            volatilityFilters.entityId,
            volatilityFilters.hippodrome ?? '',
            volatilityFilters.startDate ?? '',
            volatilityFilters.endDate ?? '',
          ]
        : ['analytics', 'volatility', 'idle'],
    [volatilityFilters],
  )

  const valueQuery = useQuery({
    queryKey: valueQueryKey,
    queryFn: () =>
      analyticsService.getValueOpportunities({
        entityType: valueFilters!.entityType,
        entityId: valueFilters!.entityId,
        hippodrome: valueFilters?.hippodrome,
        startDate: valueFilters?.startDate,
        endDate: valueFilters?.endDate,
        minEdge: valueFilters?.minEdge,
        limit: valueFilters?.limit,
      }),
    enabled: Boolean(valueFilters?.entityId),
  })

  const volatilityQuery = useQuery({
    queryKey: volatilityQueryKey,
    queryFn: () =>
      analyticsService.getVolatilityProfile({
        entityType: volatilityFilters!.entityType,
        entityId: volatilityFilters!.entityId,
        hippodrome: volatilityFilters?.hippodrome,
        startDate: volatilityFilters?.startDate,
        endDate: volatilityFilters?.endDate,
      }),
    enabled: Boolean(volatilityFilters?.entityId),
  })

  const efficiencyQueryKey = useMemo(
    () =>
      efficiencyFilters
        ? [
            'analytics',
            'efficiency',
            efficiencyFilters.entityType,
            efficiencyFilters.entityId,
            efficiencyFilters.hippodrome ?? '',
            efficiencyFilters.startDate ?? '',
            efficiencyFilters.endDate ?? '',
          ]
        : ['analytics', 'efficiency', 'idle'],
    [efficiencyFilters],
  )

  const efficiencyQuery = useQuery({
    queryKey: efficiencyQueryKey,
    queryFn: () =>
      analyticsService.getEfficiencyProfile({
        entityType: efficiencyFilters!.entityType,
        entityId: efficiencyFilters!.entityId,
        hippodrome: efficiencyFilters?.hippodrome,
        startDate: efficiencyFilters?.startDate,
        endDate: efficiencyFilters?.endDate,
      }),
    enabled: Boolean(efficiencyFilters?.entityId),
  })

  const workloadQueryKey = useMemo(
    () =>
      workloadFilters
        ? [
            'analytics',
            'workload',
            workloadFilters.entityType,
            workloadFilters.entityId,
            workloadFilters.hippodrome ?? '',
            workloadFilters.startDate ?? '',
            workloadFilters.endDate ?? '',
          ]
        : ['analytics', 'workload', 'idle'],
    [workloadFilters],
  )

  const workloadQuery = useQuery({
    queryKey: workloadQueryKey,
    queryFn: () =>
      analyticsService.getWorkloadProfile({
        entityType: workloadFilters!.entityType,
        entityId: workloadFilters!.entityId,
        hippodrome: workloadFilters?.hippodrome,
        startDate: workloadFilters?.startDate,
        endDate: workloadFilters?.endDate,
      }),
    enabled: Boolean(workloadFilters?.entityId),
  })

  const momentumQueryKey = useMemo(
    () =>
      momentumFilters
        ? [
            'analytics',
            'momentum',
            momentumFilters.entityType,
            momentumFilters.entityId,
            momentumFilters.hippodrome ?? '',
            momentumFilters.startDate ?? '',
            momentumFilters.endDate ?? '',
            momentumFilters.window ?? '',
            momentumFilters.baselineWindow ?? '',
          ]
        : ['analytics', 'momentum', 'idle'],
    [momentumFilters],
  )

  const momentumQuery = useQuery({
    queryKey: momentumQueryKey,
    queryFn: () =>
      analyticsService.getMomentumProfile({
        entityType: momentumFilters!.entityType,
        entityId: momentumFilters!.entityId,
        hippodrome: momentumFilters?.hippodrome,
        startDate: momentumFilters?.startDate,
        endDate: momentumFilters?.endDate,
        window: momentumFilters?.window,
        baselineWindow: momentumFilters?.baselineWindow,
      }),
    enabled: Boolean(momentumFilters?.entityId),
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

  const handleValueSubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = valueIdInput.trim()
    const minEdgeValue = Number(valueMinEdgeInput.trim() || '0')
    const limitValue = Number(valueLimitInput.trim() || '25')

    if (!id) {
      setValueError("Veuillez saisir un identifiant Aspiturf valide.")
      setValueFilters(null)
      return
    }

    if (Number.isNaN(minEdgeValue) || minEdgeValue < 0) {
      setValueError("Le seuil d'écart doit être positif ou nul.")
      setValueFilters(null)
      return
    }

    if (Number.isNaN(limitValue) || limitValue < 5 || limitValue > 100) {
      setValueError('Le nombre de courses doit être compris entre 5 et 100.')
      setValueFilters(null)
      return
    }

    if (valueStartInput && valueEndInput && valueStartInput > valueEndInput) {
      setValueError('La date de début doit précéder la date de fin.')
      setValueFilters(null)
      return
    }

    setValueError(null)
    setValueFilters({
      entityType: valueTypeInput,
      entityId: id,
      hippodrome: valueHippoInput.trim() || undefined,
      startDate: valueStartInput || undefined,
      endDate: valueEndInput || undefined,
      minEdge: minEdgeValue,
      limit: limitValue,
    })
  }

  const handleVolatilitySubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = volatilityIdInput.trim()

    if (!id) {
      setVolatilityError("Veuillez saisir un identifiant Aspiturf valide.")
      setVolatilityFilters(null)
      return
    }

    if (volatilityStartInput && volatilityEndInput && volatilityStartInput > volatilityEndInput) {
      setVolatilityError('La date de début doit précéder la date de fin.')
      setVolatilityFilters(null)
      return
    }

    setVolatilityError(null)
    setVolatilityFilters({
      entityType: volatilityTypeInput,
      entityId: id,
      hippodrome: volatilityHippoInput.trim() || undefined,
      startDate: volatilityStartInput || undefined,
      endDate: volatilityEndInput || undefined,
    })
  }

  const handleEfficiencySubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = efficiencyIdInput.trim()

    if (!id) {
      setEfficiencyError("Veuillez saisir un identifiant Aspiturf valide.")
      setEfficiencyFilters(null)
      return
    }

    if (efficiencyStartInput && efficiencyEndInput && efficiencyStartInput > efficiencyEndInput) {
      setEfficiencyError('La date de début doit précéder la date de fin.')
      setEfficiencyFilters(null)
      return
    }

    setEfficiencyError(null)
    setEfficiencyFilters({
      entityType: efficiencyTypeInput,
      entityId: id,
      hippodrome: efficiencyHippoInput.trim() || undefined,
      startDate: efficiencyStartInput || undefined,
      endDate: efficiencyEndInput || undefined,
    })
  }

  const handleWorkloadSubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = workloadIdInput.trim()

    if (!id) {
      setWorkloadError("Veuillez saisir un identifiant Aspiturf valide.")
      setWorkloadFilters(null)
      return
    }

    if (workloadStartInput && workloadEndInput && workloadStartInput > workloadEndInput) {
      setWorkloadError('La date de début doit précéder la date de fin.')
      setWorkloadFilters(null)
      return
    }

    setWorkloadError(null)
    setWorkloadFilters({
      entityType: workloadTypeInput,
      entityId: id,
      hippodrome: workloadHippoInput.trim() || undefined,
      startDate: workloadStartInput || undefined,
      endDate: workloadEndInput || undefined,
    })
  }

  const handleMomentumSubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = momentumIdInput.trim()
    const windowValue = Number(momentumWindowInput.trim() || '5')
    const baselineRaw = momentumBaselineInput.trim()
    const baselineValue = baselineRaw ? Number(baselineRaw) : windowValue

    if (!id) {
      setMomentumError("Veuillez saisir un identifiant Aspiturf valide.")
      setMomentumFilters(null)
      return
    }

    if (Number.isNaN(windowValue) || windowValue < 1 || windowValue > 50) {
      setMomentumError('La fenêtre récente doit être comprise entre 1 et 50 courses.')
      setMomentumFilters(null)
      return
    }

    if (baselineRaw && (Number.isNaN(baselineValue) || baselineValue < 1 || baselineValue > 50)) {
      setMomentumError('La fenêtre de référence doit être comprise entre 1 et 50 courses.')
      setMomentumFilters(null)
      return
    }

    if (momentumStartInput && momentumEndInput && momentumStartInput > momentumEndInput) {
      setMomentumError('La date de début doit précéder la date de fin.')
      setMomentumFilters(null)
      return
    }

    setMomentumError(null)
    setMomentumFilters({
      entityType: momentumTypeInput,
      entityId: id,
      hippodrome: momentumHippoInput.trim() || undefined,
      startDate: momentumStartInput || undefined,
      endDate: momentumEndInput || undefined,
      window: windowValue,
      baselineWindow: baselineValue,
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

  const handleCalendarSubmit = (event: FormEvent) => {
    event.preventDefault()
    const id = calendarIdInput.trim()

    if (!id) {
      setCalendarError("Veuillez saisir un identifiant Aspiturf valide.")
      setCalendarFilters(null)
      return
    }

    if (calendarStartInput && calendarEndInput && calendarStartInput > calendarEndInput) {
      setCalendarError('La date de début doit précéder la date de fin.')
      setCalendarFilters(null)
      return
    }

    setCalendarError(null)
    setCalendarFilters({
      entityType: calendarTypeInput,
      entityId: id,
      hippodrome: calendarHippoInput.trim() || undefined,
      startDate: calendarStartInput || undefined,
      endDate: calendarEndInput || undefined,
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
      title="Profil de volatilité"
      description="Évaluez la régularité d'un cheval, d'un jockey ou d'un entraîneur via l'écart-type des positions et des cotes."
    >
      <form
        onSubmit={handleVolatilitySubmit}
        className="grid gap-4 md:grid-cols-[repeat(6,minmax(0,1fr)),auto]"
      >
        <select
          value={volatilityTypeInput}
          onChange={(event) => setVolatilityTypeInput(event.target.value as TrendEntityType)}
          className="input"
        >
          <option value="horse">Cheval</option>
          <option value="jockey">Jockey</option>
          <option value="trainer">Entraîneur</option>
        </select>
        <input
          value={volatilityIdInput}
          onChange={(event) => setVolatilityIdInput(event.target.value)}
          placeholder="Identifiant Aspiturf (id)"
          className="input"
        />
        <input
          value={volatilityHippoInput}
          onChange={(event) => setVolatilityHippoInput(event.target.value)}
          placeholder="Filtrer par hippodrome (optionnel)"
          className="input"
        />
        <input
          type="date"
          value={volatilityStartInput}
          onChange={(event) => setVolatilityStartInput(event.target.value)}
          className="input"
        />
        <input
          type="date"
          value={volatilityEndInput}
          onChange={(event) => setVolatilityEndInput(event.target.value)}
          className="input"
        />
        <button type="submit" className="btn btn-primary">
          Analyser
        </button>
      </form>
      {volatilityError && <p className="text-sm text-red-600">{volatilityError}</p>}
      {volatilityQuery.isPending && (
        <p className="text-sm text-gray-500">Calcul de la volatilité en cours…</p>
      )}
    {volatilityQuery.isError && (
      <p className="text-sm text-red-600">Erreur: {(volatilityQuery.error as Error).message}</p>
    )}
    {volatilityQuery.data && <VolatilityPanel data={volatilityQuery.data} />}
  </SectionCard>

  <SectionCard
    title="Efficacité des résultats"
    description="Mesurez l'écart entre les probabilités implicites des cotes et les performances réelles d'une entité."
  >
    <form
      onSubmit={handleEfficiencySubmit}
      className="grid gap-4 md:grid-cols-[repeat(6,minmax(0,1fr)),auto]"
    >
      <select
        value={efficiencyTypeInput}
        onChange={(event) => setEfficiencyTypeInput(event.target.value as TrendEntityType)}
        className="input"
      >
        <option value="horse">Cheval</option>
        <option value="jockey">Jockey</option>
        <option value="trainer">Entraîneur</option>
      </select>
      <input
        value={efficiencyIdInput}
        onChange={(event) => setEfficiencyIdInput(event.target.value)}
        placeholder="Identifiant Aspiturf (id)"
        className="input"
      />
      <input
        value={efficiencyHippoInput}
        onChange={(event) => setEfficiencyHippoInput(event.target.value)}
        placeholder="Filtrer par hippodrome (optionnel)"
        className="input"
      />
      <input
        type="date"
        value={efficiencyStartInput}
        onChange={(event) => setEfficiencyStartInput(event.target.value)}
        className="input"
      />
      <input
        type="date"
        value={efficiencyEndInput}
        onChange={(event) => setEfficiencyEndInput(event.target.value)}
        className="input"
      />
      <button type="submit" className="btn btn-primary">
        Comparer
      </button>
    </form>
    {efficiencyError && <p className="text-sm text-red-600">{efficiencyError}</p>}
    {efficiencyQuery.isPending && (
      <p className="text-sm text-gray-500">Calcul du différentiel attendu/observé…</p>
    )}
    {efficiencyQuery.isError && (
      <p className="text-sm text-red-600">Erreur: {(efficiencyQuery.error as Error).message}</p>
    )}
    {efficiencyQuery.data && <EfficiencyPanel data={efficiencyQuery.data} />}
  </SectionCard>

  <SectionCard
    title="Charge de travail"
    description="Évaluez les temps de repos et la fréquence d'engagement d'un cheval, d'un jockey ou d'un entraîneur."
  >
    <form
      onSubmit={handleWorkloadSubmit}
      className="grid gap-4 md:grid-cols-[repeat(6,minmax(0,1fr)),auto]"
    >
      <select
        value={workloadTypeInput}
        onChange={(event) => setWorkloadTypeInput(event.target.value as TrendEntityType)}
        className="input"
      >
        <option value="horse">Cheval</option>
        <option value="jockey">Jockey</option>
        <option value="trainer">Entraîneur</option>
      </select>
      <input
        value={workloadIdInput}
        onChange={(event) => setWorkloadIdInput(event.target.value)}
        placeholder="Identifiant Aspiturf (id)"
        className="input"
      />
      <input
        value={workloadHippoInput}
        onChange={(event) => setWorkloadHippoInput(event.target.value)}
        placeholder="Filtrer par hippodrome (optionnel)"
        className="input"
      />
      <input
        type="date"
        value={workloadStartInput}
        onChange={(event) => setWorkloadStartInput(event.target.value)}
        className="input"
      />
      <input
        type="date"
        value={workloadEndInput}
        onChange={(event) => setWorkloadEndInput(event.target.value)}
        className="input"
      />
      <button type="submit" className="btn btn-primary">
        Diagnostiquer
      </button>
    </form>
    {workloadError && <p className="text-sm text-red-600">{workloadError}</p>}
    {workloadQuery.isPending && (
      <p className="text-sm text-gray-500">Calcul des rythmes de participation en cours…</p>
    )}
    {workloadQuery.isError && (
      <p className="text-sm text-red-600">Erreur: {(workloadQuery.error as Error).message}</p>
    )}
    {workloadQuery.data && <WorkloadPanel data={workloadQuery.data} />}
  </SectionCard>

  <SectionCard
    title="Momentum récent"
    description="Comparez la dynamique actuelle d'un cheval, jockey ou entraîneur à sa période précédente pour détecter un regain ou une baisse de forme."
  >
      <form
        onSubmit={handleMomentumSubmit}
        className="grid gap-4 md:grid-cols-[repeat(7,minmax(0,1fr)),auto]"
      >
        <select
          value={momentumTypeInput}
          onChange={(event) => setMomentumTypeInput(event.target.value as TrendEntityType)}
          className="input"
        >
          <option value="horse">Cheval</option>
          <option value="jockey">Jockey</option>
          <option value="trainer">Entraîneur</option>
        </select>
        <input
          value={momentumIdInput}
          onChange={(event) => setMomentumIdInput(event.target.value)}
          placeholder="Identifiant Aspiturf (id)"
          className="input"
        />
        <input
          value={momentumHippoInput}
          onChange={(event) => setMomentumHippoInput(event.target.value)}
          placeholder="Filtrer par hippodrome (optionnel)"
          className="input"
        />
        <input
          type="date"
          value={momentumStartInput}
          onChange={(event) => setMomentumStartInput(event.target.value)}
          className="input"
        />
        <input
          type="date"
          value={momentumEndInput}
          onChange={(event) => setMomentumEndInput(event.target.value)}
          className="input"
        />
        <input
          type="number"
          min={1}
          max={50}
          value={momentumWindowInput}
          onChange={(event) => setMomentumWindowInput(event.target.value)}
          placeholder="Fenêtre récente"
          className="input"
        />
        <input
          type="number"
          min={1}
          max={50}
          value={momentumBaselineInput}
          onChange={(event) => setMomentumBaselineInput(event.target.value)}
          placeholder="Fenêtre référence"
          className="input"
        />
        <button type="submit" className="btn btn-primary">
          Comparer
        </button>
      </form>
      {momentumError && <p className="text-sm text-red-600">{momentumError}</p>}
      {momentumQuery.isPending && (
        <p className="text-sm text-gray-500">Analyse du momentum en cours…</p>
      )}
      {momentumQuery.isError && (
        <p className="text-sm text-red-600">Erreur: {(momentumQuery.error as Error).message}</p>
      )}
      {momentumQuery.data && <MomentumPanel data={momentumQuery.data} />}
    </SectionCard>

    <SectionCard
      title="Opportunités value bet"
      description="Repérez les courses où la cote observée offre un edge positif par rapport aux estimations Aspiturf."
    >
      <form
        onSubmit={handleValueSubmit}
        className="grid gap-4 md:grid-cols-[repeat(7,minmax(0,1fr)),auto]"
      >
        <select
          value={valueTypeInput}
          onChange={(event) => setValueTypeInput(event.target.value as TrendEntityType)}
          className="input"
        >
          <option value="horse">Cheval</option>
          <option value="jockey">Jockey</option>
          <option value="trainer">Entraîneur</option>
        </select>
        <input
          value={valueIdInput}
          onChange={(event) => setValueIdInput(event.target.value)}
          placeholder="Identifiant Aspiturf (id)"
          className="input"
        />
        <input
          value={valueHippoInput}
          onChange={(event) => setValueHippoInput(event.target.value)}
          placeholder="Filtrer par hippodrome (optionnel)"
          className="input"
        />
        <input
          type="date"
          value={valueStartInput}
          onChange={(event) => setValueStartInput(event.target.value)}
          className="input"
        />
        <input
          type="date"
          value={valueEndInput}
          onChange={(event) => setValueEndInput(event.target.value)}
          className="input"
        />
        <input
          type="number"
          min={0}
          step={0.1}
          value={valueMinEdgeInput}
          onChange={(event) => setValueMinEdgeInput(event.target.value)}
          placeholder="Seuil d'écart"
          className="input"
        />
        <input
          type="number"
          min={5}
          max={100}
          value={valueLimitInput}
          onChange={(event) => setValueLimitInput(event.target.value)}
          placeholder="Courses max"
          className="input"
        />
        <button type="submit" className="btn btn-primary">
          Calculer
        </button>
      </form>
      {valueError && <p className="text-sm text-red-600">{valueError}</p>}
      {valueQuery.isPending && (
        <p className="text-sm text-gray-500">Recherche des meilleures opportunités…</p>
      )}
      {valueQuery.isError && (
        <p className="text-sm text-red-600">Erreur: {(valueQuery.error as Error).message}</p>
      )}
      {valueQuery.data && <ValuePanel data={valueQuery.data} />}
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
      title="Calendrier des performances"
      description="Visualisez l'enchaînement des résultats jour par jour pour un cheval, un jockey ou un entraîneur."
    >
      <form
        onSubmit={handleCalendarSubmit}
        className="grid gap-4 md:grid-cols-[repeat(5,minmax(0,1fr)),auto]"
      >
        <select
          value={calendarTypeInput}
          onChange={(event) => setCalendarTypeInput(event.target.value as TrendEntityType)}
          className="input"
        >
          <option value="horse">Cheval</option>
          <option value="jockey">Jockey</option>
          <option value="trainer">Entraîneur</option>
        </select>
        <input
          value={calendarIdInput}
          onChange={(event) => setCalendarIdInput(event.target.value)}
          placeholder="Identifiant Aspiturf (id)"
          className="input"
        />
        <input
          value={calendarHippoInput}
          onChange={(event) => setCalendarHippoInput(event.target.value)}
          placeholder="Filtrer par hippodrome (optionnel)"
          className="input"
        />
        <input
          type="date"
          value={calendarStartInput}
          onChange={(event) => setCalendarStartInput(event.target.value)}
          className="input"
        />
        <input
          type="date"
          value={calendarEndInput}
          onChange={(event) => setCalendarEndInput(event.target.value)}
          className="input"
        />
        <button type="submit" className="btn btn-primary">
          Générer
        </button>
      </form>

      {calendarError && <p className="text-sm text-red-600">{calendarError}</p>}

      {calendarQuery.isPending && (
        <p className="text-sm text-gray-500">Agrégation des résultats par journée…</p>
      )}

      {calendarQuery.isError && (
        <p className="text-sm text-red-600">Erreur: {(calendarQuery.error as Error).message}</p>
      )}

      {calendarQuery.data && <CalendarPanel data={calendarQuery.data} />}
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
