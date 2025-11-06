// Copyright (c) 2025 PronoTurf AI. All rights reserved.
// This source code is proprietary and confidential.
// Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

export interface AnalyticsMetadata {
  hippodrome_filter?: string | null
  date_start?: string | null
  date_end?: string | null
}

export interface PerformanceBreakdown {
  label: string
  total: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
}

export interface RecentRace {
  date?: string | null
  hippodrome?: string | null
  course_number?: number | null
  distance?: number | null
  final_position?: number | null
  odds?: number | null
  is_win: boolean
  is_podium: boolean
}

export interface FormRace extends RecentRace {
  score: number
}

export interface PerformanceSummary {
  sample_size?: number | null
  wins?: number | null
  places?: number | null
  win_rate?: number | null
  place_rate?: number | null
}

export interface LeaderboardEntry {
  entity_id: string
  label: string
  sample_size: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  last_seen?: string | null
}

export interface HeadToHeadBreakdown {
  opponent_id: string
  meetings: number
  ahead: number
  behind: number
  ties: number
}

export interface ComparisonEntitySummary {
  entity_id: string
  label?: string | null
  sample_size: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  best_finish?: number | null
  last_seen?: string | null
  head_to_head: HeadToHeadBreakdown[]
}

export interface AnalyticsComparisonResponse {
  entity_type: 'horse' | 'jockey' | 'trainer'
  shared_races: number
  entities: ComparisonEntitySummary[]
  metadata: AnalyticsMetadata
}

export type TrendEntityType = 'horse' | 'jockey' | 'trainer'

export type TrendGranularity = 'week' | 'month'

export interface PerformanceTrendPoint {
  period_start: string
  period_end: string
  label: string
  races: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  average_odds?: number | null
}

export interface PerformanceTrendResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  granularity: TrendGranularity
  metadata: AnalyticsMetadata
  points: PerformanceTrendPoint[]
}

export type DistributionDimension = 'distance' | 'draw' | 'hippodrome' | 'discipline'

export interface DistributionBucket {
  label: string
  races: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  average_odds?: number | null
}

export interface PerformanceDistributionResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  dimension: DistributionDimension
  metadata: AnalyticsMetadata
  buckets: DistributionBucket[]
}

export type SeasonalityGranularity = 'month' | 'weekday'

export interface SeasonalityBucket {
  key: string
  label: string
  races: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  average_odds?: number | null
}

export interface AnalyticsSeasonalityResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  granularity: SeasonalityGranularity
  metadata: AnalyticsMetadata
  total_races: number
  total_wins: number
  total_podiums: number
  buckets: SeasonalityBucket[]
}

export interface CalendarRaceDetail {
  hippodrome?: string | null
  course_number?: number | null
  distance?: number | null
  final_position?: number | null
  odds?: number | null
}

export interface CalendarDaySummary {
  date: string
  hippodromes: string[]
  races: number
  wins: number
  podiums: number
  average_finish?: number | null
  average_odds?: number | null
  race_details: CalendarRaceDetail[]
}

export interface AnalyticsCalendarResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  metadata: AnalyticsMetadata
  total_races: number
  total_wins: number
  total_podiums: number
  days: CalendarDaySummary[]
}

export interface AnalyticsFormResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  window: number
  metadata: AnalyticsMetadata
  total_races: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  average_odds?: number | null
  median_odds?: number | null
  best_position?: number | null
  consistency_index?: number | null
  form_score?: number | null
  races: FormRace[]
}

export interface ValueOpportunitySample {
  date?: string | null
  hippodrome?: string | null
  course_number?: number | null
  distance?: number | null
  final_position?: number | null
  odds_actual?: number | null
  odds_implied?: number | null
  edge?: number | null
  is_win?: boolean | null
  profit?: number | null
}

export interface AnalyticsValueResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  metadata: AnalyticsMetadata
  sample_size: number
  wins: number
  win_rate?: number | null
  positive_edges: number
  negative_edges: number
  average_edge?: number | null
  median_edge?: number | null
  average_odds?: number | null
  median_odds?: number | null
  stake_count: number
  profit?: number | null
  roi?: number | null
  hippodromes: string[]
  samples: ValueOpportunitySample[]
}

export interface VolatilityRaceSample {
  date?: string | null
  hippodrome?: string | null
  course_number?: number | null
  distance?: number | null
  final_position?: number | null
  odds_actual?: number | null
  odds_implied?: number | null
  edge?: number | null
  is_win: boolean
  is_podium: boolean
}

export interface VolatilityMetrics {
  sample_size: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  position_std_dev?: number | null
  average_odds?: number | null
  odds_std_dev?: number | null
  average_edge?: number | null
  consistency_index?: number | null
}

export interface AnalyticsVolatilityResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  metadata: AnalyticsMetadata
  metrics: VolatilityMetrics
  races: VolatilityRaceSample[]
}

export interface OddsBucketMetrics {
  label: string
  sample_size: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  average_odds?: number | null
  profit?: number | null
  roi?: number | null
}

export interface AnalyticsOddsResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  metadata: AnalyticsMetadata
  total_races: number
  races_with_odds: number
  races_without_odds: number
  buckets: OddsBucketMetrics[]
  overall: OddsBucketMetrics
}

export interface EfficiencySample {
  date?: string | null
  hippodrome?: string | null
  course_number?: number | null
  odds?: number | null
  expected_win_probability?: number | null
  expected_podium_probability?: number | null
  finish_position?: number | null
  is_win: boolean
  is_podium: boolean
  edge?: number | null
}

export interface EfficiencyMetrics {
  sample_size: number
  wins: number
  expected_wins?: number | null
  win_delta?: number | null
  podiums: number
  expected_podiums?: number | null
  podium_delta?: number | null
  average_odds?: number | null
  average_expected_win_probability?: number | null
  stake_count: number
  profit?: number | null
  roi?: number | null
}

export interface AnalyticsEfficiencyResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  metadata: AnalyticsMetadata
  metrics: EfficiencyMetrics
  samples: EfficiencySample[]
}

export interface WorkloadTimelineEntry {
  date?: string | null
  hippodrome?: string | null
  course_number?: number | null
  distance?: number | null
  final_position?: number | null
  rest_days?: number | null
  odds?: number | null
  is_win: boolean
  is_podium: boolean
}

export interface WorkloadSummary {
  sample_size: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_rest_days?: number | null
  median_rest_days?: number | null
  shortest_rest_days?: number | null
  longest_rest_days?: number | null
  races_last_30_days: number
  races_last_90_days: number
  average_monthly_races?: number | null
  rest_distribution: Record<string, number>
}

export interface AnalyticsWorkloadResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  metadata: AnalyticsMetadata
  summary: WorkloadSummary
  timeline: WorkloadTimelineEntry[]
}

export type ProgressionTrend =
  | 'improvement'
  | 'decline'
  | 'stable'
  | 'initial'
  | 'unknown'

export interface ProgressionRace {
  date?: string | null
  hippodrome?: string | null
  course_number?: number | null
  distance?: number | null
  final_position?: number | null
  previous_position?: number | null
  change?: number | null
  trend: ProgressionTrend
}

export interface ProgressionSummary {
  races: number
  improvements: number
  declines: number
  stable: number
  average_change?: number | null
  best_change?: number | null
  worst_change?: number | null
  longest_improvement_streak: number
  longest_decline_streak: number
  net_progress?: number | null
}

export interface AnalyticsProgressionResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  metadata: AnalyticsMetadata
  summary: ProgressionSummary
  races: ProgressionRace[]
}

export interface MomentumSlice {
  label: string
  start_date?: string | null
  end_date?: string | null
  race_count: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  average_odds?: number | null
  roi?: number | null
  races: RecentRace[]
}

export interface MomentumShift {
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  roi?: number | null
}

export interface AnalyticsMomentumResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  metadata: AnalyticsMetadata
  recent_window: MomentumSlice
  reference_window?: MomentumSlice | null
  deltas: MomentumShift
}

export interface PerformanceStreak {
  type: 'win' | 'podium'
  length: number
  start_date?: string | null
  end_date?: string | null
  is_active: boolean
}

export interface AnalyticsStreakResponse {
  entity_type: TrendEntityType
  entity_id: string
  entity_label?: string | null
  metadata: AnalyticsMetadata
  total_races: number
  wins: number
  podiums: number
  best_win_streak?: PerformanceStreak | null
  best_podium_streak?: PerformanceStreak | null
  current_win_streak?: PerformanceStreak | null
  current_podium_streak?: PerformanceStreak | null
  streak_history: PerformanceStreak[]
}

export interface HorseAnalyticsResponse {
  horse_id: string
  horse_name?: string | null
  sample_size: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  average_odds?: number | null
  recent_results: RecentRace[]
  hippodrome_breakdown: PerformanceBreakdown[]
  distance_breakdown: PerformanceBreakdown[]
  metadata: AnalyticsMetadata
}

export interface JockeyAnalyticsResponse {
  jockey_id: string
  jockey_name?: string | null
  sample_size: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  recent_results: RecentRace[]
  horse_breakdown: PerformanceBreakdown[]
  hippodrome_breakdown: PerformanceBreakdown[]
  metadata: AnalyticsMetadata
}

export interface TrainerAnalyticsResponse {
  trainer_id: string
  trainer_name?: string | null
  sample_size: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  recent_results: RecentRace[]
  horse_breakdown: PerformanceBreakdown[]
  hippodrome_breakdown: PerformanceBreakdown[]
  metadata: AnalyticsMetadata
}

export interface CoupleAnalyticsResponse {
  horse_id: string
  jockey_id: string
  horse_name?: string | null
  jockey_name?: string | null
  sample_size: number
  wins: number
  podiums: number
  win_rate?: number | null
  podium_rate?: number | null
  average_finish?: number | null
  recent_results: RecentRace[]
  metadata: AnalyticsMetadata
}

export interface PartantInsight {
  numero?: number | null
  horse_id?: string | null
  horse_name?: string | null
  jockey_id?: string | null
  jockey_name?: string | null
  trainer_id?: string | null
  trainer_name?: string | null
  odds?: number | null
  probable_odds?: number | null
  recent_form?: string | null
  days_since_last_race?: number | null
  handicap_value?: number | null
  jockey_stats?: PerformanceSummary | null
  trainer_stats?: PerformanceSummary | null
  horse_stats?: PerformanceSummary | null
  couple_stats?: PerformanceSummary | null
}

export interface CourseAnalyticsResponse {
  date: string
  hippodrome: string
  course_number: number
  distance?: number | null
  discipline?: string | null
  allocation?: number | null
  currency?: string | null
  partants: PartantInsight[]
  metadata: AnalyticsMetadata
}

export interface AnalyticsInsightsResponse {
  metadata: AnalyticsMetadata
  top_horses: LeaderboardEntry[]
  top_jockeys: LeaderboardEntry[]
  top_trainers: LeaderboardEntry[]
}

export type AnalyticsSearchType = 'horse' | 'jockey' | 'trainer' | 'hippodrome'

export interface AnalyticsSearchMetadata {
  total_races?: number | null
  hippodromes?: string[]
  last_seen?: string | null
  course_count?: number | null
  last_meeting?: string | null
  disciplines?: string[]
}

export interface AnalyticsSearchResult {
  type: AnalyticsSearchType
  id: string
  label: string
  metadata: AnalyticsSearchMetadata
}