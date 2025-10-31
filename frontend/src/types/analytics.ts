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

export interface PerformanceSummary {
  sample_size?: number | null
  wins?: number | null
  places?: number | null
  win_rate?: number | null
  place_rate?: number | null
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
