import apiClient from './api'
import type {
  HorseAnalyticsResponse,
  JockeyAnalyticsResponse,
  TrainerAnalyticsResponse,
  CoupleAnalyticsResponse,
  CourseAnalyticsResponse,
  AnalyticsSearchResult,
  AnalyticsSearchType,
  AnalyticsInsightsResponse,
  AnalyticsStreakResponse,
  PerformanceTrendResponse,
  TrendEntityType,
  TrendGranularity,
} from '../types/analytics'

type Nullable<T> = T | null | undefined

export const analyticsService = {
  async getHorseAnalytics(horseId: string, hippodrome?: Nullable<string>) {
    const response = await apiClient.get<HorseAnalyticsResponse>(`/analytics/horse/${horseId}`, {
      params: hippodrome ? { hippodrome } : undefined,
    })

    return response.data
  },

  async getJockeyAnalytics(jockeyId: string, hippodrome?: Nullable<string>) {
    const response = await apiClient.get<JockeyAnalyticsResponse>(`/analytics/jockey/${jockeyId}`, {
      params: hippodrome ? { hippodrome } : undefined,
    })

    return response.data
  },

  async getTrainerAnalytics(trainerId: string, hippodrome?: Nullable<string>) {
    const response = await apiClient.get<TrainerAnalyticsResponse>(`/analytics/trainer/${trainerId}`, {
      params: hippodrome ? { hippodrome } : undefined,
    })

    return response.data
  },

  async getCoupleAnalytics(
    horseId: string,
    jockeyId: string,
    hippodrome?: Nullable<string>,
  ) {
    const response = await apiClient.get<CoupleAnalyticsResponse>('/analytics/couple', {
      params: {
        horse_id: horseId,
        jockey_id: jockeyId,
        ...(hippodrome ? { hippodrome } : {}),
      },
    })

    return response.data
  },

  async getCourseAnalytics(
    courseDate: string,
    hippodrome: string,
    courseNumber: number,
  ) {
    const response = await apiClient.get<CourseAnalyticsResponse>('/analytics/course', {
      params: {
        course_date: courseDate,
        hippodrome,
        course_number: courseNumber,
      },
    })

    return response.data
  },

  async searchEntities(type: AnalyticsSearchType, query: string, limit = 10) {
    const response = await apiClient.get<AnalyticsSearchResult[]>('/analytics/search', {
      params: { type, query, limit },
    })

    return response.data
  },

  async getInsights(params?: {
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
    limit?: number
  }) {
    const response = await apiClient.get<AnalyticsInsightsResponse>('/analytics/insights', {
      params: {
        ...(params?.hippodrome ? { hippodrome: params.hippodrome } : {}),
        ...(params?.startDate ? { start_date: params.startDate } : {}),
        ...(params?.endDate ? { end_date: params.endDate } : {}),
        ...(params?.limit ? { limit: params.limit } : {}),
      },
    })

    return response.data
  },

  async getPerformanceTrend(params: {
    entityType: TrendEntityType
    entityId: string
    granularity?: TrendGranularity
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
  }) {
    const response = await apiClient.get<PerformanceTrendResponse>('/analytics/trends', {
      params: {
        entity_type: params.entityType,
        entity_id: params.entityId,
        ...(params.granularity ? { granularity: params.granularity } : {}),
        ...(params.hippodrome ? { hippodrome: params.hippodrome } : {}),
        ...(params.startDate ? { start_date: params.startDate } : {}),
        ...(params.endDate ? { end_date: params.endDate } : {}),
      },
    })

    return response.data
  },

  async getPerformanceStreaks(params: {
    entityType: TrendEntityType
    entityId: string
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
  }) {
    const response = await apiClient.get<AnalyticsStreakResponse>('/analytics/streaks', {
      params: {
        entity_type: params.entityType,
        entity_id: params.entityId,
        ...(params.hippodrome ? { hippodrome: params.hippodrome } : {}),
        ...(params.startDate ? { start_date: params.startDate } : {}),
        ...(params.endDate ? { end_date: params.endDate } : {}),
      },
    })

    return response.data
  },
}

export default analyticsService
