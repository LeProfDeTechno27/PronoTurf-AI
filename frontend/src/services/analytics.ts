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
  PerformanceDistributionResponse,
  DistributionDimension,
  AnalyticsFormResponse,
  AnalyticsComparisonResponse,
  AnalyticsCalendarResponse,
  AnalyticsValueResponse,
  AnalyticsVolatilityResponse,
  AnalyticsEfficiencyResponse,
  AnalyticsWorkloadResponse,
  AnalyticsMomentumResponse,
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

  async getPerformanceDistribution(params: {
    entityType: TrendEntityType
    entityId: string
    dimension: DistributionDimension
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
    distanceStep?: number
  }) {
    const response = await apiClient.get<PerformanceDistributionResponse>('/analytics/distributions', {
      params: {
        entity_type: params.entityType,
        entity_id: params.entityId,
        dimension: params.dimension,
        ...(params.hippodrome ? { hippodrome: params.hippodrome } : {}),
        ...(params.startDate ? { start_date: params.startDate } : {}),
        ...(params.endDate ? { end_date: params.endDate } : {}),
        ...(params.distanceStep ? { distance_step: params.distanceStep } : {}),
      },
    })

    return response.data
  },

  async getPerformanceCalendar(params: {
    entityType: TrendEntityType
    entityId: string
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
  }) {
    const response = await apiClient.get<AnalyticsCalendarResponse>('/analytics/calendar', {
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

  async getValueOpportunities(params: {
    entityType: TrendEntityType
    entityId: string
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
    minEdge?: number
    limit?: number
  }) {
    const response = await apiClient.get<AnalyticsValueResponse>('/analytics/value', {
      params: {
        entity_type: params.entityType,
        entity_id: params.entityId,
        ...(params.hippodrome ? { hippodrome: params.hippodrome } : {}),
        ...(params.startDate ? { start_date: params.startDate } : {}),
        ...(params.endDate ? { end_date: params.endDate } : {}),
        ...(params.minEdge !== undefined ? { min_edge: params.minEdge } : {}),
        ...(params.limit ? { limit: params.limit } : {}),
      },
    })

    return response.data
  },

  async getVolatilityProfile(params: {
    entityType: TrendEntityType
    entityId: string
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
  }) {
    const response = await apiClient.get<AnalyticsVolatilityResponse>('/analytics/volatility', {
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

  /**
   * Compare les victoires observées avec les probabilités implicites issues des cotes.
   */
  async getEfficiencyProfile(params: {
    entityType: TrendEntityType
    entityId: string
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
  }) {
    const response = await apiClient.get<AnalyticsEfficiencyResponse>('/analytics/efficiency', {
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

  /**
   * Récupère la charge de travail et les temps de repos d'une entité Aspiturf.
   */
  async getWorkloadProfile(params: {
    entityType: TrendEntityType
    entityId: string
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
  }) {
    const response = await apiClient.get<AnalyticsWorkloadResponse>('/analytics/workload', {
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

  async getMomentumProfile(params: {
    entityType: TrendEntityType
    entityId: string
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
    window?: number
    baselineWindow?: number
  }) {
    const response = await apiClient.get<AnalyticsMomentumResponse>('/analytics/momentum', {
      params: {
        entity_type: params.entityType,
        entity_id: params.entityId,
        ...(params.hippodrome ? { hippodrome: params.hippodrome } : {}),
        ...(params.startDate ? { start_date: params.startDate } : {}),
        ...(params.endDate ? { end_date: params.endDate } : {}),
        ...(params.window ? { window: params.window } : {}),
        ...(params.baselineWindow ? { baseline_window: params.baselineWindow } : {}),
      },
    })

    return response.data
  },

  async getComparisons(params: {
    entityType: TrendEntityType
    entityIds: string[]
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
  }) {
    const queryParams = new URLSearchParams()
    queryParams.set('type', params.entityType)

    params.entityIds.forEach((id) => {
      if (id) {
        queryParams.append('ids', id)
      }
    })

    if (params.hippodrome) {
      queryParams.set('hippodrome', params.hippodrome)
    }
    if (params.startDate) {
      queryParams.set('start_date', params.startDate)
    }
    if (params.endDate) {
      queryParams.set('end_date', params.endDate)
    }

    const response = await apiClient.get<AnalyticsComparisonResponse>('/analytics/comparisons', {
      params: queryParams,
    })

    return response.data
  },

  async getFormSnapshot(params: {
    entityType: TrendEntityType
    entityId: string
    window?: number
    hippodrome?: Nullable<string>
    startDate?: Nullable<string>
    endDate?: Nullable<string>
  }) {
    const response = await apiClient.get<AnalyticsFormResponse>('/analytics/form', {
      params: {
        entity_type: params.entityType,
        entity_id: params.entityId,
        ...(params.window ? { window: params.window } : {}),
        ...(params.hippodrome ? { hippodrome: params.hippodrome } : {}),
        ...(params.startDate ? { start_date: params.startDate } : {}),
        ...(params.endDate ? { end_date: params.endDate } : {}),
      },
    })

    return response.data
  },
}

export default analyticsService
