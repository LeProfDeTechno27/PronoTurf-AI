import apiClient from './api'
import type {
  HorseAnalyticsResponse,
  JockeyAnalyticsResponse,
  TrainerAnalyticsResponse,
  CoupleAnalyticsResponse,
  CourseAnalyticsResponse,
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
}

export default analyticsService
