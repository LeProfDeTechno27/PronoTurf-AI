import apiClient from './api'
import type {
  LoginPayload,
  LoginResponse,
  RegisterPayload,
  RegisterResponse,
} from '../types/auth'

export async function registerUser(payload: RegisterPayload) {
  const response = await apiClient.post<RegisterResponse>('/auth/register', payload)
  return response.data
}

export async function loginUser(payload: LoginPayload) {
  const response = await apiClient.post<LoginResponse>('/auth/login', payload)
  return response.data
}
