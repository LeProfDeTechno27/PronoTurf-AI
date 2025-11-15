export type UserRole = 'admin' | 'subscriber' | 'guest'

export type BankrollStrategy = 'kelly' | 'flat' | 'martingale'

export interface RegisterPayload {
  email: string
  password: string
  first_name?: string
  last_name?: string
  role?: UserRole
}

export interface LoginPayload {
  email: string
  password: string
}

export interface LoginResponse {
  access_token: string
  refresh_token: string
  token_type: 'bearer'
}

export interface UserRead {
  user_id: number
  email: string
  first_name?: string | null
  last_name?: string | null
  role: UserRole
  telegram_id?: string | null
  profile_picture_url?: string | null
  initial_bankroll: string
  current_bankroll: string
  preferred_strategy: BankrollStrategy
  created_at: string
  last_login?: string | null
  is_active: boolean
}

export interface RegisterResponse extends UserRead {}
