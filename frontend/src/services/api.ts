// Copyright (c) 2025 PronoTurf AI. All rights reserved.
// This source code is proprietary and confidential.
// Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000,
})

export default apiClient
