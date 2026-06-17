import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
})

export const get = (endpoint, config) => api.get(endpoint, config).then((r) => r.data)

export const post = (endpoint, data, config) => api.post(endpoint, data, config).then((r) => r.data)
