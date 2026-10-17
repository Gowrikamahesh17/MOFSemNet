import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Dev-time only: forwards /api/* to the FastAPI backend so the frontend never needs
    // to know the backend's host/port. Production serves both from the same origin (see
    // backend/app/main.py), so this proxy has no equivalent there.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8010',
        changeOrigin: true,
      },
    },
  },
})
