import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// В Docker статика собирается в dist и отдаётся nginx, а /api проксирует Caddy.
// Локальный `npm run dev` проксирует /api на адрес из VITE_DEV_PROXY (по умолчанию — Caddy на :80).
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_DEV_PROXY || 'http://localhost',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
