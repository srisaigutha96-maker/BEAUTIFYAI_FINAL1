import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      // Forward /api/* requests to the BeautifyAI backend
      '/api': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: true,
      },
      '/beautify': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: true,
      },
      '/chat': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: true,
      },
    },
  },
})
