import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/predict': 'http://127.0.0.1:5000',
      '/uploads': 'http://127.0.0.1:5000',
      '/history': 'http://127.0.0.1:5000',
      '/analytics': 'http://127.0.0.1:5000',
    }
  }
})
