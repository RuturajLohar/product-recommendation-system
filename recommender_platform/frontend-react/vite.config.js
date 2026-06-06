import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

const proxyTarget = (env) =>
  env.VITE_PROXY_API?.trim() || 'http://127.0.0.1:8000'

const apiProxy = (env) => ({
  '/api': {
    target: proxyTarget(env),
    changeOrigin: true,
  },
})

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react()],
    server: {
      proxy: apiProxy(env),
    },
    preview: {
      proxy: apiProxy(env),
    },
  }
})
