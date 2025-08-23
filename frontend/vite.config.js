import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    manifest: true,   // emit dist/manifest.json for backend to resolve entry
  },
})
