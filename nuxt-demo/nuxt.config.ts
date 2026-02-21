// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  ssr: false,
  // build: {
  //   transpile: ['booknlp-ts'],
  // }
  vite: {
    // build: {
    //   assetsInlineLimit: 50 * 1024 * 1024, // 50MB, default is 4KB
    // },
    worker: {
      format: 'es', // ensure ES module output for workers
      // rollupOptions: {
      //   output: {
      //     manualChunks: undefined, // disables code splitting for workers
      //   },
      // },
    },
  }
})
