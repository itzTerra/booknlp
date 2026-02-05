import { defineConfig } from 'vite';
import { resolve } from 'path';
import dts from 'vite-plugin-dts';

export default defineConfig({
  plugins: [
    dts({
      insertTypesEntry: true,
      include: ['src/**/*'],
      exclude: ['src/**/*.test.ts'],
    }),
  ],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'BookNLP',
      formats: ['es', 'umd'],
      fileName: (format) => `booknlp.${format === 'es' ? 'js' : 'umd.cjs'}`,
    },
    rollupOptions: {
      external: [],
      output: {
        globals: {},
        assetFileNames: (assetInfo) => {
          if (assetInfo.name === 'style.css') return 'booknlp.css';
          return assetInfo.name ?? 'asset';
        },
      },
    },
    sourcemap: true,
    minify: 'esbuild',
  },
  publicDir: resolve(__dirname, '../booknlp/english/data'),
  resolve: {
    alias: {
      types: resolve(__dirname, 'src/types'),
      validation: resolve(__dirname, 'src/validation'),
      preprocessing: resolve(__dirname, 'src/preprocessing'),
      'sequence-postprocessor': resolve(__dirname, 'src/sequence-postprocessor'),
      'batch-processor': resolve(__dirname, 'src/batch-processor'),
      'tagger-controller': resolve(__dirname, 'src/tagger-controller'),
      'crf-decoder': resolve(__dirname, 'src/crf-decoder'),
      'advanced-postprocessor': resolve(__dirname, 'src/advanced-postprocessor'),
      'entity-tagger': resolve(__dirname, 'src/entity-tagger'),
      'english-booknlp': resolve(__dirname, 'src/english-booknlp'),
    },
  },
});
