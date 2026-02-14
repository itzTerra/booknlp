import type { ProgressCallback } from './types';

let cacheName = 'booknlp-resources-v1';

let originalFetch: typeof fetch | null = null;
let globalProgressCb: ProgressCallback | undefined;
const activeProgress = new Map<string, number>();

function emitOverallProgress() {
  if (!globalProgressCb) return;
  if (activeProgress.size === 0) {
    globalProgressCb(1);
    return;
  }
  // Simple average of active download progresses (0..1)
  let sum = 0;
  for (const v of activeProgress.values()) sum += v;
  const overall = sum / activeProgress.size;
  globalProgressCb(Math.min(1, Math.max(0, overall)));
}

export function installGlobalFetch(progressCb?: ProgressCallback, customCacheName?: string): void {
  if (customCacheName) {
    cacheName = customCacheName;
  }

  if ((globalThis as any).__booknlp_fetch_installed) {
    globalProgressCb = progressCb || globalProgressCb;
    return;
  }

  originalFetch = globalThis.fetch.bind(globalThis);
  globalProgressCb = progressCb;

  const wrappedFetch: typeof fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    try {
      const cache = await caches.open(cacheName);
      const cached = await cache.match(url);
      if (cached) {
        // mark as completed for progress reporting
        activeProgress.set(url, 1);
        emitOverallProgress();
        return cached.clone();
      }

      // Not cached: perform network fetch and stream into cache
      const response = await (originalFetch as typeof fetch)(input, init);
      if (!response.ok) return response;

      const contentLengthHeader = response.headers.get('content-length');
      const contentLength = contentLengthHeader ? parseInt(contentLengthHeader, 10) : NaN;

      if (!response.body || typeof response.body.getReader !== 'function') {
        // No streaming available: cache whole response directly
        const respClone = response.clone();
        await cache.put(url, respClone);
        activeProgress.set(url, 1);
        emitOverallProgress();
        return response;
      }

      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let received = 0;
      activeProgress.set(url, 0);
      emitOverallProgress();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          received += value.length;
          if (!Number.isNaN(contentLength) && contentLength > 0) {
            activeProgress.set(url, Math.min(1, received / contentLength));
          } else {
            // unknown length: use heuristic progress growth
            const prev = activeProgress.get(url) || 0;
            const next = Math.min(0.95, prev + value.length / (1024 * 256));
            activeProgress.set(url, next);
          }
          emitOverallProgress();
        }
      }

      const blob = new Blob(chunks as BlobPart[]);
      const headers: HeadersInit = {};
      response.headers.forEach((v, k) => (headers[k] = v));
      const storedResponse = new Response(blob, { status: response.status, statusText: response.statusText, headers });
      await cache.put(url, storedResponse.clone());

      activeProgress.set(url, 1);
      emitOverallProgress();

      return storedResponse;
    } catch (err) {
      // On error fallback to original fetch if available
      if (originalFetch) return originalFetch(input as any, init);
      throw err;
    } finally {
      // cleanup finished downloads
      try {
        const key = typeof input === 'string' ? input : (input as Request).url;
        if (activeProgress.get(key) === 1) {
          // remove after a small delay to allow overall progress to settle
          setTimeout(() => {
            activeProgress.delete(key);
            emitOverallProgress();
          }, 200);
        }
      } catch (e) {}
    }
  };

  (globalThis as any).__booknlp_fetch_installed = true;
  (globalThis as any).__booknlp_original_fetch = originalFetch;
  globalThis.fetch = wrappedFetch as any;
}

export async function clearCache(): Promise<void> {
  await caches.delete(cacheName);
}

export async function hasCached(url: string): Promise<boolean> {
  const cache = await caches.open(cacheName);
  const match = await cache.match(url);
  return !!match;
}

export default {
  installGlobalFetch,
  clearCache,
  hasCached,
};
