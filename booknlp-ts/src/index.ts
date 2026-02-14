import { SpaCyContext, BookNLPConfig, BookNLPResult, ProgressCallback } from './types';
import { EnglishBookNLP, createPipeline } from './english-booknlp';
import { installGlobalFetch } from './cache-service';
import { env } from '@huggingface/transformers';

/**
 * BookNLP TypeScript implementation for browser environment.
 *
 * IMPORTANT: This implementation requires pre-computed spaCy context as input.
 * Unlike the Python version, it does NOT perform tokenization, POS tagging, or
 * dependency parsing. You must obtain SpaCyContext from an external spaCy processor:
 *
 * 1. Use Python spaCy: nlp = spacy.load("en_core_web_sm"); doc = nlp(text)
 * 2. Convert to SpaCyContext with proper token features (pos, lemma, deprel, etc.)
 * 3. Pass to BookNLP.process()
 *
 * The BookNLP TypeScript version only handles:
 * - Entity tagging (via ONNX model)
 * - Supersense tagging (via ONNX model)
 * - Event detection (via ONNX model)
 *
 * All linguistic preprocessing must be done externally.
 */
export class BookNLP {
  private pipeline: EnglishBookNLP | null = null;

  async initialize(config: BookNLPConfig, progressCallback?: ProgressCallback): Promise<void> {
    // Install global cached fetch which will route all resource loads through
    // the cache service and emit progress updates via the provided callback.
    if (!config.cacheName) {
      config.cacheName = 'booknlp-resources-v1';
    }
    env.cacheKey = config.cacheName;
    this.pipeline = await createPipeline(config, progressCallback);
  }

  async process(spaCyContext: SpaCyContext): Promise<BookNLPResult> {
    if (!this.pipeline) {
      throw new Error('Pipeline not initialized. Call initialize() first.');
    }

    return this.pipeline.process(spaCyContext);
  }
}

export * from './types';
export * from './validation';
export * from './preprocessing';
export * from './tagger-controller';
export * from './entity-tagger';
export * from './english-booknlp';
export * from './crf-decoder';
export * from './advanced-postprocessor';
export * from './name-coref';
export { installGlobalFetch, clearCache, hasCached } from './cache-service';
