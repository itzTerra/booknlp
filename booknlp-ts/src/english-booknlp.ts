import { SpaCyContext, BookNLPConfig, BookNLPResult, Token, ResourceUrls } from 'types';
import { validateSpaCyContext, validateBookNLPConfig, throwIfValidationErrors } from 'validation';
import { convertSpaCyToTokens } from 'preprocessing';
import { EntityTagger } from 'entity-tagger';

import entityTagsetUrl from '../public/data/entity_cat.tagset?url';
import supersenseTagsetUrl from '../public/data/supersense.tagset?url';
import wordNetUrl from '../public/data/wordnet.first.sense?url';
import crfTransitionsUrl from '../public/data/crf_transitions.json?url';

function resolveResourceUrls(config: BookNLPConfig): ResourceUrls {
  if (config.resourceUrls) {
    return config.resourceUrls;
  }

  const baseUrl = config.resourceBaseUrl;
  if (baseUrl) {
    const normalized = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    return {
      entityTagset: `${normalized}entity_cat.tagset`,
      supersenseTagset: `${normalized}supersense.tagset`,
      wordNet: `${normalized}wordnet.first.sense`,
      crfTransitions: `${normalized}crf_transitions.json`,
    };
  }

  return {
    entityTagset: entityTagsetUrl,
    supersenseTagset: supersenseTagsetUrl,
    wordNet: wordNetUrl,
    crfTransitions: crfTransitionsUrl,
  };
}

export class BookNLPPipeline {
  private config: BookNLPConfig;
  private entityTagger: EntityTagger;
  private tokens: Token[] = [];
  private timing: Record<string, number> = {};
  private initialized: boolean = false;

  constructor(config: BookNLPConfig) {
    const configErrors = validateBookNLPConfig(config);
    throwIfValidationErrors(configErrors);

    const normalizedPipeline = Array.isArray(config.pipeline)
      ? config.pipeline
      : config.pipeline.split(',').map((task) => task.trim()).filter(Boolean);

    this.config = {
      ...config,
      pipeline: normalizedPipeline,
    };

    const resourceUrls = resolveResourceUrls(config);
    const modelPath = config.modelPath || 'Terraa/entities_google_bert_uncased_L-4_H-256_A-4-v1.0-ONNX';
    const executionProviders = config.executionProviders ?? ['wasm'];

    this.entityTagger = new EntityTagger(
      modelPath,
      resourceUrls,
      executionProviders,
      config.wasmPaths
    );
  }

  async initialize(): Promise<void> {
    await this.entityTagger.initialize();
    this.initialized = true;
  }

  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('BookNLPPipeline not initialized. Call initialize() first.');
    }
  }

  async process(spaCyContext: SpaCyContext): Promise<BookNLPResult> {
    this.ensureInitialized();

    const startTime = performance.now();

    const contextErrors = validateSpaCyContext(spaCyContext);
    throwIfValidationErrors(contextErrors);

    const conversionTime = performance.now();
    this.tokens = convertSpaCyToTokens(spaCyContext);
    this.timing['token_conversion'] = performance.now() - conversionTime;

    const taggerTime = performance.now();
    const taggerResults = await this.entityTagger.tag(
      this.tokens,
      spaCyContext.tokens,
    );
    this.timing['tagger_inference'] = performance.now() - taggerTime;

    // Apply pipeline task flags to selectively populate results
    // Mirror Python behavior (english_booknlp.py:105-111): only populate results for enabled tasks
    const doEvent = this.config.pipeline.includes('event');
    const doEntities = this.config.pipeline.includes('entity');
    const doSS = this.config.pipeline.includes('supersense');

    if (doEvent && taggerResults.events) {
      this.tokens.forEach((token) => {
        token.event = taggerResults.events?.has(token.tokenId) ?? false;
      });
    }

    let entities: BookNLPResult['entities'] = [];
    if (doEntities && taggerResults.entities) {
      entities = [...taggerResults.entities].sort((a, b) => {
        if (a.startToken !== b.startToken) {
          return a.startToken - b.startToken;
        }
        if (a.endToken !== b.endToken) {
          return a.endToken - b.endToken;
        }
        if (a.prop !== b.prop) {
          return a.prop.localeCompare(b.prop);
        }
        if (a.cat !== b.cat) {
          return a.cat.localeCompare(b.cat);
        }
        return a.text.localeCompare(b.text);
      });
      taggerResults.entities.forEach((entity) => {
        for (let i = entity.startToken; i < entity.endToken; i++) {
          if (i < this.tokens.length) {
            this.tokens[i].ner = entity.cat;
          }
        }
      });
    }

    let supersense: any[] = [];
    if (doSS && taggerResults.supersense) {
      supersense = taggerResults.supersense;
    }

    this.timing['total'] = performance.now() - startTime;

    return {
      tokens: this.tokens,
      sents: spaCyContext.sentences,
      nounChunks: [],
      entities: entities,
      supersense: supersense,
      timing: this.timing,
    };
  }
}

export async function createPipeline(config: BookNLPConfig): Promise<BookNLPPipeline> {
  const pipeline = new BookNLPPipeline(config);
  await pipeline.initialize();
  return pipeline;
}
