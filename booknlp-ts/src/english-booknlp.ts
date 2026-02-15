import { SpaCyContext, BookNLPConfig, BookNLPResult, Token, Resources, type ProgressCallback } from './types';
import { validateSpaCyContext, validateBookNLPConfig, throwIfValidationErrors } from './validation';
import { EntityTagger } from './entity-tagger';
import { NameCoref } from './name-coref';

import entityTagsetContent from './assets/data/entity_cat.tagset?raw';
import supersenseTagsetContent from './assets/data/supersense.tagset?raw';
import wordNetContent from './assets/data/wordnet.first.sense?raw';
import crfTransitionsContent from './assets/data/crf_transitions.json';

function resolveResources(): Resources {
  return {
    entityTagset: entityTagsetContent,
    supersenseTagset: supersenseTagsetContent,
    wordNet: wordNetContent,
    crfTransitions: crfTransitionsContent,
  };
}

export class EnglishBookNLP {
  private config: BookNLPConfig;
  private entityTagger: EntityTagger;
  private tokens: Token[] = [];
  private initialized: boolean = false;

  constructor(config: BookNLPConfig) {
    const configErrors = validateBookNLPConfig(config);
    throwIfValidationErrors(configErrors);

    this.config = config;
    const resources = resolveResources();
    const modelPath = config.modelPath || 'Terraa/entities_google_bert_uncased_L-4_H-256_A-4-v1.0-ONNX';
    const executionProviders = config.executionProviders ?? ['wasm'];

    this.entityTagger = new EntityTagger(
      modelPath,
      resources,
      executionProviders,
      config.cacheName,
      config.dtype,
    );
  }

  async initialize(progressCallback?: ProgressCallback): Promise<void> {
    await this.entityTagger.initialize(progressCallback);
    this.initialized = true;
  }

  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('EnglishBookNLP not initialized. Call initialize() first.');
    }
  }

  async process(spaCyContext: SpaCyContext): Promise<BookNLPResult> {
    this.ensureInitialized();

    const contextErrors = validateSpaCyContext(spaCyContext);
    throwIfValidationErrors(contextErrors);

    // spaCyContext.tokens is expected to already conform to the BookNLP `Token` shape
    // (no conversion step). Use them directly as the pipeline tokens.
    this.tokens = spaCyContext.tokens as Token[];

    const taggerResults = await this.entityTagger.tag(this.tokens);

    // ignore any debug payloads from tagger

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
      // Mirror Python entity sorting behavior: sort by (startToken, endToken, fullLabel, text)
      // Python sorts the tuple (start, end, label, text) where label is the combined prop_cat string.
      // Reference: booknlp/english/english_booknlp.py:254 sorted(entity_vals["entities"])
      const fullLabelEntities = taggerResults.entities.map(e => ({
        ...e,
        fullLabel: `${e.prop}_${e.cat}`,
      }));
      entities = fullLabelEntities
        .sort((a, b) => {
          if (a.startToken !== b.startToken) return a.startToken - b.startToken;
          if (a.endToken !== b.endToken) return a.endToken - b.endToken;
          if (a.fullLabel !== b.fullLabel) return a.fullLabel.localeCompare(b.fullLabel);
          return a.text.localeCompare(b.text);
        })
        .map(({ fullLabel, ...e }) => e);

      // compute in_quotes (1 if either boundary token is inside a quote else 0)
      const in_quotes: number[] = [];
      for (const ent of entities) {
        const start = ent.startToken;
        const end = ent.endToken;
        const startIn = this.tokens[start]?.inQuote ?? false;
        const endIn = this.tokens[end]?.inQuote ?? false;
        in_quotes.push(startIn || endIn ? 1 : 0);
      }

      // run name clustering routines (narrator, identical propers, only-nouns)
      const nameResolver = new NameCoref();
      const tupleEntities: Array<[number, number, string, string]> = entities.map((e) => [
        e.startToken,
        e.endToken,
        `${e.prop}_${e.cat}`,
        e.text,
      ]);

      let refs: number[] = new Array(entities.length).fill(-1);
      refs = nameResolver.cluster_narrator(tupleEntities, in_quotes, this.tokens);
      refs = nameResolver.cluster_identical_propers(tupleEntities, refs);
      refs = nameResolver.cluster_only_nouns(tupleEntities, refs, this.tokens);

      entities = entities.map((e, idx) => ({
        ...e,
        coref: refs[idx],
      }));
    }

    let supersense: any[] = [];
    if (doSS && taggerResults.supersense) {
      supersense = taggerResults.supersense;
    }

    // Extract noun chunks from spaCy context if available.
    // Python obtains these from spaCy's doc.noun_chunks.
    // Reference: booknlp/common/pipelines.py:184 and pipelines.py:tag() returning doc.noun_chunks
    const nounChunks = spaCyContext.nounChunks || [];

    return {
      tokens: this.tokens,
      sents: spaCyContext.sentences,
      nounChunks,
      entities: entities,
      supersense: supersense,
    };
  }

  async process_batch(spaCyContexts: SpaCyContext[]): Promise<BookNLPResult[]> {
    this.ensureInitialized();

    // Validate all contexts first
    for (const ctx of spaCyContexts) {
      const ctxErrors = validateSpaCyContext(ctx);
      throwIfValidationErrors(ctxErrors);
    }

    const tokensLists: Token[][] = spaCyContexts.map((c) => (c.tokens as Token[]));

    const taggerResults = await this.entityTagger.tag_batch(tokensLists);

    const doEvent = this.config.pipeline.includes('event');
    const doEntities = this.config.pipeline.includes('entity');
    const doSS = this.config.pipeline.includes('supersense');

    const results: BookNLPResult[] = [];

    for (let i = 0; i < spaCyContexts.length; i++) {
      const ctx = spaCyContexts[i];
      const tokens = tokensLists[i];
      const tr = taggerResults[i] || { entities: [], supersense: [], events: new Set<number>() };

      if (doEvent && tr.events) {
        tokens.forEach((token) => {
          token.event = tr.events?.has(token.tokenId) ?? false;
        });
      }

      let entities: BookNLPResult['entities'] = [];
      if (doEntities && tr.entities) {
        const fullLabelEntities = tr.entities.map((e) => ({
          ...e,
          fullLabel: `${e.prop}_${e.cat}`,
        }));

        entities = fullLabelEntities
          .sort((a, b) => {
            if (a.startToken !== b.startToken) return a.startToken - b.startToken;
            if (a.endToken !== b.endToken) return a.endToken - b.endToken;
            if (a.fullLabel !== b.fullLabel) return a.fullLabel.localeCompare(b.fullLabel);
            return a.text.localeCompare(b.text);
          })
          .map(({ fullLabel, ...e }) => e);

        const in_quotes: number[] = [];
        for (const ent of entities) {
          const start = ent.startToken;
          const end = ent.endToken;
          const startIn = tokens[start]?.inQuote ?? false;
          const endIn = tokens[end]?.inQuote ?? false;
          in_quotes.push(startIn || endIn ? 1 : 0);
        }

        const nameResolver = new NameCoref();
        const tupleEntities: Array<[number, number, string, string]> = entities.map((e) => [
          e.startToken,
          e.endToken,
          `${e.prop}_${e.cat}`,
          e.text,
        ]);

        let refs: number[] = new Array(entities.length).fill(-1);
        refs = nameResolver.cluster_narrator(tupleEntities, in_quotes, tokens);
        refs = nameResolver.cluster_identical_propers(tupleEntities, refs);
        refs = nameResolver.cluster_only_nouns(tupleEntities, refs, tokens);

        entities = entities.map((e, idx) => ({
          ...e,
          coref: refs[idx],
        }));
      }

      let supersense: any[] = [];
      if (doSS && tr.supersense) {
        supersense = tr.supersense;
      }

      const nounChunks = ctx.nounChunks || [];

      results.push({
        tokens,
        sents: ctx.sentences,
        nounChunks,
        entities: entities,
        supersense: supersense,
      });
    }

    return results;
  }
}

export async function createPipeline(config: BookNLPConfig, progressCallback?: ProgressCallback): Promise<EnglishBookNLP> {
  const pipeline = new EnglishBookNLP(config);
  await pipeline.initialize(progressCallback);
  return pipeline;
}
