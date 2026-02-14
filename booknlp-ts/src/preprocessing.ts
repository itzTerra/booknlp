// import { AutoTokenizer, env } from '@huggingface/transformers';
import { SpaCyContext, SpaCyToken, Token, BertTokenizationResult } from './types';

const SPECIAL_TOKENS = {
  CLS: '[CLS]',
  SEP: '[SEP]',
  CAP: '[CAP]',
  PAD: '[PAD]',
};

// Python splits at 500 wordpieces before adding [CLS]/[SEP], so allow +2 here.
// Reference: booknlp/english/entity_tagger.py (max_sentence_length = 500).
const MAX_SUBWORD_LENGTH = 502;

export class Tokenizer {
  // private tokenizer: any;
  private initialized: boolean = false;
  // private static environmentConfigured: boolean = false;
  private vocab: Map<string, number> = new Map();
  private unkTokenId: number = 100;
  private clsTokenId: number = 101;
  private sepTokenId: number = 102;
  private maxInputCharsPerWord: number = 100;

  async initialize(modelId: string): Promise<void> {
    // this.configureTransformersEnvironment();
    const normalizedModelId = this.normalizeModelId(modelId);
    // this.tokenizer = await AutoTokenizer.from_pretrained(normalizedModelId, {
    //   legacy: true
    // });
    await this.loadVocab(normalizedModelId);

    // Ensure [CAP] exists in the vocab mapping. Python adds [CAP] via
    // tokenizer.add_tokens(...). If add_tokens is available use it; otherwise
    // ensure our loaded `vocab.txt` contains a mapping for [CAP], inserting a
    // fallback id (30522) when missing so counts/ids remain stable.
    // if (this.tokenizer.add_tokens && typeof this.tokenizer.add_tokens === 'function') {
    //   try {
    //     console.warn('[Tokenizer] Calling add_tokens([CAP])...');
    //     const result = this.tokenizer.add_tokens([SPECIAL_TOKENS.CAP], { special_tokens: true });
    //     console.warn(`[Tokenizer] add_tokens returned: ${result}`);
    //   } catch (e) {
    //     console.warn('[Tokenizer] add_tokens failed, falling back to vocab mapping');
    //   }
    // } else {
    //   console.warn('[Tokenizer] add_tokens method not available; ensuring vocab mapping for [CAP]');
    // }

    // If `[CAP]` missing from loaded vocab, add a fallback id 30522.
    if (!this.vocab.has(SPECIAL_TOKENS.CAP)) {
      this.vocab.set(SPECIAL_TOKENS.CAP, 30522);
    }

    // Debug: decode a common token id to inspect tokenizer if available
    // try {
    //   const decoded = this.tokenizer.decode([this.vocab.get(SPECIAL_TOKENS.CAP) ?? 30522]);
    //   console.warn(`[Tokenizer] Token ${this.vocab.get(SPECIAL_TOKENS.CAP) ?? 30522} decodes to: "${decoded}"`);
    // } catch (e) {
    //   console.warn(`[Tokenizer] Failed to decode token for [CAP]: ${e}`);
    // }

    this.initialized = true;
  }

  getCapTokenId(): number {
    return this.vocab.get(SPECIAL_TOKENS.CAP) ?? 30522;
  }

  private resolveVocabUrl(modelId: string): string {
    if (modelId.startsWith('http://') || modelId.startsWith('https://')) {
      return `${modelId.replace(/\/$/, '')}/vocab.txt`;
    }

    if (modelId.includes('/')) {
      return `https://huggingface.co/${modelId}/resolve/main/vocab.txt`;
    }

    return `${modelId.replace(/\/$/, '')}/vocab.txt`;
  }

  private async loadVocab(modelId: string): Promise<void> {
    const vocabUrl = this.resolveVocabUrl(modelId);
    const response = await fetch(vocabUrl);
    if (!response.ok) {
      throw new Error(`Failed to load vocab.txt from ${vocabUrl}`);
    }

    const text = await response.text();
    const tokens = text.split(/\r?\n/);
    this.vocab = new Map();
    tokens.forEach((token, index) => {
      if (token.length > 0) {
        this.vocab.set(token, index);
      }
    });

    const unkId = this.vocab.get('[UNK]');
    const clsId = this.vocab.get('[CLS]');
    const sepId = this.vocab.get('[SEP]');
    if (unkId !== undefined) this.unkTokenId = unkId;
    if (clsId !== undefined) this.clsTokenId = clsId;
    if (sepId !== undefined) this.sepTokenId = sepId;
  }

  private wordpieceTokenize(text: string): string[] {
    if (!text) {
      return [];
    }

    const outputTokens: string[] = [];
    const words = text.split(/\s+/).filter(Boolean);

    for (const word of words) {
      if (word.length > this.maxInputCharsPerWord) {
        outputTokens.push('[UNK]');
        continue;
      }

      let isBad = false;
      let start = 0;
      const subTokens: string[] = [];

      while (start < word.length) {
        let end = word.length;
        let curSubstr: string | null = null;

        while (start < end) {
          let substr = word.slice(start, end);
          if (start > 0) {
            substr = `##${substr}`;
          }
          if (this.vocab.has(substr)) {
            curSubstr = substr;
            break;
          }
          end -= 1;
        }

        if (!curSubstr) {
          isBad = true;
          break;
        }

        subTokens.push(curSubstr);
        start = end;
      }

      if (isBad) {
        outputTokens.push('[UNK]');
      } else {
        outputTokens.push(...subTokens);
      }
    }

    return outputTokens;
  }

  private tokensToIds(tokens: string[]): number[] {
    return tokens.map(token => this.vocab.get(token) ?? this.unkTokenId);
  }

  // private configureTransformersEnvironment(): void {
  //   if (Tokenizer.environmentConfigured) {
  //     return;
  //   }

  //   // env.allowLocalModels = false;
  //   // env.allowRemoteModels = true;
  //   // env.localModelPath = '';
  //   Tokenizer.environmentConfigured = true;
  // }

  private normalizeModelId(modelId: string): string {
    if (modelId.startsWith('http://') || modelId.startsWith('https://')) {
      return modelId;
    }

    if (modelId.startsWith('/')) {
      return modelId.slice(1);
    }

    return modelId;
  }

  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('BertTokenizer not initialized. Call initialize() first.');
    }
  }

  private prepareTokenText(text: string): string {
    if (!text) {
      return text;
    }

    const firstChar = text[0];
    if (firstChar && firstChar.toLowerCase() !== firstChar) {
      return `${SPECIAL_TOKENS.CAP} ${text.toLowerCase()}`;
    }

    return text;
  }

  private countCapTokensSeparately(text: string): number {
    const firstChar = text[0];
    const isCapitalized = firstChar && firstChar.toLowerCase() !== firstChar;

    if (!isCapitalized) {
      const tokens = this.wordpieceTokenize(text);
      return tokens.length;
    }

    // Python registers [CAP] as a special token (ID 30522) via add_tokens().
    // transformers.js doesn't support add_tokens(), so we count [CAP] manually as 1 token.
    // Then count the lowercase word tokens separately.
    // Reference: booknlp/english/tagger.py:80-81
    const wordTokens = this.wordpieceTokenize(text.toLowerCase());
    return 1 + wordTokens.length;  // [CAP] (1) + lowercase word tokens
  }

  debugEncodeToken(text: string): { prepared: string; ids: number[]; tokens: string[] } {
    this.ensureInitialized();

    const firstChar = text[0];
    const isCapitalized = firstChar && firstChar.toLowerCase() !== firstChar;

    if (isCapitalized) {
      // Match actual tokenizeTokens behavior: [CAP] token + word tokens
      const lowercased = text.toLowerCase();
      const wordTokens = this.wordpieceTokenize(lowercased);
      const wordIds = this.tokensToIds(wordTokens);
      // Use a stable vocab id for [CAP] (fallback to 30522)
      const capId = this.vocab.get(SPECIAL_TOKENS.CAP) ?? 30522;
      const ids = [capId, ...wordIds];
      const prepared = `[CAP] ${lowercased}`;
      return { prepared, ids, tokens: [SPECIAL_TOKENS.CAP, ...wordTokens] };
    }

    // Non-capitalized: encode as-is
    const prepared = text;
    const tokens = this.wordpieceTokenize(prepared);
    const ids = this.tokensToIds(tokens);

    return { prepared, ids, tokens };
  }

  tokenize(spaCyContext: SpaCyContext): BertTokenizationResult {
    return this.tokenizeTokens(spaCyContext.tokens);
  }

  tokenizeTokens(spaCyTokens: SpaCyToken[]): BertTokenizationResult {
    this.ensureInitialized();

    const tokens: string[] = [];
    const tokenIds: number[] = [];
    const attentionMask: number[] = [];
    const subwordToTokenMap: Array<number | null> = [];
    const subwordRanges: Array<{ start: number; end: number; tokenIdx: number | null }> = [];

    // Helper to normalize tokenizer output: handle both {ids, tokens} and bare array formats
    const normalizeEncoded = (encoded: any): { tokens: string[]; ids: number[] } => {
      if (Array.isArray(encoded)) {
        // Browser format: just array of IDs
        return {
          ids: encoded,
          tokens: encoded.map((id: number) => `[${id}]`), // Placeholder tokens
        };
      } else if (encoded && encoded.ids && Array.isArray(encoded.ids)) {
        // Standard format: {ids, tokens}
        return encoded;
      }
      throw new Error(`Unexpected tokenizer output format: ${JSON.stringify(encoded)}`);
    };

    const addEncodedTokens = (
      encoded: any,
      originalTokenIndex: number | null,
      subwordToTokenMapValue: number | null,
    ): void => {
      const normalized = normalizeEncoded(encoded);
      const startIdx = tokenIds.length;

      for (let i = 0; i < normalized.ids.length; i++) {
        tokens.push(normalized.tokens[i] ?? `[${normalized.ids[i]}]`);
        tokenIds.push(normalized.ids[i]);
        attentionMask.push(1);
        subwordToTokenMap.push(subwordToTokenMapValue);
      }
      const endIdx = tokenIds.length;

      subwordRanges.push({
        start: startIdx,
        end: endIdx,
        tokenIdx: originalTokenIndex,
      });
    };

    addEncodedTokens({ ids: [this.clsTokenId], tokens: [SPECIAL_TOKENS.CLS] }, null, null);

    for (let i = 0; i < spaCyTokens.length; i++) {
      const token = spaCyTokens[i];
      const firstChar = token.text[0];
      const isCapitalized = firstChar && firstChar.toLowerCase() !== firstChar;

      if (isCapitalized) {
        // Mirror Python: treat [CAP] as a single token, then tokenize the lowercase word.
        const capStartIdx = tokenIds.length;
        tokens.push(SPECIAL_TOKENS.CAP);
        tokenIds.push(this.vocab.get('[CAP]') ?? 30522);
        attentionMask.push(1);
        subwordToTokenMap.push(i);

        const lowercased = token.text.toLowerCase();
        const wordTokens = this.wordpieceTokenize(lowercased);
        const wordIds = this.tokensToIds(wordTokens);
        for (let j = 0; j < wordIds.length; j++) {
          tokens.push(wordTokens[j] ?? `[${wordIds[j]}]`);
          tokenIds.push(wordIds[j]);
          attentionMask.push(1);
          subwordToTokenMap.push(i);
        }

        const capEndIdx = tokenIds.length;
        subwordRanges.push({
          start: capStartIdx,
          end: capEndIdx,
          tokenIdx: i,
        });
      } else {
        // Normal tokenization for non-capitalized tokens
        const wordTokens = this.wordpieceTokenize(token.text);
        const wordIds = this.tokensToIds(wordTokens);
        addEncodedTokens({ ids: wordIds, tokens: wordTokens }, i, i);
      }
    }

    addEncodedTokens({ ids: [this.sepTokenId], tokens: [SPECIAL_TOKENS.SEP] }, null, null);

    if (tokens.length > MAX_SUBWORD_LENGTH) {
      tokens.splice(MAX_SUBWORD_LENGTH);
      tokenIds.splice(MAX_SUBWORD_LENGTH);
      attentionMask.splice(MAX_SUBWORD_LENGTH);
      subwordToTokenMap.splice(MAX_SUBWORD_LENGTH);

      subwordRanges.forEach(range => {
        if (range.end > MAX_SUBWORD_LENGTH) {
          range.end = MAX_SUBWORD_LENGTH;
        }
      });
    }

    const transformMatrix = this.buildTransformMatrix(subwordRanges, tokens.length, spaCyTokens.length + 2);

    return {
      tokens,
      tokenIds,
      attentionMask,
      transformMatrix,
      subwordToTokenMap,
    };
  }

  private buildTransformMatrix(
    subwordRanges: Array<{ start: number; end: number; tokenIdx: number | null }>,
    numSubwords: number,
    numOriginalTokens: number,
  ): number[][] {
    const matrix: number[][] = [];

    for (let originalIdx = 0; originalIdx < numOriginalTokens; originalIdx++) {
      const row = new Array(numSubwords).fill(0);

      const range = subwordRanges[originalIdx];
      if (range && range.start < numSubwords) {
        const subwordCount = Math.min(range.end, numSubwords) - range.start;
        const weight = subwordCount > 0 ? 1.0 / subwordCount : 0;

        for (let subwordIdx = range.start; subwordIdx < Math.min(range.end, numSubwords); subwordIdx++) {
          row[subwordIdx] = weight;
        }
      }

      matrix.push(row);
    }

    return matrix;
  }

  countTokens(text: string): number {
    this.ensureInitialized();

    if (!text) {
      return 0;
    }

    // Use the same deterministic encoder used by batching so counts match
    // the actual tokenization output. This reduces mismatch risk between
    // the Python tokenizer and this JS implementation.
    try {
      const enc = this.debugEncodeToken(text);
      return enc.ids.length;
    } catch (e) {
      // Fallback: previous count logic
      return this.countCapTokensSeparately(text);
    }
  }
}
// Note: conversion from raw spaCy token list into BookNLP `Token` objects
// was previously handled by `convertSpaCyToTokens`. That conversion has been
// removed: callers are expected to supply a `SpaCyContext` whose `tokens`
// already conform to the BookNLP `Token` shape (see `types.ts`).
