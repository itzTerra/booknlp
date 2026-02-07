import { AutoTokenizer, env } from '@huggingface/transformers';
import { SpaCyContext, SpaCyToken, Token, BertTokenizationResult } from 'types';

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
  private tokenizer: any;
  private initialized: boolean = false;
  private static environmentConfigured: boolean = false;

  async initialize(modelId: string): Promise<void> {
    this.configureTransformersEnvironment();
    const normalizedModelId = this.normalizeModelId(modelId);
    this.tokenizer = await AutoTokenizer.from_pretrained(normalizedModelId, {
      legacy: true
    });
    this.initialized = true;
  }

  private configureTransformersEnvironment(): void {
    if (Tokenizer.environmentConfigured) {
      return;
    }

    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.localModelPath = '';
    Tokenizer.environmentConfigured = true;
  }

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

    const clsEncoded = this.tokenizer.encode(SPECIAL_TOKENS.CLS, { add_special_tokens: false });
    addEncodedTokens(clsEncoded, null, null);

    for (let i = 0; i < spaCyTokens.length; i++) {
      const token = spaCyTokens[i];
      let tokenText = token.text;

      // Mirror Python behavior (entity_tagger.py:110): prepend "[CAP] " to text before tokenization
      // Preserve original casing unless we add the [CAP] marker (uncased BERT expects lowercase after it)
      if (this.isCapitalized(token.text)) {
        tokenText = `${SPECIAL_TOKENS.CAP} ${token.text.toLowerCase()}`;
      }

      const encoded = this.tokenizer.encode(tokenText, { add_special_tokens: false });
      const normalized = normalizeEncoded(encoded);
      const normalizedEncoded = normalized.ids.length > 0
        ? encoded
        : this.tokenizer.encode('[UNK]', { add_special_tokens: false });
      addEncodedTokens(normalizedEncoded, i, i);
    }

    const sepEncoded = this.tokenizer.encode(SPECIAL_TOKENS.SEP, { add_special_tokens: false });
    addEncodedTokens(sepEncoded, null, null);

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

    // Mirror Python behavior: preserve original casing, except when using [CAP] marker
    const textToTokenize = this.isCapitalized(text)
      ? `${SPECIAL_TOKENS.CAP} ${text.toLowerCase()}`
      : text;

    const encoded = this.tokenizer.encode(textToTokenize, { add_special_tokens: false });

    // Handle both formats: {ids: number[], tokens: string[]} or just number[]
    let ids: number[];
    if (Array.isArray(encoded)) {
      ids = encoded;
    } else if (encoded && encoded.ids && Array.isArray(encoded.ids)) {
      ids = encoded.ids;
    } else {
      console.error('Tokenizer encode returned unexpected format:', encoded);
      throw new Error(
        `Tokenizer encode failed: expected {ids: number[], tokens: string[]} or number[], got ${JSON.stringify(encoded)}`
      );
    }

    // Match Python behavior: return actual length, not Math.max(1, length)
    // Python uses len(toks) which can be 0 for some edge cases
    // Reference: booknlp/english/entity_tagger.py:116-126
    return ids.length;
  }

  private isCapitalized(text: string): boolean {
    if (text.length === 0) return false;
    return text[0] === text[0].toUpperCase() && text[0] !== text[0].toLowerCase();
  }
}

export function convertSpaCyToTokens(
  spaCyContext: SpaCyContext,
  paragraphId: number = 0,
): Token[] {
  const tokens: Token[] = [];
  let tokenId = 0;
  let currentParagraphId = paragraphId;

  const normalizeWhitespace = (text: string): string =>
    text.replace(/ /g, 'S').replace(/[\n\r]/g, 'N').replace(/\t/g, 'T');

  const isWhitespaceToken = (text: string): boolean => text.trim().length === 0;

  const sentences = spaCyContext.sentences ?? [{ start: 0, end: spaCyContext.tokens.length }];
  let skippedGlobal = 0;
  let currentWhitespace = '';
  let sentenceId = 0;

  for (const sentence of sentences) {
    let skippedInSentence = 0;
    const skipsInSentence: number[] = [];
    let curSkips = 0;

    for (let i = sentence.start; i < sentence.end; i++) {
      const tok = spaCyContext.tokens[i];
      if (isWhitespaceToken(tok.text)) {
        curSkips += 1;
      }
      skipsInSentence.push(curSkips);
    }

    let hasWord = false;

    for (let i = sentence.start; i < sentence.end; i++) {
      const spaCyToken = spaCyContext.tokens[i];

      if (i > 0 && spaCyContext.paragraphs) {
        for (const para of spaCyContext.paragraphs) {
          if (i === para.start && para.start > 0) {
            currentParagraphId++;
            break;
          }
        }
      }

      if (isWhitespaceToken(spaCyToken.text)) {
        skippedGlobal += 1;
        skippedInSentence += 1;
        currentWhitespace += spaCyToken.text;
        continue;
      }

      if (currentWhitespace.includes('\n\n')) {
        currentParagraphId++;
      }

      hasWord = true;

      const headInSentence = spaCyToken.dephead - sentence.start;
      const headSkip =
        headInSentence >= 0 && headInSentence < skipsInSentence.length
          ? skipsInSentence[headInSentence]
          : 0;
      const tokenSkip = skipsInSentence[i - sentence.start] ?? 0;
      const skipsBetweenTokenAndHead = headSkip - tokenSkip;
      const dephead = spaCyToken.dephead - skippedGlobal - skipsBetweenTokenAndHead;

      const normalizedText = normalizeWhitespace(spaCyToken.text);
      const token: Token = {
        paragraphId: currentParagraphId,
        sentenceId,
        withinSentenceId: i - sentence.start - skippedInSentence,
        tokenId: tokenId++,
        text: normalizedText,
        pos: spaCyToken.pos,
        finePos: spaCyToken.finePos,
        lemma: spaCyToken.lemma,
        deprel: spaCyToken.deprel,
        dephead,
        ner: null,
        startByte: spaCyToken.startByte,
        endByte: spaCyToken.endByte,
        morph: spaCyToken.morph || {},
        likeNum: spaCyToken.likeNum,
        isStop: spaCyToken.isStop,
        itext: normalizedText.toLowerCase(),
        inQuote: false,
        event: false,
      };

      tokens.push(token);
      currentWhitespace = '';
    }

    if (hasWord) {
      sentenceId += 1;
    }
  }
  return tokens;
}
