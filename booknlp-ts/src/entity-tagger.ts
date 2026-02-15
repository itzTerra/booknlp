import { Token, EntityAnnotation, SupersenseAnnotation, Resources as Resources, ExecutionProvider, type ProgressCallback, type DType } from './types';
import { ONNXTaggerController } from './tagger-controller';
import { Tokenizer } from './preprocessing';
import { CRFDecoder } from './crf-decoder';
import { AdvancedPostProcessor } from './advanced-postprocessor';

interface WordNetSense {
  [key: string]: number;
}

export class EntityTagger {
  private controller: ONNXTaggerController;
  private advancedPostProcessor: AdvancedPostProcessor;
  private crfDecoder: CRFDecoder;
  private resources: Resources;
  private tokenizer: Tokenizer;
  private modelId: string;
  private wordNetSenses: WordNetSense = {};
  private entityTagset: Map<number, string> = new Map();
  private supersenseTagset: Map<number, string> = new Map();
  private entityCategoryMap: Map<string, string> = new Map();
  private initialized: boolean = false;

  constructor(
    modelPath: string,
    resources: Resources,
    executionProviders: ExecutionProvider[] = ['wasm'],
    wasmPaths?: string | Record<string, string>,
    dtype?: DType,
  ) {
    this.controller = new ONNXTaggerController(modelPath, executionProviders, wasmPaths, dtype);
    this.advancedPostProcessor = new AdvancedPostProcessor();
    this.crfDecoder = new CRFDecoder();
    this.resources = resources;
    this.tokenizer = new Tokenizer();
    this.modelId = modelPath;
    this.buildEntityCategoryMap();
  }

  static async fromHuggingFace(
    repoId: string,
    resources: Resources,
    executionProviders: ExecutionProvider[] = ['wasm'],
    wasmPaths?: string | Record<string, string>,
  ): Promise<EntityTagger> {
    const tagger = new EntityTagger(
      repoId,
      resources,
      executionProviders,
      wasmPaths,
    );
    await tagger.initialize();
    return tagger;
  }

  private loadTagsets(entityContent: string, supersenseContent: string): void {
    entityContent.split('\n').forEach(line => {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith('#')) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 2) {
          const tag = parts[0];
          const id = parseInt(parts[1], 10);
          this.entityTagset.set(id, tag);
        }
      }
    });

    supersenseContent.split('\n').forEach(line => {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith('#')) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 2) {
          const tag = parts[0];
          const id = parseInt(parts[1], 10);
          this.supersenseTagset.set(id, tag);
        }
      }
    });

    this.advancedPostProcessor.setTagsets(this.entityTagset, this.supersenseTagset);
  }

  private loadWordNetSenses(content: string): void {
    content.split('\n').forEach(line => {
      const trimmed = line.trim();
      if (trimmed) {
        const parts = trimmed.split(/\t/);
        if (parts.length >= 3) {
          const word = parts[0];
          const pos = parts[1];
          const senseStr = parts[2].split(' ')[0];
          const sense = parseInt(senseStr, 10);
          const key = `${word}.${pos}`;
          this.wordNetSenses[key] = sense;
        }
      }
    });
  }

  private async loadResources(): Promise<void> {
    // Resources are expected to be provided as inlined contents (strings) at build time
    const entityContent = this.resources.entityTagset;
    const supersenseContent = this.resources.supersenseTagset;
    const wordNetContent = this.resources.wordNet;

    this.loadTagsets(entityContent, supersenseContent);
    this.loadWordNetSenses(wordNetContent);
    // Initialize CRF decoder with transition matrices
    // Transitions define penalty scores for invalid tag sequences (e.g., I-PER after O)
    this.crfDecoder.loadTransitions(this.resources.crfTransitions);
  }

  private buildEntityCategoryMap(): void {
    const patterns = [
      { prefix: 'PROP_PER', category: 'PER', propType: 'PROP' },
      { prefix: 'NOM_PER', category: 'PER', propType: 'NOM' },
      { prefix: 'PRON_PER', category: 'PER', propType: 'PRON' },
      { prefix: 'PROP_FAC', category: 'FAC', propType: 'PROP' },
      { prefix: 'NOM_FAC', category: 'FAC', propType: 'NOM' },
      { prefix: 'PRON_FAC', category: 'FAC', propType: 'PRON' },
      { prefix: 'PROP_LOC', category: 'LOC', propType: 'PROP' },
      { prefix: 'NOM_LOC', category: 'LOC', propType: 'NOM' },
      { prefix: 'PRON_LOC', category: 'LOC', propType: 'PRON' },
      { prefix: 'PROP_GPE', category: 'GPE', propType: 'PROP' },
      { prefix: 'NOM_GPE', category: 'GPE', propType: 'NOM' },
      { prefix: 'PRON_GPE', category: 'GPE', propType: 'PRON' },
      { prefix: 'PROP_ORG', category: 'ORG', propType: 'PROP' },
      { prefix: 'NOM_ORG', category: 'ORG', propType: 'NOM' },
      { prefix: 'PRON_ORG', category: 'ORG', propType: 'PRON' },
      { prefix: 'PROP_VEH', category: 'VEH', propType: 'PROP' },
      { prefix: 'NOM_VEH', category: 'VEH', propType: 'NOM' },
      { prefix: 'PRON_VEH', category: 'VEH', propType: 'PRON' },
    ];

    patterns.forEach(({ prefix, category }) => {
      this.entityCategoryMap.set(prefix, category);
      this.entityCategoryMap.set(`B-${prefix}`, category);
      this.entityCategoryMap.set(`I-${prefix}`, category);
    });
  }

  async initialize(progressCallback?: ProgressCallback): Promise<void> {
    await Promise.all([
      this.controller.loadModel(progressCallback),
      this.loadResources(),
      this.tokenizer.initialize(this.modelId),
    ]);
    this.initialized = true;
  }

  private getWordNetSense(token: Token): number {
    const text = token.text;
    let pos = token.pos;
    if (pos === 'NOUN') {
      pos = 'n';
    } else if (pos === 'VERB') {
      pos = 'v';
    } else {
      return 1;
    }

    const term = text.split(' ').pop()?.toLowerCase() || '';
    const key = `${term}.${pos}`;

    return this.wordNetSenses[key] || 1;
    }
  async tag(
    tokens: Token[],
  ): Promise<{
    entities?: EntityAnnotation[];
    supersense?: SupersenseAnnotation[];
    events?: Set<number>;
  }> {
    if (!this.initialized) {
      throw new Error('EntityTagger not initialized. Call initialize() first.');
    }

    const maxBatchSize = 32;
    const filteredTokens = tokens.filter((token) => token.text.trim().length > 0);

    if (filteredTokens.length === 0) {
      return { entities: [], supersense: [], events: new Set<number>() };
    }

    const {
      batches: sentenceBatches,
      firstPhaseCount: firstPhaseChunks,
      secondPhaseGroupLengths,
      secondPhaseGroupChunkCounts,
      chunkWordpieceLengths,
      chunkWordpieceSamples,
      sentsSample,
    } = this.createSentenceBatches(filteredTokens);

    // Match Python: get_batches sorts by BERT token length before batching.
    // Reference: booknlp/common/layered_reader.py:107-140
    const sentenceLengths = sentenceBatches.map((batch) =>
      this.estimateSentenceLength(batch)
    );
    const orderedSentences = sentenceBatches
      .map((batch, index) => ({
        batch,
        length: sentenceLengths[index],
        index,
      }))
      .sort((a, b) => a.length - b.length);

    const preGroupCount = sentenceBatches.length;
    const postSortCount = orderedSentences.length;
    const secondPhaseGroups = sentenceBatches.length;

    const allEntityAnnotations: EntityAnnotation[] = [];
    const allSupersenseAnnotations: SupersenseAnnotation[] = [];
    const allEvents = new Set<number>();

    let batchCount = 0;
    let batchIdx = 0;
    let currentBatchSize = maxBatchSize;

    // debug collection removed

    while (batchIdx < orderedSentences.length) {
      const batchSizeUsed = currentBatchSize;
      const batchEnd = Math.min(batchIdx + batchSizeUsed, orderedSentences.length);
      const batchSlice = orderedSentences.slice(batchIdx, batchEnd);
      const batchMaxLen = Math.max(...batchSlice.map(item => item.length));
      const batchInputs = batchSlice.map(item => item.batch);

      // debug collection removed

      // Debug: Collect WordNet sense batch information ONCE per ONNX inference batch
      // Match Python: _get_wn() creates one wn_batch per ONNX batch (not per phase-2 group)
      // Python iterates through batched_pos (74 items for 470 sentences), creating 74 wn_batches
      // Reference: booknlp/english/entity_tagger.py:49-89, booknlp/english/entity_tagger.py:222
      let batchTokenCount = 0;
      let maxOriginalTokens = 0;

      for (const batch of batchInputs) {
        batchTokenCount += batch.length;
        // Calculate max original token count (before BERT tokenization) across all sentences
        // Matches processSingleBatch: originalTokenCount = batch.tokens.length + 2 ([CLS] + tokens + [SEP])
        // Reference: booknlp-ts/src/entity-tagger.ts:579
        const originalTokenCount = batch.length + 2;
        maxOriginalTokens = Math.max(maxOriginalTokens, originalTokenCount);
      }

      const wnSensesLength = maxOriginalTokens;
      // (no-op: wordnet batch collection removed)

      const batchResults = await this.processSentenceBatch(batchInputs);
      batchCount++;

      if (batchResults.entities) {
        allEntityAnnotations.push(...batchResults.entities);
      }
      if (batchResults.supersense) {
        allSupersenseAnnotations.push(...batchResults.supersense);
      }
      if (batchResults.events) {
        batchResults.events.forEach(tokenId => allEvents.add(tokenId));
      }
      // (no-op: per-batch debug collection removed)

      // Adaptive batch sizing to match Python
      // Reference: booknlp/common/layered_reader.py:318-325
      if (batchMaxLen > 200) {
        currentBatchSize = 6;
      } else if (batchMaxLen > 100) {
        currentBatchSize = 12;
      } else {
        currentBatchSize = maxBatchSize;
      }

      batchIdx += batchSizeUsed;
        }

        return {
          entities: allEntityAnnotations,
          supersense: allSupersenseAnnotations,
          events: allEvents,
        };
  }

  async tag_batch(
    texts: Token[][],
  ): Promise<Array<{
    entities?: EntityAnnotation[];
    supersense?: SupersenseAnnotation[];
    events?: Set<number>;
  }>> {
    if (!this.initialized) {
      throw new Error('EntityTagger not initialized. Call initialize() first.');
    }

    const maxBatchSize = 32;

    // We'll clone tokens and assign globally-unique tokenId offsets per text so
    // inference outputs (which use tokenId) can be mapped back to their origin.
    const clonedTexts: Token[][] = [];
    const clonedIdToTextIndex = new Map<number, number>();
    const clonedIdToOriginalId = new Map<number, number>();

    let offset = 0;
    for (let ti = 0; ti < texts.length; ti++) {
      const tokens = texts[ti] || [];
      // Keep original token indices so we can map back when tokenId is missing
      const indexed = tokens.map((t, idx) => ({ t, idx }));
      const filtered = indexed.filter(({ t }) => t.text && t.text.trim().length > 0);
      if (filtered.length === 0) {
        clonedTexts.push([]);
        continue;
      }

      const base = offset;
      const cloned = filtered.map(({ t, idx }, localIndex) => {
        const originalId = typeof t.tokenId === 'number' ? t.tokenId : idx;
        // Assign a unique cloned id per token using the running offset + local index
        const copy: Token = { ...t, tokenId: base + localIndex };
        clonedIdToTextIndex.set(copy.tokenId as number, ti);
        // Map cloned id back to the original token index (or original tokenId if present)
        clonedIdToOriginalId.set(copy.tokenId as number, originalId);
        return copy;
      });

      clonedTexts.push(cloned);
      // Advance offset by the number of tokens we added for this text to keep ranges unique
      offset += filtered.length;
    }

    // Prepare empty results per input text
    const results: Array<{ entities: EntityAnnotation[]; supersense: SupersenseAnnotation[]; events: Set<number> }> =
      texts.map(() => ({ entities: [], supersense: [], events: new Set<number>() }));

    // Build combined batches from cloned texts
    const combinedBatches: Token[][] = [];
    for (const cloned of clonedTexts) {
      if (!cloned || cloned.length === 0) continue;
      const { batches } = this.createSentenceBatches(cloned);
      for (const b of batches) combinedBatches.push(b);
    }

    if (combinedBatches.length === 0) {
      return results;
    }

    const sentenceLengths = combinedBatches.map((batch) => this.estimateSentenceLength(batch));
    const orderedSentences = combinedBatches
      .map((batch, index) => ({ batch, length: sentenceLengths[index], index }))
      .sort((a, b) => a.length - b.length);

    let batchIdx = 0;
    let currentBatchSize = maxBatchSize;

    while (batchIdx < orderedSentences.length) {
      const batchSizeUsed = currentBatchSize;
      const batchEnd = Math.min(batchIdx + batchSizeUsed, orderedSentences.length);
      const batchSlice = orderedSentences.slice(batchIdx, batchEnd);
      const batchMaxLen = Math.max(...batchSlice.map((item) => item.length));
      const batchInputs = batchSlice.map((item) => item.batch);

      const batchResults = await this.processSentenceBatch(batchInputs);

      // Remap entities to their original text and original token ids
      if (batchResults.entities) {
        for (const ent of batchResults.entities) {
          const clonedStart = ent.startToken as number;
          const clonedEnd = ent.endToken as number;
          const textIndex = clonedIdToTextIndex.get(clonedStart);
          const origStart = clonedIdToOriginalId.get(clonedStart);
          const origEnd = clonedIdToOriginalId.get(clonedEnd);
          if (textIndex === undefined || origStart === undefined || origEnd === undefined) continue;
          results[textIndex].entities.push({ ...ent, startToken: origStart, endToken: origEnd });
        }
      }

      if (batchResults.supersense) {
        for (const ss of batchResults.supersense) {
          const clonedStart = ss[0] as number;
          const clonedEnd = ss[1] as number;
          const textIndex = clonedIdToTextIndex.get(clonedStart);
          const origStart = clonedIdToOriginalId.get(clonedStart);
          const origEnd = clonedIdToOriginalId.get(clonedEnd);
          if (textIndex === undefined || origStart === undefined || origEnd === undefined) continue;
          results[textIndex].supersense.push([origStart, origEnd, ss[2], ss[3]] as any);
        }
      }

      if (batchResults.events) {
        for (const tokenId of batchResults.events) {
          const clonedId = tokenId as number;
          const textIndex = clonedIdToTextIndex.get(clonedId);
          const orig = clonedIdToOriginalId.get(clonedId);
          if (textIndex === undefined || orig === undefined) continue;
          results[textIndex].events.add(orig);
        }
      }

      if (batchMaxLen > 200) {
        currentBatchSize = 6;
      } else if (batchMaxLen > 100) {
        currentBatchSize = 12;
      } else {
        currentBatchSize = maxBatchSize;
      }

      batchIdx += batchSizeUsed;
    }

    return results;
  }

  private createSentenceBatches(
    tokens: Token[],
  ): {
    batches: Array<Token[]>;
    firstPhaseCount: number;
    secondPhaseGroupLengths: number[];
    secondPhaseGroupChunkCounts: number[];
    chunkWordpieceLengths: number[];
    chunkWordpieceSamples: any[];
    sentsSample: any[];
  } {
    const maxSequenceLength = 500;
    const sentenceChunks: Array<Token[]> = [];

    let currentSentenceTokens: Token[] = [];
    let currentLength = 0;
    let lastSentenceId = -1;
    let splitCount = 0;

    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      const tokenLength = this.estimateTokenLength(token);
      const sentenceChanged = lastSentenceId !== -1 && token.sentenceId !== lastSentenceId;

      // Match Python split condition when sentences exceed max length.
      // Reference: booknlp/english/entity_tagger.py:108-114
      if ((sentenceChanged || currentLength + tokenLength > maxSequenceLength) && currentSentenceTokens.length > 0) {
        sentenceChunks.push(currentSentenceTokens);
        splitCount++;
        currentSentenceTokens = [];
        currentLength = 0;
      }

      currentSentenceTokens.push(token);
      // tokenLength is already an integer from countTokens, no Math.floor needed
      currentLength += tokenLength;
      lastSentenceId = token.sentenceId;
    }

    if (currentSentenceTokens.length > 0) {
      sentenceChunks.push(currentSentenceTokens);
    }
    // debug collection removed
    const firstPhaseCount = sentenceChunks.length;
    // Prepare a compact sample placeholder of the first few phase-1 sentence chunks
    const sentsSample: any[] = [];
    // Phase 2: Group phase-1 chunks into batches, recalculating wordpiece lengths
    // Python does this in entity_tagger.py:150-187, recalculating lengths fresh for each chunk
    // Reference: booknlp/english/entity_tagger.py:151-153
    const chunkWordpieceLengths: number[] = [];
    const chunkWordpieceSamples: any[] = [];

    // Batches for ONNX inference - don't need wordpieceLength after phase 2 grouping
    const batches: Array<Token[]> = [];
    let combinedTokens: Token[] = [];
    let combinedLength = 0;
    let groupChunkCount = 0;
    const secondPhaseGroupLengths: number[] = [];
    const secondPhaseGroupChunkCounts: number[] = [];

    for (const chunk of sentenceChunks) {
      // Recalculate chunk wordpiece length in phase 2, exactly like Python does
      // Python doesn't store lengths from phase 1; it calculates them fresh when grouping
      // Reference: booknlp/english/entity_tagger.py:159-162
      const chunkLength = this.calculateChunkWordpieceLength(chunk);
      // Store for debug output
      chunkWordpieceLengths.push(chunkLength);

      // debug collection removed
      // Known Python values for indices 9, 20, 36-37, 84, 113...
      const chunkIdx = chunkWordpieceLengths.length - 1;
      if (chunkIdx < 150) {
        // Minimal sample placeholder
        chunkWordpieceSamples.push({ chunkIdx, tokenCount: chunk.length });
      }

      // Match Python: check if adding this chunk would exceed limit
      // Python: if sent_len + cur_length >= max_sentence_length
      // Reference: booknlp/english/entity_tagger.py:164
      if (chunkLength + combinedLength >= maxSequenceLength) {
        // Emit current group before adding this chunk
        // Python always appends sentence here, even if just [CLS][SEP]
        // Reference: booknlp/english/entity_tagger.py:165-171
        batches.push(combinedTokens);
        secondPhaseGroupLengths.push(combinedLength);
        secondPhaseGroupChunkCounts.push(groupChunkCount);
        // Reset for new group (Python sets sentence = [["[CLS]"]])
        combinedTokens = [];
        combinedLength = 0;
        groupChunkCount = 0;
      }

      // Add chunk to current group AFTER the split check
      // Python: cur_length += sent_len (line 176), then sentence.extend(sent) (line 179)
      // Reference: booknlp/english/entity_tagger.py:176-179
      combinedLength += chunkLength;
      groupChunkCount += 1;
      combinedTokens.push(...chunk);
    }

    // Final append: match Python's check for len(sentence) > 1
    // Python  sentence starts as [["[CLS]"]], so len > 1 means chunks were added
    // TypeScript combinedTokens starts empty, so length > 0 means chunks were added with tokens
    // BUT: we need to match Python's behavior of counting CHUNKS, not tokens
    // Reference: booknlp/english/entity_tagger.py:182-187
    if (groupChunkCount > 0) {
      batches.push(combinedTokens);
      secondPhaseGroupLengths.push(combinedLength);
      secondPhaseGroupChunkCounts.push(groupChunkCount);
    }

    return {
      batches,
      firstPhaseCount,
      chunkWordpieceLengths,
      chunkWordpieceSamples,
      sentsSample,
      secondPhaseGroupLengths,
      secondPhaseGroupChunkCounts,
    };
  }

  private estimateTokenLength(token: Token): number {
    return this.tokenizer.countTokens(token.text);
  }

  private estimateSentenceLength(tokens: Token[]): number {
    const tokenization = this.tokenizer.tokenizeTokens(tokens);
    return tokenization.tokenIds.length;
  }

  private calculateChunkWordpieceLength(tokens: Token[]): number {
    // Match Python phase 2 logic: iterate through tokens and sum wordpiece counts
    // Python: for toks in sent: sent_len += len(toks)
    // Reference: booknlp/english/entity_tagger.py:151-153
    let totalLength = 0;
    for (const token of tokens) {
      totalLength += this.estimateTokenLength(token);
    }
    return totalLength;
  }

  private async processSentenceBatch(
    batches: Array<Token[]>,
  ): Promise<{
    entities?: EntityAnnotation[];
    supersense?: SupersenseAnnotation[];
    events?: Set<number>;
  }> {
    const maxSequenceLength = 500;

    // Process each batch separately, each batch already concatenates sentences up to max length
    // CRITICAL FIX: Process all batches together in ONE ONNX call, not individually
    // Python calls tag_all ONCE with all batched data
    // Reference: booknlp/english/tagger.py:208-214
    const allEntities: EntityAnnotation[] = [];
    const allSupersense: SupersenseAnnotation[] = [];
    const allEvents = new Set<number>();
    const singleBatchResults = await this.processSingleBatch(
      batches,
      maxSequenceLength
    );

    if (singleBatchResults.entities) {
        allEntities.push(...singleBatchResults.entities);
    }
    if (singleBatchResults.supersense) {
        allSupersense.push(...singleBatchResults.supersense);
    }
    if (singleBatchResults.events) {
        singleBatchResults.events.forEach(tokenId => allEvents.add(tokenId));
    }

    return {
      entities: allEntities,
      supersense: allSupersense,
      events: allEvents,
    };
  }

  private async processSingleBatch(
    batches: Array<Token[]>,
    maxSequenceLength: number
  ): Promise<{
    entities?: EntityAnnotation[];
    supersense?: SupersenseAnnotation[];
    events?: Set<number>;
  }> {

    const allInputIds: number[][] = [];
    const allAttentionMask: number[][] = [];
    const allTransformMatrices: number[][][] = [];
    const allSubwordToTokenMap: Array<Array<number | null>> = [];
    const allWordnetSenses: number[][] = [];
    const tokenCounts: number[] = [];



    let maxSeqLen = 0;
    let maxOriginalTokens = 0;

    for (const batch of batches) {
      const tokenization = this.tokenizer.tokenizeTokens(batch);
      const inputIds = [...tokenization.tokenIds];
      const attentionMask = [...tokenization.attentionMask];
      const transformMatrix = [...tokenization.transformMatrix];
      const subwordToTokenMap = tokenization.subwordToTokenMap ? [...tokenization.subwordToTokenMap] : [];
      const wnSenses = this.buildWordNetSenses(batch);

      if (inputIds.length > maxSequenceLength) {
        inputIds.splice(maxSequenceLength);
        attentionMask.splice(maxSequenceLength);
        transformMatrix.splice(maxSequenceLength);
        for (let j = 0; j < transformMatrix.length; j++) {
          if (transformMatrix[j].length > maxSequenceLength) {
            transformMatrix[j].splice(maxSequenceLength);
          }
        }
        wnSenses.splice(maxSequenceLength);
      }

      const seqLen = inputIds.length;
      const originalTokenCount = batch.length + 2;

      maxSeqLen = Math.max(maxSeqLen, seqLen);
      maxOriginalTokens = Math.max(maxOriginalTokens, originalTokenCount);

      allInputIds.push(inputIds);
      allAttentionMask.push(attentionMask);
      allTransformMatrices.push(transformMatrix);
      allSubwordToTokenMap.push(subwordToTokenMap);
      allWordnetSenses.push(wnSenses);
      tokenCounts.push(batch.length);
    }

    const paddedWordpieceLen = Math.ceil(maxSeqLen / 8) * 8;
    const paddedOriginalLen = Math.ceil(maxOriginalTokens / 8) * 8;
    const paddedTagSeqLen = paddedOriginalLen - 1;

    for (let i = 0; i < allInputIds.length; i++) {
      while (allInputIds[i].length < paddedWordpieceLen) {
        allInputIds[i].push(0);
        allAttentionMask[i].push(0);
      }

      while (allWordnetSenses[i].length < paddedOriginalLen) {
        allWordnetSenses[i].push(0);
      }

      while (allTransformMatrices[i].length < paddedOriginalLen) {
        allTransformMatrices[i].push(new Array(paddedWordpieceLen).fill(0));
      }

      for (let j = 0; j < allTransformMatrices[i].length; j++) {
        while (allTransformMatrices[i][j].length < paddedWordpieceLen) {
          allTransformMatrices[i][j].push(0);
        }
      }

    }

    const identityMatrix = this.createIdentityMatrix(paddedOriginalLen, paddedOriginalLen);
    const identityMatrices = batches.map(() => identityMatrix);

    const wnBatchShape: [number, number] = [allWordnetSenses.length, paddedOriginalLen];

    const logitsPass1 = await this.controller.predict(
      allInputIds,
      allAttentionMask,
      allTransformMatrices,
      identityMatrices,
      identityMatrices,
      allWordnetSenses,
      [],
      paddedOriginalLen,
    );

    const entityLogits1 = logitsPass1.entityLogits1;
    const supersenseLogits = logitsPass1.supersenseLogits;
    const eventLogits = logitsPass1.eventLogits;

    if (!entityLogits1) {
      return {
        entities: [],
        supersense: [],
        events: new Set<number>(),
      };
    }

    const layer1Transforms: Array<{ matrix: number[][]; missing: number[]; len: number; tags: string[] }> = [];
    const layer2Transforms: Array<{ matrix: number[][]; missing: number[]; len: number; tags: string[] }> = [];

    for (let i = 0; i < batches.length; i++) {
      const tokenCount = tokenCounts[i];

      const entityViterbi1 = this.crfDecoder.decodeEntity(
        [entityLogits1[i]],
        [tokenCount]
      );
      const entities1Indices = entityViterbi1.paths[0];
      const layer1Transform = this.advancedPostProcessor.computeLayerTransformationFromIndices(
        entities1Indices,
        paddedTagSeqLen
      );
      layer1Transforms.push(layer1Transform);

      // per-batch debug collection removed
    }

    const layer1Matrices = layer1Transforms.map((transform) =>
      this.embedMatrix(transform.matrix, paddedOriginalLen)
    );

    const logitsPass2 = await this.controller.predict(
      allInputIds,
      allAttentionMask,
      allTransformMatrices,
      layer1Matrices,
      identityMatrices,
      allWordnetSenses,
      [],
      paddedOriginalLen
    );

    const entityLogits2 = logitsPass2.entityLogits2;

    if (!entityLogits2) {
      return {
        entities: [],
        supersense: [],
        events: new Set<number>(),
      };
    }

    for (let i = 0; i < batches.length; i++) {
      const layer1Transform = layer1Transforms[i];

      const entityViterbi2 = this.crfDecoder.decodeEntity(
        [entityLogits2[i]],
        [layer1Transform.len]
      );
      const entities2IndicesRaw = entityViterbi2.paths[0];
      const layer2Transform = this.advancedPostProcessor.computeLayerTransformationFromIndices(
        entities2IndicesRaw,
        paddedTagSeqLen
      );
      layer2Transforms.push(layer2Transform);

      // per-batch debug collection removed
    }

    const layer2Matrices = layer2Transforms.map((transform) =>
      this.embedMatrix(transform.matrix, paddedOriginalLen)
    );

    const logitsPass3 = await this.controller.predict(
      allInputIds,
      allAttentionMask,
      allTransformMatrices,
      layer1Matrices,
      layer2Matrices,
      allWordnetSenses,
      [],
      paddedOriginalLen
    );

    const entityLogits3 = logitsPass3.entityLogits3;

    if (!entityLogits3) {
      return {
        entities: [],
        supersense: [],
        events: new Set<number>(),
      };
    }

    const allEntities: EntityAnnotation[] = [];
    const allSupersense: SupersenseAnnotation[] = [];
    const allEvents = new Set<number>();

    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      const tokenCount = tokenCounts[i];
      const layer1Transform = layer1Transforms[i];
      const layer2Transform = layer2Transforms[i];

      // Process entity logits with CRF decoding for 3-layer hierarchical NER
      if (entityLogits1 && entityLogits2 && entityLogits3) {
        const entityViterbi1 = this.crfDecoder.decodeEntity(
          [entityLogits1[i]],
          [tokenCount]
        );
        const entities1Indices = entityViterbi1.paths[0].slice(0, tokenCount);

        const entityViterbi2 = this.crfDecoder.decodeEntity(
          [entityLogits2[i]],
          [tokenCount]
        );
        const entities2IndicesRaw = entityViterbi2.paths[0].slice(0, layer1Transform.len);
        const entities2TagsFixed = this.advancedPostProcessor.fixBIOTags(
          this.advancedPostProcessor.convertIndicesToTags(entities2IndicesRaw, this.entityTagset)
        );
        let entities2Indices = this.advancedPostProcessor.convertTagsToIndices(entities2TagsFixed);

        const entityViterbi3 = this.crfDecoder.decodeEntity(
          [entityLogits3[i]],
          [layer2Transform.len]
        );
        const entities3IndicesRaw = entityViterbi3.paths[0];
        const entities3TagsFixed = this.advancedPostProcessor.fixBIOTags(
          this.advancedPostProcessor.convertIndicesToTags(entities3IndicesRaw, this.entityTagset)
        );
        let entities3Indices = this.advancedPostProcessor.convertTagsToIndices(entities3TagsFixed);

        entities3Indices = this.advancedPostProcessor.restoreCompressedTokens(
          entities3Indices,
          layer2Transform.missing
        );
        entities3Indices = this.advancedPostProcessor.restoreCompressedTokens(
          entities3Indices,
          layer1Transform.missing
        );

        entities2Indices = this.advancedPostProcessor.restoreCompressedTokens(
          entities2Indices,
          layer1Transform.missing
        );

        entities2Indices = entities2Indices.slice(0, tokenCount);
        entities3Indices = entities3Indices.slice(0, tokenCount);

        const entities2Tags = this.advancedPostProcessor.convertIndicesToTags(
          entities2Indices,
          this.entityTagset
        ).slice(0, tokenCount);

        const entities3Tags = this.advancedPostProcessor.convertIndicesToTags(
          entities3Indices,
          this.entityTagset
        ).slice(0, tokenCount);

        const entities1TagsFixed = layer1Transform.tags.slice(0, tokenCount);

        const entities1ForExtraction = entities1TagsFixed.slice(0, tokenCount);
        const entities2ForExtraction = entities2Tags.slice(0, tokenCount);
        const entities3ForExtraction = entities3Tags.slice(0, tokenCount);

        // Extract entity spans from each layer's BIO tags.
        // Python passes length = original_tokens + 2 and slices tags[:length - 2],
        // which equals the original token count. Our tag sequences already align
        // to original tokens, so no extra truncation is needed.
        const entities1 = this.advancedPostProcessor.extractEntitiesFromBIO(entities1ForExtraction, batch);
        const entities2 = this.advancedPostProcessor.extractEntitiesFromBIO(entities2ForExtraction, batch);
        const entities3 = this.advancedPostProcessor.extractEntitiesFromBIO(entities3ForExtraction, batch);

        // Merge entities across 3 layers, deduplicating spans and keeping best predictions
        const mergedEntities = this.advancedPostProcessor.mergeEntityLayers(entities1, entities2, entities3);

        const typedEntities = this.advancedPostProcessor.assignEntityTypes(mergedEntities, this.entityCategoryMap);

        // entity debug logging removed

        typedEntities.forEach(entity => {
          const startTokenId = batch[entity.startToken]?.tokenId ?? entity.startToken;
          const endTokenIndex = entity.endToken - 1;
          const rawEndTokenId = batch[endTokenIndex]?.tokenId;
          const endTokenId = rawEndTokenId === -2 || rawEndTokenId === undefined
            ? startTokenId
            : rawEndTokenId;
          allEntities.push({
            startToken: startTokenId,
            endToken: endTokenId,
            cat: entity.cat,
            text: entity.text,
            prop: entity.prop,
            coref: -1 // Placeholder for coreference cluster ID, to be filled in later
          });
        });
      }

      // Process supersense logits with CRF decoding for semantic role labels
      if (supersenseLogits) {
        const supersenseViterbi = this.crfDecoder.decodeSupersense(
          [supersenseLogits[i]],
          [tokenCount]
        );
        const supersenseIndices = supersenseViterbi.paths[0].slice(0, tokenCount);
        const supersenseTags = this.advancedPostProcessor.convertIndicesToTags(
          supersenseIndices,
          this.supersenseTagset
        );
        const fixedSupersense = this.advancedPostProcessor.fixBIOTags(supersenseTags);

        // supersense debug logging removed
        const supersenseAnnotations = this.advancedPostProcessor.applySupersenseAnnotations(
          batch,
          fixedSupersense
        );

        supersenseAnnotations.forEach(annotation => {
          const startIndex = annotation[0];
          const endIndex = annotation[1] - 1;
          const startTokenId = batch[startIndex]?.tokenId ?? startIndex;
          const rawEndTokenId = batch[endIndex]?.tokenId;
          const endTokenId = rawEndTokenId === -2 || rawEndTokenId === undefined
            ? startTokenId
            : rawEndTokenId;
          allSupersense.push([
            startTokenId,
            endTokenId,
            annotation[2],
            annotation[3],
          ]);
        });
      }

      // Process event logits via argmax for binary event classification (non-event vs event)
      if (eventLogits) {
        const eventLogitsBatch = eventLogits[i];
        for (let tokenIdx = 0; tokenIdx <= tokenCount && tokenIdx < eventLogitsBatch.length; tokenIdx++) {
          const logitPair = eventLogitsBatch[tokenIdx];
          if (logitPair && logitPair.length >= 2) {
            const [logit0, logit1] = logitPair;
            if (logit1 > logit0 && tokenIdx < tokenCount) {
              const token = batch[tokenIdx];
              if (token) {
                allEvents.add(token.tokenId);
              }
            }
          }
        }
      }
    }

    return {
      entities: allEntities,
      supersense: allSupersense,
      events: allEvents,
    };
  }

  private buildWordNetSenses(
    tokens: Token[],
  ): number[] {
    const wnSenses: number[] = [0];

    for (const token of tokens) {
      wnSenses.push(this.getWordNetSense(token));
    }

    wnSenses.push(0);

    return wnSenses;
  }

  private createIdentityMatrix(rows: number, cols: number): number[][] {
    const matrix: number[][] = [];
    for (let i = 0; i < rows; i++) {
      const row = new Array(cols).fill(0);
      if (i < cols) {
        row[i] = 1;
      }
      matrix.push(row);
    }
    return matrix;
  }

  private embedMatrix(baseMatrix: number[][], paddedLen: number): number[][] {
    const full = this.createIdentityMatrix(paddedLen, paddedLen).map(row => row.map(() => 0));
    for (let i = 0; i < baseMatrix.length; i++) {
      const row = baseMatrix[i];
      for (let j = 0; j < row.length; j++) {
        full[i + 1][j + 1] = row[j];
      }
    }
    return full;
  }

  private getReverseTagset(tagset: Map<number, string>): Map<string, number> {
    const reverseMap = new Map<string, number>();
    tagset.forEach((tag, id) => {
      reverseMap.set(tag, id);
    });
    return reverseMap;
  }
}
