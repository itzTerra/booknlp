import { SpaCyToken, Token, EntityAnnotation, SupersenseAnnotation, ResourceUrls, ExecutionProvider } from 'types';
import { ONNXTaggerController } from 'tagger-controller';
import { Tokenizer } from 'preprocessing';
import { CRFDecoder } from 'crf-decoder';
import { AdvancedPostProcessor } from 'advanced-postprocessor';

interface WordNetSense {
  [key: string]: number;
}

export class EntityTagger {
  private controller: ONNXTaggerController;
  private advancedPostProcessor: AdvancedPostProcessor;
  private crfDecoder: CRFDecoder;
  private resourceUrls: ResourceUrls;
  private tokenizer: Tokenizer;
  private modelId: string;
  private wordNetSenses: WordNetSense = {};
  private entityTagset: Map<number, string> = new Map();
  private supersenseTagset: Map<number, string> = new Map();
  private entityCategoryMap: Map<string, string> = new Map();
  private initialized: boolean = false;

  constructor(
    modelPath: string,
    resourceUrls: ResourceUrls,
    executionProviders: ExecutionProvider[] = ['wasm'],
    wasmPaths?: string | Record<string, string>
  ) {
    this.controller = new ONNXTaggerController(modelPath, executionProviders, wasmPaths);
    this.advancedPostProcessor = new AdvancedPostProcessor();
    this.crfDecoder = new CRFDecoder();
    this.resourceUrls = resourceUrls;
    this.tokenizer = new Tokenizer();
    this.modelId = modelPath;
    this.buildEntityCategoryMap();
  }

  static async fromHuggingFace(
    repoId: string,
    resourceUrls: ResourceUrls,
    executionProviders: ExecutionProvider[] = ['wasm'],
    wasmPaths?: string | Record<string, string>
  ): Promise<EntityTagger> {
    const tagger = new EntityTagger(
      repoId,
      resourceUrls,
      executionProviders,
      wasmPaths
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
    const [entityContent, supersenseContent, wordNetContent, crfTransitionsData] = await Promise.all([
      this.fetchText(this.resourceUrls.entityTagset),
      this.fetchText(this.resourceUrls.supersenseTagset),
      this.fetchText(this.resourceUrls.wordNet),
      this.fetchJson(this.resourceUrls.crfTransitions),
    ]);

    this.loadTagsets(entityContent, supersenseContent);
    this.loadWordNetSenses(wordNetContent);
    // Initialize CRF decoder with transition matrices
    // Transitions define penalty scores for invalid tag sequences (e.g., I-PER after O)
    this.crfDecoder.loadTransitions(crfTransitionsData);
  }

  private async fetchText(url: string): Promise<string> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load resource: ${url}`);
    }

    return response.text();
  }

  private async fetchJson(url: string): Promise<any> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load resource: ${url}`);
    }

    return response.json();
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

  async initialize(): Promise<void> {
    await Promise.all([
      this.controller.loadModel(),
      this.loadResources(),
      this.tokenizer.initialize(this.modelId),
    ]);
    this.initialized = true;
  }

  private getWordNetSense(token: SpaCyToken): number {
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
    spaCyTokens: SpaCyToken[],
  ): Promise<{
    entities?: EntityAnnotation[];
    supersense?: SupersenseAnnotation[];
    events?: Set<number>;
    _debug?: Record<string, any>;
  }> {
    if (!this.initialized) {
      throw new Error('EntityTagger not initialized. Call initialize() first.');
    }

    const maxBatchSize = 32;
    const filteredSpaCyTokens = spaCyTokens.filter((token) => token.text.trim().length > 0);

    if (filteredSpaCyTokens.length !== tokens.length) {
      throw new Error('SpaCy token count does not match BookNLP token count after filtering.');
    }

    const {
      batches: sentenceBatches,
      firstPhaseCount: firstPhaseChunks,
      secondPhaseGroupLengths,
      secondPhaseGroupChunkCounts,
      chunkWordpieceLengths,
    } = this.createSentenceBatches(tokens, filteredSpaCyTokens);

    // Match Python: get_batches sorts by BERT token length before batching.
    // Reference: booknlp/common/layered_reader.py:107-140
    const sentenceLengths = sentenceBatches.map((batch) =>
      this.estimateSentenceLength(batch.spaCyTokens)
    );
    const orderedSentences = sentenceBatches
      .map((batch, index) => ({
        batch,
        length: sentenceLengths[index],
        index,
      }))
      .sort((a, b) => a.length - b.length);

    // Debug: Track sentence grouping (corresponds to Python's len(sentences))
    const preGroupCount = sentenceBatches.length;
    const postSortCount = orderedSentences.length;
    const secondPhaseGroups = sentenceBatches.length;

    const allEntityAnnotations: EntityAnnotation[] = [];
    const allSupersenseAnnotations: SupersenseAnnotation[] = [];
    const allEvents = new Set<number>();
    const debugWordNetBatches: any[] = [];
    const debugWordNetBatchShapes: Array<[number, number]> = [];

    let batchCount = 0;
    let batchIdx = 0;
    let currentBatchSize = maxBatchSize;

    // Debug: Track batching behavior to match Python's adaptive batching
    // Python adaptively reduces batch size when sequence length exceeds thresholds
    // Reference: booknlp/common/layered_reader.py:318-325
    let debugBatchSizes: number[] = [];

    while (batchIdx < orderedSentences.length) {
      const batchSizeUsed = currentBatchSize;
      const batchEnd = Math.min(batchIdx + batchSizeUsed, orderedSentences.length);
      const batchSlice = orderedSentences.slice(batchIdx, batchEnd);
      const batchMaxLen = Math.max(...batchSlice.map(item => item.length));
      const batchInputs = batchSlice.map(item => item.batch);

      // Debug: Record actual batch size used
      debugBatchSizes.push(batchSlice.length);

      // Debug: Collect WordNet sense batch information ONCE per ONNX inference batch
      // Match Python: _get_wn() creates one wn_batch per ONNX batch (not per phase-2 group)
      // Python iterates through batched_pos (74 items for 470 sentences), creating 74 wn_batches
      // Reference: booknlp/english/entity_tagger.py:49-89, booknlp/english/entity_tagger.py:222
      let batchTokenCount = 0;
      let batchSpaCyTokenCount = 0;
      let maxOriginalTokens = 0;

      for (const batch of batchInputs) {
        batchTokenCount += batch.tokens.length;
        batchSpaCyTokenCount += batch.spaCyTokens.length;
        // Calculate max original token count (before BERT tokenization) across all sentences
        // Matches processSingleBatch: originalTokenCount = batch.tokens.length + 2 ([CLS] + tokens + [SEP])
        // Reference: booknlp-ts/src/entity-tagger.ts:579
        const originalTokenCount = batch.spaCyTokens.length + 2;
        maxOriginalTokens = Math.max(maxOriginalTokens, originalTokenCount);
      }

      const wnSensesLength = maxOriginalTokens;
      debugWordNetBatches.push({
        batchTokenCount,
        batchSpaCyTokenCount,
        wnSensesLength,
      });

      // Match Python: wn_batches_shapes uses unpadded max sentence length from _get_wn
      // Reference: booknlp/english/entity_tagger.py:49-89
      debugWordNetBatchShapes.push([batchInputs.length, wnSensesLength]);

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

    // Debug info to match Python's debug output structure
    // Python's batches_count is len(batched_sents), which is the number of inference batches
    // Reference: booknlp/english/entity_tagger.py:256
    const debugInfo: Record<string, any> = {
      batches_count: batchCount,
      chunk_wordpiece_lengths: chunkWordpieceLengths,
      first_phase_chunks: firstPhaseChunks,
      second_phase_groups: secondPhaseGroups,
      sentence_groups_before_sort: preGroupCount,
      sentence_chunks_count: firstPhaseChunks,  // First phase chunks, not second phase groups
      ordered_sentences_count: postSortCount,
      sentence_lengths: sentenceLengths,
      ordering: orderedSentences.map((entry) => entry.index),
      ordered_sentence_lengths: orderedSentences.map((entry) => entry.length),
      second_phase_group_lengths: secondPhaseGroupLengths,
      second_phase_group_chunk_counts: secondPhaseGroupChunkCounts,
      extracted_entities_count: allEntityAnnotations.length,
      extracted_supersense_count: allSupersenseAnnotations.length,
      wn_batches_count: debugWordNetBatches.length,
      wn_batches_shapes: debugWordNetBatchShapes,
      wn_batches_details: debugWordNetBatches,
      debug_batch_sizes: debugBatchSizes,
      debug_batch_sizes_sum: debugBatchSizes.reduce((a, b) => a + b, 0),
      // Debug: Supersense annotations at problematic positions
      supersense_debug_positions: allSupersenseAnnotations
        .filter(ann => {
          const tokenId = ann[0];
          return [
            26968, 26969, 26979, 27033, 27049, 27186, 27266, 27286, 30357, 34811
          ].includes(tokenId) ||
          (tokenId >= 26960 && tokenId <= 26990) ||
          (tokenId >= 27030 && tokenId <= 27060) ||
          (tokenId >= 27180 && tokenId <= 27300) ||
          (tokenId >= 30350 && tokenId <= 30365) ||
          (tokenId >= 34805 && tokenId <= 34820);
        })
        .map(ann => ({ start: ann[0], end: ann[1], category: ann[2], text: ann[3] })),
    };

    return {
      entities: allEntityAnnotations,
      supersense: allSupersenseAnnotations,
      events: allEvents,
      _debug: debugInfo,
    };
  }

  private createSentenceBatches(
    tokens: Token[],
    spaCyTokens: SpaCyToken[]
  ): {
    batches: Array<{ tokens: Token[]; spaCyTokens: SpaCyToken[] }>;
    firstPhaseCount: number;
    secondPhaseGroupLengths: number[];
    secondPhaseGroupChunkCounts: number[];
    chunkWordpieceLengths: number[];
  } {
    const maxSequenceLength = 500;
    const sentenceChunks: Array<{ tokens: Token[]; spaCyTokens: SpaCyToken[] }> = [];

    let currentSentenceTokens: Token[] = [];
    let currentSentenceSpaCy: SpaCyToken[] = [];
    let currentLength = 0;
    let lastSentenceId = -1;
    let splitCount = 0;

    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      const spaCyToken = spaCyTokens[i];
      const tokenLength = this.estimateTokenLength(spaCyToken);
      const sentenceChanged = lastSentenceId !== -1 && token.sentenceId !== lastSentenceId;

      // Match Python split condition when sentences exceed max length.
      // Reference: booknlp/english/entity_tagger.py:108-114
      if ((sentenceChanged || currentLength + tokenLength > maxSequenceLength) && currentSentenceTokens.length > 0) {
        sentenceChunks.push({
          tokens: currentSentenceTokens,
          spaCyTokens: currentSentenceSpaCy,
        });
        splitCount++;
        currentSentenceTokens = [];
        currentSentenceSpaCy = [];
        currentLength = 0;
      }

      currentSentenceTokens.push(token);
      currentSentenceSpaCy.push(spaCyToken);
      // tokenLength is already an integer from countTokens, no Math.floor needed
      currentLength += tokenLength;
      lastSentenceId = token.sentenceId;
    }

    if (currentSentenceTokens.length > 0) {
      sentenceChunks.push({
        tokens: currentSentenceTokens,
        spaCyTokens: currentSentenceSpaCy,
      });
    }
    // Debug: sentenceChunks.length should equal splitCount + 1 (splits + final append)
    // console.log(`DEBUG TS: Created ${sentenceChunks.length} sentence chunks after ${splitCount} splits`);
    const firstPhaseCount = sentenceChunks.length;
    // Phase 2: Group phase-1 chunks into batches, recalculating wordpiece lengths
    // Python does this in entity_tagger.py:150-187, recalculating lengths fresh for each chunk
    // Reference: booknlp/english/entity_tagger.py:151-153
    const chunkWordpieceLengths: number[] = [];

    // Batches for ONNX inference - don't need wordpieceLength after phase 2 grouping
    const batches: Array<{ tokens: Token[]; spaCyTokens: SpaCyToken[] }> = [];
    let combinedTokens: Token[] = [];
    let combinedSpaCy: SpaCyToken[] = [];
    let combinedLength = 0;
    let groupChunkCount = 0;
    const secondPhaseGroupLengths: number[] = [];
    const secondPhaseGroupChunkCounts: number[] = [];

    for (const chunk of sentenceChunks) {
      // Recalculate chunk wordpiece length in phase 2, exactly like Python does
      // Python doesn't store lengths from phase 1; it calculates them fresh when grouping
      // Reference: booknlp/english/entity_tagger.py:159-162
      const chunkLength = this.calculateChunkWordpieceLength(chunk.spaCyTokens);
      // Store for debug output
      chunkWordpieceLengths.push(chunkLength);

      // Debug: Log when chunk wordpiece length differs from expected Python values
      // Known Python values for indices 9, 20, 36-37, 84, 113...
      const chunkIdx = chunkWordpieceLengths.length - 1;
      if (chunkIdx < 150) {
        // Log details of the chunk for the first significant mismatches
        const pythonExpected: Record<number, number> = {
          9: 26,  // TS has 30 (+4)
          20: 80, // TS has 84 (+4)
          36: 57, // TS has 61 (+4)
          37: 42, // TS has 50 (+8)
          84: 66, // TS has 70 (+4)
          113: 21, // TS has 25 (+4)
          130: 36, // TS has 44 (+8)
        };

        if (pythonExpected[chunkIdx] && chunkLength !== pythonExpected[chunkIdx]) {
          console.log(`CHUNK ${chunkIdx} LENGTH MISMATCH: Expected=${pythonExpected[chunkIdx]}, Got=${chunkLength}, Diff=${chunkLength - pythonExpected[chunkIdx]}`);
          console.log(`  Chunk has ${chunk.spaCyTokens.length} spaCy tokens`);

          // Show token counts for ALL tokens
          const tokenCounts = chunk.spaCyTokens.map((t, i) => ({
            idx: i,
            text: t.text,
            count: this.tokenizer.countTokens(t.text)
          }));
          console.log(`  All ${tokenCounts.length} token counts: ${JSON.stringify(tokenCounts)}`);
        }
      }

      // Match Python: check if adding this chunk would exceed limit
      // Python: if sent_len + cur_length >= max_sentence_length
      // Reference: booknlp/english/entity_tagger.py:164
      if (chunkLength + combinedLength >= maxSequenceLength) {
        // Emit current group before adding this chunk
        // Python always appends sentence here, even if just [CLS][SEP]
        // Reference: booknlp/english/entity_tagger.py:165-171
        batches.push({
          tokens: combinedTokens,
          spaCyTokens: combinedSpaCy,
        });
        secondPhaseGroupLengths.push(combinedLength);
        secondPhaseGroupChunkCounts.push(groupChunkCount);
        // Reset for new group (Python sets sentence = [["[CLS]"]])
        combinedTokens = [];
        combinedSpaCy = [];
        combinedLength = 0;
        groupChunkCount = 0;
      }

      // Add chunk to current group AFTER the split check
      // Python: cur_length += sent_len (line 176), then sentence.extend(sent) (line 179)
      // Reference: booknlp/english/entity_tagger.py:176-179
      combinedLength += chunkLength;
      groupChunkCount += 1;
      combinedTokens.push(...chunk.tokens);
      combinedSpaCy.push(...chunk.spaCyTokens);
    }

    // Final append: match Python's check for len(sentence) > 1
    // Python  sentence starts as [["[CLS]"]], so len > 1 means chunks were added
    // TypeScript combinedTokens starts empty, so length > 0 means chunks were added with tokens
    // BUT: we need to match Python's behavior of counting CHUNKS, not tokens
    // Reference: booknlp/english/entity_tagger.py:182-187
    if (groupChunkCount > 0) {
      batches.push({
        tokens: combinedTokens,
        spaCyTokens: combinedSpaCy,
      });
      secondPhaseGroupLengths.push(combinedLength);
      secondPhaseGroupChunkCounts.push(groupChunkCount);
    }

    return {
      batches,
      firstPhaseCount,
      chunkWordpieceLengths,
      secondPhaseGroupLengths,
      secondPhaseGroupChunkCounts,
    };
  }

  private estimateTokenLength(token: SpaCyToken): number {
    return this.tokenizer.countTokens(token.text);
  }

  private estimateSentenceLength(spaCyTokens: SpaCyToken[]): number {
    const tokenization = this.tokenizer.tokenizeTokens(spaCyTokens);
    return tokenization.tokenIds.length;
  }

  private calculateChunkWordpieceLength(spaCyTokens: SpaCyToken[]): number {
    // Match Python phase 2 logic: iterate through tokens and sum wordpiece counts
    // Python: for toks in sent: sent_len += len(toks)
    // Reference: booknlp/english/entity_tagger.py:151-153
    let totalLength = 0;
    for (const token of spaCyTokens) {
      totalLength += this.estimateTokenLength(token);
    }
    return totalLength;
  }

  private async processSentenceBatch(
    batches: Array<{ tokens: Token[]; spaCyTokens: SpaCyToken[] }>,
  ): Promise<{
    entities?: EntityAnnotation[];
    supersense?: SupersenseAnnotation[];
    events?: Set<number>;
    _debug?: {
      wnBatchShape?: [number, number];
    };
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
      _debug: singleBatchResults.wnBatchShape
        ? { wnBatchShape: singleBatchResults.wnBatchShape }
        : undefined,
    };
  }

  private async processSingleBatch(
    batches: Array<{ tokens: Token[]; spaCyTokens: SpaCyToken[] }>,
    maxSequenceLength: number
  ): Promise<{
    entities?: EntityAnnotation[];
    supersense?: SupersenseAnnotation[];
    events?: Set<number>;
    wnBatchShape?: [number, number];
  }> {

    const allInputIds: number[][] = [];
    const allAttentionMask: number[][] = [];
    const allTransformMatrices: number[][][] = [];
    const allWordnetSenses: number[][] = [];
    const tokenCounts: number[] = [];

    let maxSeqLen = 0;
    let maxOriginalTokens = 0;

    for (const batch of batches) {
      const tokenization = this.tokenizer.tokenizeTokens(batch.spaCyTokens);
      const inputIds = [...tokenization.tokenIds];
      const attentionMask = [...tokenization.attentionMask];
      const transformMatrix = [...tokenization.transformMatrix];
      const wnSenses = this.buildWordNetSenses(batch.spaCyTokens);

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
      const originalTokenCount = batch.tokens.length + 2;

      maxSeqLen = Math.max(maxSeqLen, seqLen);
      maxOriginalTokens = Math.max(maxOriginalTokens, originalTokenCount);

      allInputIds.push(inputIds);
      allAttentionMask.push(attentionMask);
      allTransformMatrices.push(transformMatrix);
      allWordnetSenses.push(wnSenses);
      tokenCounts.push(batch.tokens.length);
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
        wnBatchShape,
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
      const batchTokens = batch.tokens;
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
        const entities1 = this.advancedPostProcessor.extractEntitiesFromBIO(entities1ForExtraction, batchTokens);
        const entities2 = this.advancedPostProcessor.extractEntitiesFromBIO(entities2ForExtraction, batchTokens);
        const entities3 = this.advancedPostProcessor.extractEntitiesFromBIO(entities3ForExtraction, batchTokens);

        // Merge entities across 3 layers, deduplicating spans and keeping best predictions
        const mergedEntities = this.advancedPostProcessor.mergeEntityLayers(entities1, entities2, entities3);

        const typedEntities = this.advancedPostProcessor.assignEntityTypes(mergedEntities, this.entityCategoryMap);

        typedEntities.forEach(entity => {
          const startTokenId = batchTokens[entity.startToken]?.tokenId ?? entity.startToken;
          const endTokenIndex = entity.endToken - 1;
          const rawEndTokenId = batchTokens[endTokenIndex]?.tokenId;
          const endTokenId = rawEndTokenId === -2 || rawEndTokenId === undefined
            ? startTokenId
            : rawEndTokenId;
          allEntities.push({
            ...entity,
            startToken: startTokenId,
            endToken: endTokenId,
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

        // Debug: Log tag sequences for problematic token ranges
        // Reference: Python's _get_spans in tagger.py:598-631
        const debugRanges = [
          { start: 26960, end: 26990, name: 'Range 26960-26990 (verb.motion/stative issue)' },
          { start: 27030, end: 27060, name: 'Range 27030-27060 (noun.person span issue)' },
          { start: 27180, end: 27300, name: 'Range 27180-27300 (noun.communication/substance/attribute issues)' },
          { start: 30350, end: 30365, name: 'Range 30350-30365 (noun.event issue)' },
          { start: 34805, end: 34820, name: 'Range 34805-34820 (verb.cognition/possession issue)' },
        ];

        for (const range of debugRanges) {
          const globalStartToken = batchTokens[0]?.tokenId ?? 0;
          const globalEndToken = batchTokens[tokenCount - 1]?.tokenId ?? 0;

          if (globalStartToken <= range.end && globalEndToken >= range.start) {
            const localStart = Math.max(0, range.start - globalStartToken);
            const localEnd = Math.min(tokenCount, range.end - globalStartToken + 1);

            if (localStart < tokenCount && localEnd > 0) {
              console.log(`\n[DEBUG Supersense] ${range.name}`);
              console.log(`  Batch tokens ${globalStartToken} to ${globalEndToken}`);
              console.log(`  Local indices ${localStart} to ${localEnd}`);
              console.log(`  Raw tags: ${supersenseTags.slice(localStart, localEnd).join(', ')}`);
              console.log(`  Fixed tags: ${fixedSupersense.slice(localStart, localEnd).join(', ')}`);
              console.log(`  Token IDs: ${batchTokens.slice(localStart, localEnd).map(t => t.tokenId).join(', ')}`);
              console.log(`  Token texts: ${batchTokens.slice(localStart, localEnd).map(t => t.text).join(' ')}`);
            }
          }
        }

        const supersenseAnnotations = this.advancedPostProcessor.applySupersenseAnnotations(
          batchTokens,
          fixedSupersense
        );

        supersenseAnnotations.forEach(annotation => {
          const startIndex = annotation[0];
          const endIndex = annotation[1] - 1;
          const startTokenId = batchTokens[startIndex]?.tokenId ?? startIndex;
          const rawEndTokenId = batchTokens[endIndex]?.tokenId;
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
      // Event logits are computed for original tokens after removing [CLS]
      // These correspond to: [token0, token1, ..., tokenN-1, SEP]
      // Python code: for col in range(batched_orig_token_lens[b][row] - 1)
      // where batched_orig_token_lens[b][row] includes [CLS], tokens, and [SEP]
      // So for N original tokens: range(N + 2 - 1) = range(N + 1), processing 0 to N (inclusive)
      if (eventLogits) {
        const eventLogitsBatch = eventLogits[i];

        // Process tokens 0 to tokenCount (inclusive), matching Python's range(tokenCount + 1) behavior
        for (let tokenIdx = 0; tokenIdx <= tokenCount && tokenIdx < eventLogitsBatch.length; tokenIdx++) {
          const logitPair = eventLogitsBatch[tokenIdx];
          if (logitPair && logitPair.length >= 2) {
            const [logit0, logit1] = logitPair;
            if (logit1 > logit0) {
              if (tokenIdx < tokenCount) {
                const token = batchTokens[tokenIdx];
                if (token) {
                  allEvents.add(token.tokenId);
                }
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
      wnBatchShape,
    };
  }

  private buildWordNetSenses(
    spaCyTokens: SpaCyToken[],
  ): number[] {
    const wnSenses: number[] = [0];

    for (const token of spaCyTokens) {
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
