import { SpaCyToken, Token, EntityAnnotation, SupersenseAnnotation, ResourceUrls as Resources, ExecutionProvider } from 'types';
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
    resourceUrls: Resources,
    executionProviders: ExecutionProvider[] = ['wasm'],
    wasmPaths?: string | Record<string, string>
  ) {
    this.controller = new ONNXTaggerController(modelPath, executionProviders, wasmPaths);
    this.advancedPostProcessor = new AdvancedPostProcessor();
    this.crfDecoder = new CRFDecoder();
    this.resources = resourceUrls;
    this.tokenizer = new Tokenizer();
    this.modelId = modelPath;
    this.buildEntityCategoryMap();
  }

  static async fromHuggingFace(
    repoId: string,
    resourceUrls: Resources,
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
    // Resources are expected to be provided as inlined contents (strings) at build time
    const entityContent = this.resources.entityTagset;
    const supersenseContent = this.resources.supersenseTagset;
    const wordNetContent = this.resources.wordNet;

    let crfTransitionsData: any;
    try {
      crfTransitionsData = typeof this.resources.crfTransitions === 'string'
        ? JSON.parse(this.resources.crfTransitions)
        : this.resources.crfTransitions;
    } catch (e) {
      throw new Error(`Failed to parse CRF transitions resource: ${(e as Error).message}`);
    }

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
      chunkWordpieceSamples,
      sentsSample,
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
    const supersenseDebugLogits: any[] = [];
    const allBatchDebugDetails: any[] = [];

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
      // Collect any debug logits produced by processSingleBatch for later inclusion
      if ((batchResults as any)._debug && (batchResults as any)._debug.supersense_debug_logits) {
        supersenseDebugLogits.push(...(batchResults as any)._debug.supersense_debug_logits);
      }
      // Collect per-batch debug details if present
      if ((batchResults as any)._debug && (batchResults as any)._debug.batch_debug_details) {
        allBatchDebugDetails.push(...(batchResults as any)._debug.batch_debug_details);
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
      chunk_wordpiece_samples: chunkWordpieceSamples,
      sents_sample: sentsSample,
      tokenizer_cap_token_id: this.tokenizer.getCapTokenId(),
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
      supersense_debug_logits: supersenseDebugLogits.length > 0 ? supersenseDebugLogits : undefined,
      batch_debug_details: allBatchDebugDetails.length > 0 ? allBatchDebugDetails : undefined,
      // Debug: Supersense annotations at problematic positions
      // Sort by start token ID to match Python output order (Python iterates through tokens)
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
        .sort((a, b) => a[0] - b[0])  // Sort by start tokenId to match Python's iteration order
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
    chunkWordpieceSamples: any[];
    sentsSample: any[];
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
    // Prepare a compact sample of the first few phase-1 sentence chunks (for parity with Python)
    const sentsSample: any[] = [];
    try {
      for (let si = 0; si < Math.min(5, sentenceChunks.length); si++) {
        const sent = sentenceChunks[si];
        const sample: any[] = [];
        for (let ti = 0; ti < sent.spaCyTokens.length; ti++) {
          const spa = sent.spaCyTokens[ti];
          const text = spa.text;
          try {
            const enc = this.tokenizer.debugEncodeToken(text);
            sample.push({
              text: text,
              prepared: enc.prepared,
              wordpieces: enc.tokens,
              wp_len: enc.ids.length,
              token_id: sent.tokens[ti]?.tokenId ?? null,
            });
          } catch (e) {
            continue;
          }
        }
        sentsSample.push(sample);
      }
    } catch (e) {
      // ignore sample construction errors
    }
    // Phase 2: Group phase-1 chunks into batches, recalculating wordpiece lengths
    // Python does this in entity_tagger.py:150-187, recalculating lengths fresh for each chunk
    // Reference: booknlp/english/entity_tagger.py:151-153
    const chunkWordpieceLengths: number[] = [];
    const chunkWordpieceSamples: any[] = [];

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
        // Build detailed token debug info for this chunk
        const tokenDetails = chunk.spaCyTokens.map((t, i) => {
          const enc = this.tokenizer.debugEncodeToken(t.text);
          const cnt = this.tokenizer.countTokens(t.text);
          return { idx: i, text: t.text, count: cnt, prepared: enc.prepared, wp_ids: enc.ids, wp_tokens: enc.tokens, wp_len: enc.ids.length };
        });

        // Store per-chunk sample for top-level debug export
        chunkWordpieceSamples.push({ chunkIdx, tokenDetails });

        if (pythonExpected[chunkIdx] && chunkLength !== pythonExpected[chunkIdx]) {
          console.log(`CHUNK ${chunkIdx} LENGTH MISMATCH: Expected=${pythonExpected[chunkIdx]}, Got=${chunkLength}, Diff=${chunkLength - pythonExpected[chunkIdx]}`);
          console.log(`  Chunk has ${chunk.spaCyTokens.length} spaCy tokens`);
          console.log(`  All ${tokenDetails.length} token details: ${JSON.stringify(tokenDetails)}`);
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
      chunkWordpieceSamples,
      sentsSample,
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
    const localSupersenseDebugLogits: any[] = [];


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

    const debugOut: Record<string, any> = {};
    if ((singleBatchResults as any).wnBatchShape) {
      debugOut.wnBatchShape = (singleBatchResults as any).wnBatchShape;
    }
    if ((singleBatchResults as any).supersense_debug_logits) {
      debugOut.supersense_debug_logits = (singleBatchResults as any).supersense_debug_logits;
    }
    if ((singleBatchResults as any).batch_debug_details) {
      debugOut.batch_debug_details = (singleBatchResults as any).batch_debug_details;
    }

    return {
      entities: allEntities,
      supersense: allSupersense,
      events: allEvents,
      _debug: Object.keys(debugOut).length > 0 ? debugOut : undefined,
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
    supersense_debug_logits?: any[];
    batch_debug_details?: any[];
    paddedWordpieceLen?: number;
    paddedOriginalLen?: number;
  }> {

    const allInputIds: number[][] = [];
    const allAttentionMask: number[][] = [];
    const allTransformMatrices: number[][][] = [];
    const allSubwordToTokenMap: Array<Array<number | null>> = [];
    const allWordnetSenses: number[][] = [];
    const tokenCounts: number[] = [];

    const localSupersenseDebugLogits: any[] = [];
    const localBatchDebugs: any[] = [];

    let maxSeqLen = 0;
    let maxOriginalTokens = 0;

    for (const batch of batches) {
      const tokenization = this.tokenizer.tokenizeTokens(batch.spaCyTokens);
      const inputIds = [...tokenization.tokenIds];
      const attentionMask = [...tokenization.attentionMask];
      const transformMatrix = [...tokenization.transformMatrix];
      const subwordToTokenMap = tokenization.subwordToTokenMap ? [...tokenization.subwordToTokenMap] : [];
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
      allSubwordToTokenMap.push(subwordToTokenMap);
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

      // Record per-batch debug info for overlapping ranges (initial snapshot including layer1 viterbi)
      try {
        const batchTokens = batches[i].tokens || [];
        const gstart = batchTokens[0]?.tokenId ?? null;
        const gend = batchTokens[batchTokens.length - 1]?.tokenId ?? null;
        const debugRanges = [
          { start: 26960, end: 26990 },
          { start: 27030, end: 27060 },
          { start: 27180, end: 27300 },
          { start: 30350, end: 30365 },
          { start: 34805, end: 34820 },
        ];
        let overlaps = false;
        for (const r of debugRanges) {
          if (gstart !== null && gend !== null && gstart <= r.end && gend >= r.start) {
            overlaps = true;
            break;
          }
        }
        if (overlaps) {
          localBatchDebugs.push({
            batch_idx: i,
            global_start_token: gstart,
            global_end_token: gend,
            input_ids: allInputIds[i].slice(),
            attention_mask: allAttentionMask[i].slice(),
            transform_matrix: allTransformMatrices[i].map(r => r.slice()),
            subword_to_token_map: allSubwordToTokenMap[i] ? allSubwordToTokenMap[i].slice() : [],
            token_count: tokenCounts[i],
            entities_layer1_indices: Array.from(entities1Indices || [] as any),
          });
        }
      } catch (e) {
        // ignore
      }
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

      try {
        const dbg = localBatchDebugs.find((d: any) => d.batch_idx === i);
        if (dbg) {
          dbg.entities_layer2_indices = Array.from(entities2IndicesRaw || [] as any);
        }
      } catch (e) {
        // ignore
      }
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

        // Debug: Log entities in problematic ranges
        const debugEntityRanges = [
          { start: 70182, end: 70216, name: 'Range 70182-70216 (PER entity issue)' },
          { start: 70331, end: 70345, name: 'Range 70331-70345 (PER entity issue)' },
        ];

        const debugEntitiesInBatch = typedEntities.filter(entity =>
          debugEntityRanges.some(range =>
            !(entity.endToken < range.start || entity.startToken > range.end)
          )
        );

        if (debugEntitiesInBatch.length > 0) {
          console.log(`\n[DEBUG Entities] Extracted ${debugEntitiesInBatch.length} entities in problem ranges:`);
          debugEntitiesInBatch.forEach(entity => {
            console.log(`    Entity: tokenIds=[${entity.startToken}-${entity.endToken}] cat="${entity.cat}" text="${entity.text}"`);
          });
        }

        typedEntities.forEach(entity => {
          const startTokenId = batchTokens[entity.startToken]?.tokenId ?? entity.startToken;
          const endTokenIndex = entity.endToken - 1;
          const rawEndTokenId = batchTokens[endTokenIndex]?.tokenId;
          const endTokenId = rawEndTokenId === -2 || rawEndTokenId === undefined
            ? startTokenId
            : rawEndTokenId;
          allEntities.push({
            startToken: startTokenId,
            endToken: endTokenId,
            cat: entity.cat,
            text: entity.text,
            prop: entity.prop,
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

        // Debug: Log comprehensive tag sequences and extraction for problematic token ranges
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
              // Dump raw supersense logits for tokens in this local window
              try {
                const logitsForBatch = supersenseLogits[i];
                if (logitsForBatch && logitsForBatch.length > 0) {
                  console.log(`  [Supersense Logits - TypeScript] Logging tokens ${localStart}..${localEnd - 1}`);
                  for (let li = localStart; li < localEnd; li++) {
                    const token = batchTokens[li];
                    const tokenId = token?.tokenId ?? li;
                    const text = token?.text ?? '[UNKNOWN]';
                    const logitsRow = logitsForBatch[li] || null;
                    console.log(`    Token ${tokenId} "${text}" logits: ${JSON.stringify(logitsRow)}`);
                    try {
                      localSupersenseDebugLogits.push({
                        tokenId: tokenId,
                        text: text,
                        local_idx: li,
                        logits: logitsRow,
                      });
                    } catch (e) {
                      // ignore push errors
                    }
                  }
                }
              } catch (e) {
                // ignore logging errors
              }

              // Log extraction details: which B- tags create spans
              console.log(`  [Extraction Details - B-tag spans only]`);
              let bTagCount = 0;
              for (let idx = localStart; idx < localEnd; idx++) {
                const tag = fixedSupersense[idx];
                const tokenId = batchTokens[idx]?.tokenId ?? idx;
                const text = batchTokens[idx]?.text ?? '[UNKNOWN]';

                if (tag.startsWith('B-')) {
                  const [, category] = tag.split('-');
                  let endIdx = idx + 1;

                  // Extend span with matching I- tags
                  while (endIdx < fixedSupersense.length) {
                    const nextTag = fixedSupersense[endIdx];
                    if (nextTag.startsWith('I-')) {
                      const [, nextCat] = nextTag.split('-');
                      if (nextCat === category) {
                        endIdx++;
                      } else {
                        break;
                      }
                    } else {
                      break;
                    }
                  }

                  const endTokenId = batchTokens[endIdx - 1]?.tokenId ?? tokenId;
                  const spanTexts = batchTokens.slice(idx, endIdx).map(t => t.text).join(' ');
                  console.log(`    B-${bTagCount++}: tokenIds=[${tokenId}-${endTokenId}] category="${category}" text="${spanTexts}"`);
                } else if (tag === 'O') {
                  // O tags don't create spans
                } else if (tag.startsWith('I-')) {
                  // Log orphan I- tags (these should NOT create spans)
                  const prevTag = idx > localStart ? fixedSupersense[idx - 1] : 'O';
                  const prevCategory = prevTag.startsWith('I-') ? prevTag.split('-')[1] : null;
                  const currentCategory = tag.split('-')[1];
                  const isOrphan = !prevTag.startsWith('B-') || prevCategory !== currentCategory;
                  if (isOrphan) {
                    console.log(`    Orphan I-: tokenId=${tokenId} category="${currentCategory}" text="${text}" (NOT EXTRACTED)`);
                  }
                }
              }
            }
          }
        }

        const supersenseAnnotations = this.advancedPostProcessor.applySupersenseAnnotations(
          batchTokens,
          fixedSupersense
        );

        // Debug: Log extracted supersense spans
        if (debugRanges.some(range => {
          const globalStartToken = batchTokens[0]?.tokenId ?? 0;
          const globalEndToken = batchTokens[tokenCount - 1]?.tokenId ?? 0;
          return globalStartToken <= range.end && globalEndToken >= range.start;
        })) {
          console.log(`  [Extracted Supersense Spans]`);
          supersenseAnnotations.forEach((ann, idx) => {
            console.log(`    Span ${idx}: tokenIds=[${ann[0]}-${ann[1]-1}] category="${ann[2]}" text="${ann[3]}"`);
          });
        }

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
        const eventTokenIds: number[] = [];
        const debugEventRanges = [
          { tokenId: 59278, text: 'says' },
          { tokenId: 59539, text: 'saying' },
          { tokenId: 61149, text: 'come' },
          { tokenId: 66049, text: 'saw' },
          { tokenId: 66091, text: 'forget' },
          { tokenId: 74510, text: 'heard' },
          { tokenId: 74512, text: 'speak' },
          { tokenId: 87334, text: 'heard' },
        ];

        // Process tokens 0 to tokenCount (inclusive), matching Python's range(tokenCount + 1) behavior
        for (let tokenIdx = 0; tokenIdx <= tokenCount && tokenIdx < eventLogitsBatch.length; tokenIdx++) {
          const logitPair = eventLogitsBatch[tokenIdx];
          if (logitPair && logitPair.length >= 2) {
            const [logit0, logit1] = logitPair;
            if (logit1 > logit0) {
              if (tokenIdx < tokenCount) {
                const token = batchTokens[tokenIdx];
                if (token) {
                  eventTokenIds.push(token.tokenId);
                  allEvents.add(token.tokenId);

                  // Debug: Log if this is one of the problematic tokens
                  const debugEntry = debugEventRanges.find(d => d.tokenId === token.tokenId);
                  if (debugEntry) {
                    console.log(`\n[DEBUG Event] Token ${token.tokenId} "${debugEntry.text}": event=true (logits: ${logit0.toFixed(4)} vs ${logit1.toFixed(4)})`);
                  }
                }
              }
            }
          }
        }

        // Debug: Log event extraction summary for this batch
        const globalStartToken = batchTokens[0]?.tokenId ?? 0;
        const globalEndToken = batchTokens[tokenCount - 1]?.tokenId ?? 0;
        const debugTokensInBatch = debugEventRanges.filter(d => d.tokenId >= globalStartToken && d.tokenId <= globalEndToken);
        if (debugTokensInBatch.length > 0) {
          console.log(`  [Event batch summary] tokens ${globalStartToken}-${globalEndToken}: extracted ${eventTokenIds.length} events`);
          for (const debugToken of debugTokensInBatch) {
            const isEvent = eventTokenIds.includes(debugToken.tokenId);
            console.log(`    Token ${debugToken.tokenId} "${debugToken.text}": ${isEvent ? 'EVENT' : 'not-event'}`);
          }
        }
      }
    }

    return {
      entities: allEntities,
      supersense: allSupersense,
      events: allEvents,
      wnBatchShape,
      supersense_debug_logits: localSupersenseDebugLogits,
      batch_debug_details: localBatchDebugs.length > 0 ? localBatchDebugs : undefined,
      paddedWordpieceLen: paddedWordpieceLen,
      paddedOriginalLen: paddedOriginalLen,
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
