/**
 * Advanced Sequence Postprocessor with Layer Transformation
 *
 * This module handles post-processing of CRF outputs for hierarchical entity tagging.
 * The BookNLP entity tagger uses a 3-layer hierarchical LSTM-CRF where:
 * - Layer 1: Full token sequence with all entities
 * - Layer 2: Compressed sequence (tokens merged within same entity)
 * - Layer 3: Further compressed sequence (only major entities)
 *
 * This postprocessor:
 * 1. Converts tag indices to strings
 * 2. Fixes invalid BIO sequences
 * 3. Computes layer transformation matrices for token merging/unmerging
 * 4. Extracts entities from BIO tags
 * 5. Merges predictions across all three layers
 *
 * @see tagger.py for the corresponding Python implementation
 */

import { Token, Entity, EntityAnnotation } from 'types';

interface LayerTransformation {
  tags: string[];
  matrix: number[][];
  missing: number[];
  len: number;
}

/**
 * Utility class for advanced sequence postprocessing of hierarchical NER.
 */
export class AdvancedPostProcessor {
  private revTagset: Map<number, string> = new Map();
  private revSupersenseTagset: Map<number, string> = new Map();
  private tagset: Map<string, number> = new Map();
  private supersenseTagset: Map<string, number> = new Map();

  setTagsets(
    entityTagset: Map<number, string>,
    supersenseTagset: Map<number, string>
  ): void {
    this.revTagset = entityTagset;
    this.revSupersenseTagset = supersenseTagset;

    this.tagset.clear();
    for (const [id, tag] of entityTagset.entries()) {
      this.tagset.set(tag, id);
    }

    this.supersenseTagset.clear();
    for (const [id, tag] of supersenseTagset.entries()) {
      this.supersenseTagset.set(tag, id);
    }
  }

  /**
   * Convert tag indices to tag strings using the tagset mapping.
   *
   * @param indices - Array of tag indices from CRF decoder
   * @param tagset - Mapping from index to tag string
   * @returns Array of tag strings (e.g., ["B-PER", "I-PER", "O"])
   */
  convertIndicesToTags(indices: number[], tagset: Map<number, string>): string[] {
    return indices.map((idx) => tagset.get(idx) ?? 'O');
  }

  convertTagsToIndices(tags: string[]): number[] {
    return tags.map((tag) => this.tagset.get(tag) ?? this.tagset.get('O') ?? 0);
  }

  /**
   * Fix invalid BIO tag sequences.
   *
   * Ensures that:
   * - I-tags without preceding B-tags of same type become B-tags
   * - Different entity types in sequence start with B-tags
   *
   * Mirrors: tagger.py:_predict_all() -> fix()
   *
   * @param tags - Array of BIO tags
   * @returns Array of fixed BIO tags
   */
  fixBIOTags(tags: string[]): string[] {
    const fixed = [...tags];

    for (let i = 0; i < fixed.length; i++) {
      const tag = fixed[i];

      if (tag.startsWith('I-')) {
        const currentType = tag.slice(2);
        let foundMatchingB = false;

        // Look backwards for a matching B-tag
        for (let j = i - 1; j >= 0; j--) {
          const prevTag = fixed[j];

          if (prevTag === 'O') {
            break;
          }

          if (prevTag.startsWith('B-') || prevTag.startsWith('I-')) {
            const prevType = prevTag.slice(2);

            // Found matching B-tag
            if (prevTag.startsWith('B-') && prevType === currentType) {
              foundMatchingB = true;
              break;
            }

            // Found different entity type - stop looking
            if (prevType !== currentType) {
              break;
            }
          }
        }

        // No matching B-tag found - convert to B-tag
        if (!foundMatchingB) {
          fixed[i] = `B-${currentType}`;
        }
      }
    }

    return fixed;
  }

  /**
   * Compute layer transformation matrix and missing token indices.
   *
   * When tokens with same entity type are merged (compressed) between layers,
   * we need to track which tokens were merged and create a transformation matrix
   * to expand/compress between layers.
   *
   * Mirrors: tagger.py:_predict_all() -> get_layer_transformation()
   *
   * @param tags - BIO tags for this layer
   * @param revTagset - Mapping from tag index to string
   * @param seqLen - Maximum sequence length (for matrix size)
   * @returns LayerTransformation with matrix and missing indices
   */
  computeLayerTransformationFromIndices(
    tagIndices: number[],
    seqLen: number
  ): LayerTransformation {
    const tags = this.convertIndicesToTags(tagIndices, this.revTagset);
    const fixedTags = this.fixBIOTags(tags);
    const fixedIndices = this.convertTagsToIndices(fixedTags);
    const matrix = this.buildIndexMatrix(fixedIndices, seqLen);
    const missing = this.getMissingIndicesFromIndices(fixedIndices);
    const len = this.getCompressedLength(fixedIndices);

    return {
      tags: fixedTags,
      matrix,
      missing,
      len,
    };
  }

  /**
   * Get indices of tokens that should be compressed (merged) to next layer.
   *
   * Tokens with I-tags (continuation) within the same entity are candidates for compression.
   * These get removed from the sequence, creating a shorter layer.
   *
   * @param tags - BIO tags
   * @returns Array of indices to compress
   */
  private getMissingIndicesFromIndices(tagIndices: number[]): number[] {
    const missing: number[] = [];

    for (let idx = 0; idx < tagIndices.length; idx++) {
      if (idx === 0) {
        continue;
      }
      const tag = this.revTagset.get(tagIndices[idx]) ?? 'O';
      if (tag.startsWith('I-')) {
        missing.push(idx);
      }
    }

    return missing;
  }

  /**
   * Build transformation matrix for projecting between layers.
   *
   * Creates a matrix that maps token indices from current layer to next layer.
   * Each row represents how to aggregate features for the next (compressed) layer.
   *
   * For hierarchical NER:
   * - Layer 1 -> Layer 2: Tokens with same entity type are merged
   * - Layer 2 -> Layer 3: Further merging of minor entities
   *
   * @param tags - BIO tags for this layer
   * @param seqLen - Maximum sequence length (for padding)
   * @returns Transformation matrix [seq_len, seq_len]
   */
  private buildIndexMatrix(tagIndices: number[], seqLen: number): number[][] {
    const index: number[][] = [];

    for (let idx = 0; idx < tagIndices.length; idx++) {
      const ind = new Array(tagIndices.length).fill(0);
      const tag = this.revTagset.get(tagIndices[idx]) ?? 'O';

      if (tagIndices[idx] === -100 || !tag.startsWith('I-')) {
        ind[idx] = 1;
        index.push(ind);
      } else {
        if (index.length > 0) {
          index[index.length - 1][idx] = 1;
        } else {
          ind[idx] = 1;
          index.push(ind);
        }
      }
    }

    for (let i = 0; i < index.length; i++) {
      const row = index[i];
      const sum = row.reduce((acc, val) => acc + val, 0);
      if (sum > 0) {
        index[i] = row.map((val) => val / sum);
      }
    }

    for (let i = 0; i < index.length; i++) {
      while (index[i].length < seqLen) {
        index[i].push(0);
      }
    }

    while (index.length < seqLen) {
      index.push(new Array(seqLen).fill(0));
    }

    return index;
  }

  private getCompressedLength(tagIndices: number[]): number {
    let count = 0;
    for (let idx = 0; idx < tagIndices.length; idx++) {
      const tag = this.revTagset.get(tagIndices[idx]) ?? 'O';
      if (idx > 0 && tag.startsWith('I-')) {
        continue;
      }
      count += 1;
    }
    return count;
  }

  /**
   * Restore tokens that were compressed in previous layers.
   *
   * When moving from layer N to layer N+1, we need to re-insert tokens that were
   * compressed (merged) in layer N. These are marked in the missing indices.
   *
   * Mirrors: tagger.py:_predict_all() lines 415-435
   *
   * @param tags - Tag indices for compressed layer
   * @param missing - Indices where tokens should be restored
   * @returns Tag indices with restored tokens
   */
  restoreCompressedTokens(tags: number[], missing: number[]): number[] {
    const restored = [...tags];

    for (const m of missing) {
      if (m <= 0 || m > restored.length) {
        continue;
      }

      // Get tag of previous token to determine entity type
      const prevTag = this.revTagset.get(restored[m - 1]) ?? 'O';
      const parts = prevTag.split('-');

      let insertTag: number;
      if (parts.length > 1) {
        // Create I-tag with same entity type as previous token
        const iTag = `I-${parts[1]}`;
        insertTag = this.tagset.get(iTag) ?? this.tagset.get('O') ?? 0;
      } else {
        insertTag = this.tagset.get('O') ?? 0;
      }

      restored.splice(m, 0, insertTag);
    }

    return restored;
  }

  /**
   * Extract entities from BIO-tagged sequence.
   *
   * Groups consecutive B and I tags into entities with spans and text.
   *
   * @param tags - BIO tag strings
   * @param tokens - Token objects with text and position info
   * @returns Array of extracted entities
   */
  /**
   * Extract entity spans from BIO-tagged sequence.
   *
   * Matches Python's entity extraction behavior (tagger.py):
   * - ONLY creates entities starting from B- tags
   * - Continues with I- tags of matching category
   * - Stops when encountering B-, O, or mismatched I- tags
   * - Crucially: Orphan I- tags (not preceded by B-) are IGNORED
   * - Does NOT create entities from orphan I- tags (unlike some BIO decoders)
   *
   * Reference: Python code tagger.py iterates only when tag.startswith("B-")
   *
   * @param tags - BIO tag strings
   * @param tokens - Token objects
   * @returns Array of Entity objects {start, end, nerCat, text}
   */
  extractEntitiesFromBIO(tags: string[], tokens: Token[]): Entity[] {
    const entities: Entity[] = [];

    for (let i = 0; i < tags.length; i++) {
      const tag = tags[i];

      // ONLY process B- tags as entity starts, matching Python's strict behavior
      if (tag.startsWith('B-')) {
        const [, type] = tag.split('-');
        let j = i + 1;

        // Extend entity while seeing matching I- tags
        while (j < tags.length) {
          const nextTag = tags[j];

          // Stop if B- or O tag encountered
          if (nextTag.startsWith('B-') || nextTag === 'O') {
            break;
          }

          // For I- tags, check if category matches
          if (nextTag.startsWith('I-')) {
            const [, nextType] = nextTag.split('-');
            if (nextType !== type) {
              // Category mismatch - stop extending entity
              break;
            }
            // Category matches, continue
            j++;
          } else {
            // Unexpected tag format, stop
            break;
          }
        }

        // Create entity with span from i to j (exclusive end)
        const entity: Entity = {
          start: i,
          end: j,
          nerCat: type,
          text: tokens
            .slice(i, j)
            .map((t) => t.text)
            .join(' '),
        };

        // Filter out punctuation-only entities (matching Python behavior)
        const punctuationPattern = /^[\p{P}\p{Z}]+$/u;
        if (!punctuationPattern.test((entity.text || '').trim())) {
          entities.push(entity);
        }
      }
    }

    return entities;
  }

  /**
   * Assign entity types from BIO tags and entity category map.
   *
   * Converts extracted entities to EntityAnnotation format with proper category
   * and property information (PROP/NOM/PRON for named entity types).
   *
   * @param entities - Raw entities from BIO extraction
   * @param entityCategoryMap - Mapping from entity type to categories
   * @returns Formatted EntityAnnotations
   */
  assignEntityTypes(
    entities: Entity[],
    entityCategoryMap: Map<string, string>
  ): EntityAnnotation[] {
    return entities.map((entity) => {
      const category = entity.nerCat ?? 'O';

      let prop = 'UNKNOWN';
      let cat = category;

      // If category contains underscore (e.g., "PROP_PER"), split it
      if (category.includes('_')) {
        const parts = category.split('_');
        prop = parts[0];
        cat = parts[1];
      }

      return {
        startToken: entity.start,
        endToken: entity.end,
        cat: cat,
        text: entity.text ?? '',
        prop: prop,
      };
    });
  }

  /**
   * Merge entity predictions from all three layers.
   *
   * Combines entities detected at different compression levels:
   * - Layer 1: Most granular entities
   * - Layer 2: Medium-level entities
   * - Layer 3: Coarse entities
   *
   * Deduplicates and sorts by span. Deduplication uses (start, end, nerCat) as key,
   * matching Python's behavior in tagger.py:518-540 where entities across layers
   * are deduplicated using (doc_idx, label, start, end) within each sentence.
   * Since start/end are batch-local and unique per entity position, sentenceId is redundant.
   *
   * @param layer1 - Entities from layer 1
   * @param layer2 - Entities from layer 2
   * @param layer3 - Entities from layer 3
   * @param tokens - Token array (unused but kept for API compatibility)
   * @returns Merged and deduplicated entity list
   */
  mergeEntityLayers(layer1: Entity[], layer2: Entity[], layer3: Entity[], tokens?: Token[]): Entity[] {
    const allEntitiesMap = new Map<string, Entity>();

    // Deduplicate using batch-local indices and entity type (nerCat).
    // All three layers extract from the SAME batchTokens array during the same batch processing,
    // so entities at identical (start, end, nerCat) positions are duplicates and should be deduplicated.
    // This mimics Python's deduplication in tagger.py:520-536, which uses (doc_idx, label, idx, j)
    // where doc_idx=sentence_index (batch-local), label=entity_type, idx=start, j=end.
    // Since entities within a batch must deduplicate by span, we use (start, end, nerCat) as the key.
    const addEntities = (entities: Entity[]) => {
      for (const entity of entities) {
        // Deduplication key: (batch-local-start, batch-local-end, entity-type)
        const key = `${entity.start}-${entity.end}-${entity.nerCat}`;
        if (!allEntitiesMap.has(key)) {
          allEntitiesMap.set(key, entity);
        }
      }
    };

    addEntities(layer1);
    addEntities(layer2);
    addEntities(layer3);

    return Array.from(allEntitiesMap.values()).sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      return a.end - b.end;
    });
  }

  /**
   * Extract supersense annotations from BIO-tagged sequence.
   *
   * Matches Python's _get_spans behavior (tagger.py:598-631):
   * - ONLY creates spans starting from B- tags
   * - Continues with I- tags of matching category
   * - Stops when encountering B-, O, or mismatched I- tags
   * - Crucially: Orphan I- tags (not preceded by matching B-) are IGNORED
   *
   * This differs from standard BIO extraction where orphan I- tags might be
   * handled specially. Python strictly only processes B- tags as span starts.
   *
   * Reference: Python code tagger.py:598-631
   * - Iterates only when tag.startswith("B-")
   * - Creates span key and adds to entities dict
   * - No special handling for orphan I- tags
   *
   * @param tokens - Token objects
   * @param supersenseLabels - BIO tag strings for supersense layer
   * @returns Array of [startToken, endToken, category, text] tuples
   */
  applySupersenseAnnotations(
    tokens: Token[],
    supersenseLabels: string[]
  ): Array<[number, number, string, string]> {
    const annotations: Array<[number, number, string, string]> = [];

    for (let i = 0; i < supersenseLabels.length; i++) {
      const label = supersenseLabels[i];

      // Only process B- tags as span starts, mirroring Python's _get_spans logic
      if (label.startsWith('B-')) {
        const [, category] = label.split('-');
        let j = i + 1;

        // Extend span while seeing matching I- tags
        while (j < supersenseLabels.length) {
          const nextLabel = supersenseLabels[j];

          // Stop if B- or O tag encountered
          if (nextLabel.startsWith('B-') || nextLabel === 'O') {
            break;
          }

          // For I- tags, check if category matches
          if (nextLabel.startsWith('I-')) {
            const [, nextCategory] = nextLabel.split('-');
            if (nextCategory !== category) {
              // Category mismatch - stop extending span
              // Python: "if parts_n[1] != parts[1]: break" (tagger.py:625)
              break;
            }
            // Category matches, continue
            j++;
          } else {
            // Unexpected tag format, stop
            break;
          }
        }

        // Create annotation with span from i to j (exclusive end)
        annotations.push([
          i,
          j,
          category,
          tokens
            .slice(i, j)
            .map((t) => t.text)
            .join(' '),
        ]);
      }
    }

    return annotations;
  }

  /**
   * Load entity category mapping from tagset file.
   *
   * @param tagsetContent - Content of entity_cat.tagset file
   * @returns Mapping from entity type to category ID
   */
  loadEntityCategoryMap(tagsetContent: string): Map<string, string> {
    const map = new Map<string, string>();
    const lines = tagsetContent.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.length > 0 && !trimmed.startsWith('#')) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 2) {
          map.set(parts[0], parts[1]);
        }
      }
    }

    return map;
  }

  /**
   * Load supersense category mapping from tagset file.
   *
   * @param tagsetContent - Content of supersense.tagset file
   * @returns Mapping from supersense to itself (identity mapping)
   */
  loadSupersenseMap(tagsetContent: string): Map<string, string> {
    const map = new Map<string, string>();
    const lines = tagsetContent.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.length > 0 && !trimmed.startsWith('#')) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 1) {
          map.set(parts[0], parts[0]);
        }
      }
    }

    return map;
  }
}
