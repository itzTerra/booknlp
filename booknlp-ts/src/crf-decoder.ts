/**
 * CRF Viterbi Decoder
 *
 * Implements Viterbi decoding for Conditional Random Fields, taking logits from
 * the ONNX model and CRF transition weights to produce optimal tag sequences.
 *
 * This replaces the CRF layer from the PyTorch model which was previously embedded
 * in the ONNX export. Now the TypeScript implementation performs this decoding
 * externally for maximum flexibility.
 */

import { CRFTransitions } from './types';

interface ViterbiResult {
  scores: number[];
  paths: number[][];
}

export class CRFDecoder {
  private entityTransitions: number[][] = [];
  private supersenseTransitions: number[][] = [];
  private entityNumLabels: number = 0;
  private supersenseNumLabels: number = 0;
  private entityStartIdx: number = 0;
  private entityStopIdx: number = 0;
  private supersenseStartIdx: number = 0;
  private supersenseStopIdx: number = 0;

  /**
   * Load CRF transition matrices from the exported JSON file.
   *
   * @param transitions - CRF transition weights and configuration
   */
  loadTransitions(transitions: CRFTransitions): void {
    this.entityTransitions = transitions.entity_transitions;
    this.supersenseTransitions = transitions.supersense_transitions;
    this.entityNumLabels = transitions.entity_num_labels;
    this.supersenseNumLabels = transitions.supersense_num_labels;
    this.entityStartIdx = transitions.entity_start_idx;
    this.entityStopIdx = transitions.entity_stop_idx;
    this.supersenseStartIdx = transitions.supersense_start_idx;
    this.supersenseStopIdx = transitions.supersense_stop_idx;
  }

  /**
   * Decode entity logits using Viterbi algorithm with CRF transitions.
   *
   * Mirrors the PyTorch CRF.viterbi_decode() behavior, taking logits and sequence
   * lengths and returning optimal tag sequences.
   *
   * @param entityLogits - [batch_size, seq_len, num_labels] logits for entity tagging
   * @param seqLengths - [batch_size] actual sequence lengths (excluding padding)
   * @returns ViterbiResult with scores and paths for each batch
   */
  decodeEntity(entityLogits: number[][][], seqLengths: number[]): ViterbiResult {
    return this.viterbiDecode(
      entityLogits,
      seqLengths,
      this.entityTransitions,
      this.entityNumLabels,
      this.entityStartIdx,
      this.entityStopIdx
    );
  }

  /**
   * Decode supersense logits using Viterbi algorithm with CRF transitions.
   *
   * @param supersenseLogits - [batch_size, seq_len, num_labels] logits for supersense tagging
   * @param seqLengths - [batch_size] actual sequence lengths
   * @returns ViterbiResult with scores and paths for each batch
   */
  decodeSupersense(
    supersenseLogits: number[][][],
    seqLengths: number[]
  ): ViterbiResult {
    return this.viterbiDecode(
      supersenseLogits,
      seqLengths,
      this.supersenseTransitions,
      this.supersenseNumLabels,
      this.supersenseStartIdx,
      this.supersenseStopIdx
    );
  }

  /**
   * Core Viterbi decoding algorithm for CRF.
   *
   * Based on PyTorch CRF implementation:
   * https://github.com/kaniblu/pytorch-bilstmcrf
   *
   * @param logits - [batch_size, seq_len, num_labels] token logits
   * @param seqLengths - [batch_size] actual sequence lengths
   * @param transitions - [num_labels, num_labels] transition score matrix
   * @param numLabels - number of possible tags
   * @param startIdx - index of START token (typically num_labels - 2)
   * @param stopIdx - index of STOP token (typically num_labels - 1)
   * @returns ViterbiResult with optimal paths and scores
   */
  private viterbiDecode(
    logits: number[][][],
    seqLengths: number[],
    transitions: number[][],
    numLabels: number,
    startIdx: number,
    stopIdx: number
  ): ViterbiResult {
    const batchSize = logits.length;
    const seqLen = logits[0].length;

    // Initialize viterbi matrix: [batch_size, num_labels]
    // Start at START token with score 0, all others at -10000
    let vit = this.initializeViterbi(batchSize, numLabels, startIdx);

    // Transpose logits for efficient iteration: [seq_len, batch_size, num_labels]
    const logitsTransposed = this.transposeLogits(logits);

    // Track pointers for backtracking: [seq_len - 1, batch_size, num_labels]
    const pointers: number[][][] = [];
    let cLens = [...seqLengths];

    // Forward pass: compute Viterbi scores
    for (let tIdx = 0; tIdx < logitsTransposed.length; tIdx++) {
      const logit = logitsTransposed[tIdx]; // [batch_size, num_labels]

      // Compute transition scores: vit + transitions + logit
      // vit_exp: [batch_size, num_labels, num_labels] (vit values, broadcasted)
      // trn_exp: [batch_size, num_labels, num_labels] (transitions, broadcasted)
      // logit_exp: [batch_size, num_labels, num_labels] (logits, broadcasted)
      const vitTrnSum = this.computeViterbiTransitions(vit, transitions);

      // Find max over previous states: [batch_size, num_labels]
      const { vitMax, vitArgmax } = this.maxOverPreviousStates(vitTrnSum);

      // Store pointers for backtracking
      pointers.push(vitArgmax);

      // Update viterbi: vitMax + logit (next state scores)
      const vitNxt = this.addMatrices(vitMax, logit);

      // Apply mask for actual sequence lengths
      vit = this.applyLengthMask(vit, vitNxt, cLens, batchSize, numLabels);

      // Add transition to STOP at the end of each sequence
      for (let b = 0; b < batchSize; b++) {
        if (cLens[b] === 1) {
          const stopTransition = transitions[stopIdx];
          for (let l = 0; l < numLabels; l++) {
            vit[b][l] += stopTransition[l];
          }
        }
      }

      // Decrement lengths
      cLens = cLens.map((l) => l - 1);
    }

    // Get best path scores and indices
    const { scores, paths } = this.backtrackViterbi(vit, pointers);

    return { scores, paths };
  }

  /**
   * Initialize Viterbi matrix with START token score.
   */
  private initializeViterbi(
    batchSize: number,
    numLabels: number,
    startIdx: number
  ): number[][] {
    const vit: number[][] = [];
    for (let b = 0; b < batchSize; b++) {
      const row: number[] = new Array(numLabels).fill(-10000);
      row[startIdx] = 0;
      vit.push(row);
    }
    return vit;
  }

  /**
   * Transpose logits from [batch, seq, labels] to [seq, batch, labels] for iteration.
   */
  private transposeLogits(logits: number[][][]): number[][][] {
    const batchSize = logits.length;
    const seqLen = logits[0].length;
    const numLabels = logits[0][0].length;

    const transposed: number[][][] = [];
    for (let t = 0; t < seqLen; t++) {
      const batch: number[][] = [];
      for (let b = 0; b < batchSize; b++) {
        batch.push([...logits[b][t]]);
      }
      transposed.push(batch);
    }
    return transposed;
  }

  /**
   * Compute viterbi + transitions + logit for all states.
   *
   * This computes the score matrix for finding the best previous state.
   * Shape: [batch_size, num_labels, num_labels] where [b, from, to] represents
   * the score of transitioning from 'from' state to 'to' state in batch b.
   */
  private computeViterbiTransitions(
    vit: number[][],
    transitions: number[][]
  ): number[][][] {
    const batchSize = vit.length;
    const numLabels = vit[0].length;

    const result: number[][][] = [];
    for (let b = 0; b < batchSize; b++) {
      const batchResult: number[][] = [];
      for (let to = 0; to < numLabels; to++) {
        const toRow: number[] = [];
        for (let from = 0; from < numLabels; from++) {
          // Match Python CRF: transitions are indexed as [to][from] (see booknlp/common/crf.py)
          toRow.push(vit[b][from] + transitions[to][from]);
        }
        batchResult.push(toRow);
      }
      result.push(batchResult);
    }
    return result;
  }

  /**
   * Find maximum score and argmax over previous states.
   *
   * @param vitTrnSum - [batch_size, num_labels, num_labels]
   * @returns vitMax and vitArgmax, each [batch_size, num_labels]
   */
  private maxOverPreviousStates(
    vitTrnSum: number[][][]
  ): { vitMax: number[][]; vitArgmax: number[][] } {
    const batchSize = vitTrnSum.length;
    const numLabels = vitTrnSum[0].length;

    const vitMax: number[][] = [];
    const vitArgmax: number[][] = [];

    for (let b = 0; b < batchSize; b++) {
      const maxRow: number[] = [];
      const argmaxRow: number[] = [];

      for (let to = 0; to < numLabels; to++) {
        const scores = vitTrnSum[b][to]; // Scores from all previous states to 'to'
        let maxScore = scores[0];
        let maxIdx = 0;

        for (let from = 1; from < numLabels; from++) {
          if (scores[from] > maxScore) {
            maxScore = scores[from];
            maxIdx = from;
          }
        }

        maxRow.push(maxScore);
        argmaxRow.push(maxIdx);
      }

      vitMax.push(maxRow);
      vitArgmax.push(argmaxRow);
    }

    return { vitMax, vitArgmax };
  }

  /**
   * Add two matrices element-wise.
   */
  private addMatrices(mat1: number[][], mat2: number[][]): number[][] {
    return mat1.map((row, b) => row.map((val, l) => val + mat2[b][l]));
  }

  /**
   * Apply length mask to Viterbi scores.
   *
   * Masks out scores for positions beyond actual sequence length.
   */
  private applyLengthMask(
    vitOld: number[][],
    vitNxt: number[][],
    cLens: number[],
    batchSize: number,
    numLabels: number
  ): number[][] {
    const vit: number[][] = [];
    for (let b = 0; b < batchSize; b++) {
      const row: number[] = [];
      const mask = cLens[b] > 0 ? 1 : 0;
      for (let l = 0; l < numLabels; l++) {
        row.push(mask * vitNxt[b][l] + (1 - mask) * vitOld[b][l]);
      }
      vit.push(row);
    }
    return vit;
  }

  /**
   * Backtrack through pointers to recover optimal paths.
   *
   * @param vit - Final viterbi scores [batch_size, num_labels]
   * @param pointers - Previous state indices for each position
   * @returns scores and paths
   */
  private backtrackViterbi(
    vit: number[][],
    pointers: number[][][]
  ): { scores: number[]; paths: number[][] } {
    const batchSize = vit.length;
    const scores: number[] = [];
    const bestIdx: number[] = [];

    for (let b = 0; b < batchSize; b++) {
      let maxScore = vit[b][0];
      let maxLabel = 0;
      for (let l = 1; l < vit[b].length; l++) {
        if (vit[b][l] > maxScore) {
          maxScore = vit[b][l];
          maxLabel = l;
        }
      }
      scores.push(maxScore);
      bestIdx.push(maxLabel);
    }

    const paths: number[][] = new Array(batchSize).fill(null).map(() => new Array(pointers.length).fill(0));

    for (let b = 0; b < batchSize; b++) {
      let current = bestIdx[b];
      for (let tIdx = pointers.length - 1; tIdx >= 0; tIdx--) {
        paths[b][tIdx] = current;
        current = pointers[tIdx][b][current];
      }
    }

    return { scores, paths };
  }
}
