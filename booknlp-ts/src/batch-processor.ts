import { SpaCyContext, SentenceBoundary } from 'types';

const MAX_SEQUENCE_LENGTH = 512;
const PAD_TOKEN_ID = 0;
const MASK_VALUE = -100;

export interface Batch {
  tokenIds: number[][];
  attentionMask: number[][];
  transformMatrices: number[][][];
  sentenceIndices: number[];
  sentenceLengths: number[];
  batchSize: number;
  maxLength: number;
}

export class BatchProcessor {
  createBatches(
    tokenIdsList: number[][],
    transformMatricesList: number[][][],
    sentences: SentenceBoundary[],
    maxBatchSize: number = 32,
  ): Batch[] {
    const batches: Batch[] = [];
    let currentBatch: {
      tokenIds: number[][];
      transformMatrices: number[][][];
      sentenceIndices: number[];
    } = {
      tokenIds: [],
      transformMatrices: [],
      sentenceIndices: [],
    };

    for (let i = 0; i < tokenIdsList.length; i++) {
      currentBatch.tokenIds.push(tokenIdsList[i]);
      currentBatch.transformMatrices.push(transformMatricesList[i]);
      currentBatch.sentenceIndices.push(i);

      if (currentBatch.tokenIds.length === maxBatchSize || i === tokenIdsList.length - 1) {
        batches.push(this.padBatch(currentBatch));
        currentBatch = {
          tokenIds: [],
          transformMatrices: [],
          sentenceIndices: [],
        };
      }
    }

    return batches;
  }

  private padBatch(batchData: {
    tokenIds: number[][];
    transformMatrices: number[][][];
    sentenceIndices: number[];
  }): Batch {
    const maxLength = Math.min(
      Math.max(...batchData.tokenIds.map((ids) => ids.length)),
      MAX_SEQUENCE_LENGTH,
    );

    const paddedTokenIds: number[][] = [];
    const attentionMask: number[][] = [];
    const paddedTransformMatrices: number[][][] = [];
    const sentenceLengths: number[] = [];

    for (let i = 0; i < batchData.tokenIds.length; i++) {
      const tokenIds = batchData.tokenIds[i];
      const transformMatrix = batchData.transformMatrices[i];

      const paddedIds = new Array(maxLength).fill(PAD_TOKEN_ID);
      const mask = new Array(maxLength).fill(0);

      const copyLength = Math.min(tokenIds.length, maxLength);
      for (let j = 0; j < copyLength; j++) {
        paddedIds[j] = tokenIds[j];
        mask[j] = 1;
      }

      paddedTokenIds.push(paddedIds);
      attentionMask.push(mask);
      sentenceLengths.push(copyLength);

      const paddedMatrix = this.padTransformMatrix(transformMatrix, maxLength);
      paddedTransformMatrices.push(paddedMatrix);
    }

    return {
      tokenIds: paddedTokenIds,
      attentionMask,
      transformMatrices: paddedTransformMatrices,
      sentenceIndices: batchData.sentenceIndices,
      sentenceLengths,
      batchSize: batchData.tokenIds.length,
      maxLength,
    };
  }

  private padTransformMatrix(matrix: number[][], maxLength: number): number[][] {
    const originalColumnCount = matrix[0]?.length ?? 0;
    const paddedMatrix: number[][] = [];

    for (let i = 0; i < maxLength; i++) {
      if (i < matrix.length) {
        paddedMatrix.push([...matrix[i]]);
      } else {
        paddedMatrix.push(new Array(originalColumnCount).fill(0));
      }
    }

    return paddedMatrix;
  }

  /**
   * Apply transform matrix to convert subword predictions back to token-level predictions.
   * Now works with prediction indices from ONNX model (not CRF emissions).
   *
   * @param predictions - Prediction scores/indices from ONNX model [subwords, labels]
   * @param transformMatrix - Subword-to-token mapping matrix [subwords, tokens]
   * @returns Token-level predictions [tokens, labels]
   */
  applyTransformMatrix(
    predictions: number[][],
    transformMatrix: number[][],
  ): number[][] {
    const tokenCount = transformMatrix[0]?.length ?? 0;
    const labelCount = predictions[0]?.length ?? 0;
    const result: number[][] = new Array(tokenCount).fill(null).map(() => new Array(labelCount).fill(0));

    for (let i = 0; i < predictions.length; i++) {
      const prediction = predictions[i];
      for (let j = 0; j < tokenCount; j++) {
        const weight = transformMatrix[i][j];
        if (weight > 0) {
          for (let k = 0; k < labelCount; k++) {
            result[j][k] += prediction[k] * weight;
          }
        }
      }
    }

    return result;
  }
}

export function groupSentenceBatches(
  sentences: SentenceBoundary[],
  maxTokensPerBatch: number = 512,
): Array<SentenceBoundary[]> {
  const batches: Array<SentenceBoundary[]> = [];
  let currentBatch: SentenceBoundary[] = [];
  let currentTokenCount = 0;

  for (const sentence of sentences) {
    const sentenceLength = sentence.end - sentence.start;

    if (currentTokenCount + sentenceLength > maxTokensPerBatch && currentBatch.length > 0) {
      batches.push(currentBatch);
      currentBatch = [];
      currentTokenCount = 0;
    }

    currentBatch.push(sentence);
    currentTokenCount += sentenceLength;
  }

  if (currentBatch.length > 0) {
    batches.push(currentBatch);
  }

  return batches;
}
