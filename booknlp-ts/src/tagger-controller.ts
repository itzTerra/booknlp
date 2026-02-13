import * as ort from 'onnxruntime-web';
import { ExecutionProvider } from './types';
import { PreTrainedModel } from '@huggingface/transformers';

export interface InferenceConfig {
  executionProviders?: ExecutionProvider[];
  wasmPaths?: string | Record<string, string>;
}

/**
 * ONNX Model Output Structure (Logits Only)
 *
 * The ONNX model now outputs raw logits (not predictions) for all tasks.
 * CRF Viterbi decoding and other postprocessing is handled in TypeScript.
 *
 * Output shapes:
 * - entityLogits1, entityLogits2, entityLogits3: [batch_size, seq_len, num_entity_labels]
 *   Raw logit scores for entity tagging (39 labels: 37 entity types + START + STOP)
 *
 * - supersenseLogits: [batch_size, seq_len, num_supersense_labels]
 *   Raw logit scores for supersense tagging (85 labels: 83 categories + START + STOP)
 *
 * - eventLogits: [batch_size, seq_len, 2]
 *   Raw logit scores for binary event detection (2 labels: non-event, event)
 *
 * These logits need to be:
 * 1. Passed through CRF Viterbi decoding (for entity and supersense)
 * 2. Passed through argmax (for events)
 * 3. Further postprocessed with BIO fixing and layer transformation
 */
export interface TaggerLogits {
  entityLogits1: number[][][];
  entityLogits2: number[][][];
  entityLogits3: number[][][];
  supersenseLogits: number[][][];
  eventLogits: number[][][];
}

export class ONNXTaggerController {
  private onnxSession: ort.InferenceSession | null = null;
  private executionProviders: ExecutionProvider[];
  private modelPath: string;
  private wasmPaths?: string | Record<string, string>;

  constructor(
    modelPath: string,
    executionProviders: ExecutionProvider[] = ['wasm'],
    wasmPaths?: string | Record<string, string>
  ) {
    this.modelPath = modelPath;
    this.executionProviders = executionProviders;
    this.wasmPaths = wasmPaths;
  }

  async loadModel(): Promise<void> {
    const resolvedUrl = this.getHuggingFaceUrl(this.modelPath);
    try {
      if (this.wasmPaths) {
        ort.env.wasm.wasmPaths = this.wasmPaths;
      }

      // const executionProviders = this.executionProviders.length > 0 ? this.executionProviders : ['wasm'];
      // this.onnxSession = await ort.InferenceSession.create(resolvedUrl, { executionProviders });
      this.onnxSession = (await PreTrainedModel.from_pretrained(this.modelPath, {
        subfolder: 'onnx',
        dtype: "fp32",
        // session_options: {
        //   externalData: [
        //     {
        //       path: "model_fp16.onnx.data",
        //       data: "onnx/model_fp16.onnx.data"
        //     }
        //   ]
        // }
      })).sessions["model"]
    } catch (error) {
      throw new Error(`Failed to load ONNX model from ${resolvedUrl}: ${error}`);
    }
  }

  private getHuggingFaceUrl(repoId: string): string {
    return `https://huggingface.co/${repoId}/resolve/main/onnx/model.onnx`;
  }

  private toBigInt64Array(values: number[]): BigInt64Array {
    return BigInt64Array.from(values.map(value => BigInt(value)));
  }

  private toBigInt64Array2D(values: number[][]): BigInt64Array {
    return BigInt64Array.from(values.flat().map(value => BigInt(value)));
  }

  async predict(
    inputIds: number[][],
    attentionMask: number[][],
    transforms: number[][][],
    layer1To2TransformMatrix: number[][][],
    layer2To3TransformMatrix: number[][][],
    wordnetSenses: number[][],
    seqLengths: number[],
    maxOriginalTokens?: number,
  ): Promise<Partial<TaggerLogits>> {
    if (!this.onnxSession) {
      throw new Error('ONNX model not loaded. Call loadModel() first.');
    }

    const batchSize = inputIds.length;
    const wordpieceSeqLen = inputIds[0].length;
    const originalSeqLen = maxOriginalTokens ?? transforms[0]?.length ?? wordpieceSeqLen;
    const outputSeqLen = originalSeqLen - 1;

    const eventOutputSeqLen = outputSeqLen;

    // Prepare ONNX model inputs (asymmetric transforms: [batch, original_seq, wordpiece_seq])
    // All ONNX inputs must be properly formatted tensors matching the model's expected schema
    const feeds: Record<string, ort.Tensor> = {
      input_ids: new ort.Tensor(
        'int64',
        this.toBigInt64Array2D(inputIds),
        [batchSize, wordpieceSeqLen]
      ),
      attention_mask: new ort.Tensor(
        'int64',
        this.toBigInt64Array2D(attentionMask),
        [batchSize, wordpieceSeqLen]
      ),
      transforms: new ort.Tensor(
        'float32',
        Float32Array.from(transforms.flat(2)),
        [batchSize, originalSeqLen, wordpieceSeqLen]
      ),
      matrix1: new ort.Tensor(
        'float32',
        Float32Array.from(layer1To2TransformMatrix.flat(2)),
        [batchSize, originalSeqLen, originalSeqLen]
      ),
      matrix2: new ort.Tensor(
        'float32',
        Float32Array.from(layer2To3TransformMatrix.flat(2)),
        [batchSize, originalSeqLen, originalSeqLen]
      ),
      wn: new ort.Tensor(
        'int64',
        this.toBigInt64Array2D(wordnetSenses),
        [batchSize, originalSeqLen]
      ),
    };

    // Run inference through ONNX model
    // Model performs forward pass but outputs logits (not predictions)
    // CRF Viterbi decoding and argmax are handled in postprocessing
    const results = await this.onnxSession.run(feeds);

    // Extract logit tensors from ONNX outputs
    const entityLogits1Data = results.entity_logits1.data as Float32Array;
    const entityLogits2Data = results.entity_logits2.data as Float32Array;
    const entityLogits3Data = results.entity_logits3.data as Float32Array;
    const supersenseLogitsData = results.supersense_logits.data as Float32Array;
    const eventLogitsData = results.event_logits.data as Float32Array;

    // Reshape flat arrays into [batch_size, original_seq, num_labels] format
    const reshape3D = (
      data: Float32Array,
      batchSize: number,
      seqLen: number,
      numLabels: number
    ): number[][][] => {
      const result: number[][][] = [];
      for (let b = 0; b < batchSize; b++) {
        const batch: number[][] = [];
        for (let t = 0; t < seqLen; t++) {
          const position: number[] = [];
          for (let l = 0; l < numLabels; l++) {
            position.push(data[b * seqLen * numLabels + t * numLabels + l]);
          }
          batch.push(position);
        }
        result.push(batch);
      }
      return result;
    };

    // Entity logits have 39 labels (37 entity types + START + STOP)
    const entityNumLabels = 39;
    // Supersense logits have 85 labels (83 categories + START + STOP)
    const supersenseNumLabels = 85;
    // Event logits have 2 labels (non-event, event)
    const eventNumLabels = 2;

    return {
      entityLogits1: reshape3D(entityLogits1Data, batchSize, outputSeqLen, entityNumLabels),
      entityLogits2: reshape3D(entityLogits2Data, batchSize, outputSeqLen, entityNumLabels),
      entityLogits3: reshape3D(entityLogits3Data, batchSize, outputSeqLen, entityNumLabels),
      supersenseLogits: reshape3D(
        supersenseLogitsData,
        batchSize,
        outputSeqLen,
        supersenseNumLabels
      ),
      eventLogits: reshape3D(eventLogitsData, batchSize, eventOutputSeqLen, eventNumLabels),
    };
  }

  isLoaded(): boolean {
    return this.onnxSession !== null;
  }

  async unload(): Promise<void> {
    if (this.onnxSession) {
      await this.onnxSession.release();
      this.onnxSession = null;
    }
  }
}
