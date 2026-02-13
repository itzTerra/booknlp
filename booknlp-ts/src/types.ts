export interface SpaCyToken {
  text: string;
  startByte: number;
  endByte: number;
  pos: string;
  finePos: string;
  lemma: string;
  deprel: string;
  dephead: number;
  morph: Record<string, string>;
  likeNum: boolean;
  isStop: boolean;
  sentenceId: number;
  withinSentenceId: number;
}

export interface CRFTransitions {
  entity_transitions: number[][];
  supersense_transitions: number[][];
  entity_num_labels: number;
  supersense_num_labels: number;
  entity_start_idx: number;
  entity_stop_idx: number;
  supersense_start_idx: number;
  supersense_stop_idx: number;
}

export interface SentenceBoundary {
  start: number;
  end: number;
}

export interface ParagraphBoundary {
  start: number;
  end: number;
}

export interface NounChunk {
  start: number;
  end: number;
  text: string;
}

export interface SpaCyContext {
  tokens: SpaCyToken[];
  sentences: SentenceBoundary[];
  paragraphs?: ParagraphBoundary[];
  nounChunks?: NounChunk[];
}

export interface Token {
  paragraphId: number;
  sentenceId: number;
  withinSentenceId: number;
  tokenId: number;
  text: string;
  pos: string | null;
  finePos: string | null;
  lemma: string | null;
  deprel: string | null;
  dephead: number | null;
  ner: string | null;
  startByte: number;
  endByte: number;
  morph: Record<string, string>;
  likeNum: boolean;
  isStop: boolean;
  itext: string;
  inQuote: boolean;
  event: boolean;
}

export interface Entity {
  start: number;
  end: number;
  entityId?: number;
  quoteId?: number;
  proper?: boolean;
  nerCat?: string;
  inQuote?: boolean;
  text?: string;
  globalStart?: number;
  globalEnd?: number;
}

export interface EntityAnnotation {
  startToken: number;
  endToken: number;
  cat: string;
  text: string;
  prop: string;
  coref: number;
}

export type SupersenseAnnotation = [number, number, string, string];

export interface BookNLPResult {
  tokens: Token[];
  sents: any[];
  nounChunks: any[];
  entities: EntityAnnotation[];
  supersense: SupersenseAnnotation[];
  timing: Record<string, any>;
  _debug?: Record<string, any>;
}

export type ExecutionProvider = 'wasm' | 'webgl' | 'webgpu';

export interface Resources {
  entityTagset: string;
  supersenseTagset: string;
  wordNet: string;
  crfTransitions: CRFTransitions;
}

export interface BookNLPConfig {
  modelPath?: string;
  // Pipeline tasks to run. Mirror Python's comma-separated string: "entity,supersense,event"
  // Each string should be one of: "entity", "supersense", "event"
  pipeline: string | string[];
  verbose?: boolean;
  executionProviders?: ExecutionProvider[];
  wasmPaths?: string | Record<string, string>;
}

export interface BertTokenizationResult {
  tokens: string[];
  tokenIds: number[];
  attentionMask: number[];
  transformMatrix: number[][];
  subwordToTokenMap?: Array<number | null>;
}

export interface ViterbiResult {
  path: number[];
  score: number;
}

export interface ValidationError {
  field: string;
  message: string;
}
