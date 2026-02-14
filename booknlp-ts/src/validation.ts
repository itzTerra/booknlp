import { SpaCyContext, SpaCyToken, BookNLPConfig, ValidationError } from './types';

export class SpaCyValidationError extends Error {
  constructor(
    public errors: Array<{ field: string; message: string }>,
  ) {
    super(`Validation failed: ${errors.map((e) => `${e.field}: ${e.message}`).join('; ')}`);
    this.name = 'SpaCyValidationError';
  }
}

export function validateSpaCyContext(context: SpaCyContext): ValidationError[] {
  const errors: ValidationError[] = [];

  if (!context.tokens || !Array.isArray(context.tokens)) {
    errors.push({ field: 'tokens', message: 'tokens must be a non-empty array' });
  } else if (context.tokens.length === 0) {
    errors.push({ field: 'tokens', message: 'tokens array is empty' });
  }

  if (!context.sentences || !Array.isArray(context.sentences)) {
    errors.push({ field: 'sentences', message: 'sentences must be a non-empty array' });
  } else if (context.sentences.length === 0) {
    errors.push({ field: 'sentences', message: 'sentences array is empty' });
  }

  if (context.tokens) {
    for (let i = 0; i < context.tokens.length; i++) {
      const tokenErrors = validateSpaCyToken(context.tokens[i], i);
      errors.push(...tokenErrors);
    }
  }

  return errors;
}

export function validateSpaCyToken(token: SpaCyToken, index: number): ValidationError[] {
  const errors: ValidationError[] = [];
  const prefix = `tokens[${index}]`;

  if (typeof token.text !== 'string' || token.text.length === 0) {
    errors.push({ field: `${prefix}.text`, message: 'text must be a non-empty string' });
  }

  if (typeof token.startByte !== 'number' || token.startByte < 0) {
    errors.push({ field: `${prefix}.startByte`, message: 'startByte must be a non-negative number' });
  }

  if (typeof token.endByte !== 'number' || token.endByte <= token.startByte) {
    errors.push({ field: `${prefix}.endByte`, message: 'endByte must be greater than startByte' });
  }

  if (typeof token.pos !== 'string' || token.pos.length === 0) {
    errors.push({ field: `${prefix}.pos`, message: 'pos must be a non-empty string' });
  }

  if (typeof token.finePos !== 'string' || token.finePos.length === 0) {
    errors.push({ field: `${prefix}.finePos`, message: 'finePos must be a non-empty string' });
  }

  if (typeof token.lemma !== 'string' || token.lemma.length === 0) {
    errors.push({ field: `${prefix}.lemma`, message: 'lemma must be a non-empty string' });
  }

  if (typeof token.deprel !== 'string' || token.deprel.length === 0) {
    errors.push({ field: `${prefix}.deprel`, message: 'deprel must be a non-empty string' });
  }

  if (token.dephead !== null && typeof token.dephead !== 'number') {
    errors.push({ field: `${prefix}.dephead`, message: 'dephead must be a number or null' });
  }

  if (typeof token.paragraphId !== 'number' || token.paragraphId < 0) {
    errors.push({ field: `${prefix}.paragraphId`, message: 'paragraphId must be a non-negative number' });
  }

  if (typeof token.tokenId !== 'number' || token.tokenId < 0) {
    errors.push({ field: `${prefix}.tokenId`, message: 'tokenId must be a non-negative number' });
  }

  if (token.ner !== null && typeof token.ner !== 'string') {
    errors.push({ field: `${prefix}.ner`, message: 'ner must be a string or null' });
  }

  if (typeof token.sentenceId !== 'number' || token.sentenceId < 0) {
    errors.push({ field: `${prefix}.sentenceId`, message: 'sentenceId must be a non-negative number' });
  }

  if (typeof token.withinSentenceId !== 'number' || token.withinSentenceId < 0) {
    errors.push({
      field: `${prefix}.withinSentenceId`,
      message: 'withinSentenceId must be a non-negative number',
    });
  }

  if (typeof token.morph !== 'object' || token.morph === null) {
    errors.push({ field: `${prefix}.morph`, message: 'morph must be a non-null object' });
  }

  if (typeof token.likeNum !== 'boolean') {
    errors.push({ field: `${prefix}.likeNum`, message: 'likeNum must be a boolean' });
  }

  if (typeof token.isStop !== 'boolean') {
    errors.push({ field: `${prefix}.isStop`, message: 'isStop must be a boolean' });
  }

  if (typeof token.itext !== 'string') {
    errors.push({ field: `${prefix}.itext`, message: 'itext must be a string' });
  }

  if (typeof token.inQuote !== 'boolean') {
    errors.push({ field: `${prefix}.inQuote`, message: 'inQuote must be a boolean' });
  }

  if (typeof token.event !== 'boolean') {
    errors.push({ field: `${prefix}.event`, message: 'event must be a boolean' });
  }

  return errors;
}

export function validateBookNLPConfig(config: BookNLPConfig): ValidationError[] {
  const errors: ValidationError[] = [];

  const validTasks = ['entity', 'supersense', 'event'];

  if (config.pipeline.length === 0) {
    errors.push({ field: 'pipeline', message: 'pipeline must be a non-empty string or array' });
  }

  for (let i = 0; i < config.pipeline.length; i++) {
    if (!validTasks.includes(config.pipeline[i])) {
      errors.push({
        field: `pipeline[${i}]`,
        message: `invalid pipeline task "${config.pipeline[i]}". Must be one of: ${validTasks.join(', ')}`,
      });
    }
  }

  return errors;
}

export function throwIfValidationErrors(errors: ValidationError[]): void {
  if (errors.length > 0) {
    throw new SpaCyValidationError(errors);
  }
}
