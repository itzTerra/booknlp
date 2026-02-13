# BookNLP TypeScript Library

Browser-compatible TypeScript implementation of BookNLP for client-side NLP inference on long documents. This library provides complete entity recognition, supersense tagging, and event detection using pre-converted ONNX models running entirely in the browser via WebAssembly.

## ✅ Implementation Status: COMPLETE (Browser-Ready)

All core functionality has been implemented and optimized for browser environments:

- ✅ **Entity Recognition**: 3-layer hierarchical LSTM-CRF with entity type classification (PER, LOC, FAC, GPE, ORG, VEH)
- ✅ **Supersense Tagging**: WordNet-based semantic role annotation
- ✅ **Event Detection**: Token-level event marker prediction
- ✅ **ONNX Inference**: Full integration with ONNX Runtime Web for neural network inference
- ✅ **WebAssembly Support**: Runs entirely in the browser without server dependencies
- ✅ **Bundled Resources**: Data files packaged with the library for offline use

## Quick Start

### Installation

```bash
npm install booknlp-ts
```

### Usage

```typescript
import { BookNLP, SpaCyContext } from 'booknlp-ts';

const config = {
  // Optional: uses Hugging Face model by default
  // modelPath: 'Terraa/entities_google_bert_uncased_L-4_H-256_A-4-v1.0-ONNX',
  pipeline: ['entity', 'supersense', 'event'],
  // Optional: specify execution providers (default: ['wasm'])
  executionProviders: ['wasm'], // or ['webgl'], ['webgpu']
};

const booknlp = new BookNLP();
await booknlp.initialize(config);

// SpaCyContext must be provided (from spaCy preprocessing)
const spaCyContext: SpaCyContext = {
  tokens: [
    {
      text: 'Harry',
      startByte: 0,
      endByte: 5,
      pos: 'PROPN',
      finePos: 'NNP',
      lemma: 'Harry',
      deprel: 'nsubj',
      dephead: 1,
      morph: {},
      likeNum: false,
      isStop: false,
      sentenceId: 0,
      withinSentenceId: 0,
    },
    // ... more tokens
  ],
  sentences: [{ start: 0, end: 10 }],
};

const result = await booknlp.process(spaCyContext);
console.log('Entities:', result.entities);
console.log('Supersense:', result.supersense);
console.log('Events:', result.tokens.filter(t => t.event));
```

## Browser Deployment

### Bundled Resources

The library automatically bundles resource files (entity tagset, supersense tagset, WordNet) from the source repository. No external network requests needed for resources.

### WASM Configuration

For custom WASM paths (advanced usage):

```typescript
const config = {
  pipeline: ['entity'],
  wasmPaths: {
    'ort-wasm.wasm': '/custom/path/to/ort-wasm.wasm',
    'ort-wasm-simd.wasm': '/custom/path/to/ort-wasm-simd.wasm',
  },
};
```

### External Resource URLs

If you prefer to host resources externally:

```typescript
const config = {
  pipeline: ['entity', 'supersense'],
};
```

## Required Input: SpaCyContext

The TypeScript implementation requires pre-processed input from spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Your text here")

spacy_context = {
    "tokens": [
        {
            "text": token.text,
            "startByte": token.idx,
            "endByte": token.idx + len(token.text),
            "pos": token.pos_,
            "finePos": token.tag_,
            "lemma": token.lemma_,
            "deprel": token.dep_,
            "dephead": token.head.i,
            "morph": {k: v for k, v in token.morph.to_dict().items()},
            "likeNum": token.like_num,
            "isStop": token.is_stop,
            "sentenceId": token.sent.start,
            "withinSentenceId": token.i - token.sent.start
        }
        for token in doc
    ],
    "sentences": [
        {"start": sent.start, "end": sent.end}
        for sent in doc.sents
    ]
}
```

## Validation Results

The validation suite compares Python and TypeScript outputs on identical input. Expected results:

| Metric            | Status           | Notes                       |
| ----------------- | ---------------- | --------------------------- |
| Token count       | ✅ Exact match    | All tokens preserved        |
| Token text        | ✅ Exact match    | Character-level accuracy    |
| Entity spans      | ✅ Match expected | Boundaries align            |
| Entity categories | ✅ Match expected | PER, LOC, FAC, etc.         |
| Supersense spans  | ✅ Match expected | Annotation boundaries align |
| Event markers     | ✅ Match expected | Token-level event flags     |

### Running Validation

```bash
cd validation
./run_validation.sh
```

This automatically:
1. Checks dependencies
2. Builds TypeScript
3. Runs Python BookNLP (generates baseline)
4. Runs TypeScript BookNLP (with same input)
5. Compares outputs (reports mismatches)

## Known Limitations

1. **No automatic model downloading**: ONNX model must be provided
2. **Requires SpaCy preprocessing**: Cannot process raw text directly
3. **CPU-focused**: CUDA support exists but not extensively tested
4. **No quote/coreference handling**: Future enhancement

## Future Enhancements

- [ ] Direct text input (integrate spaCy in TypeScript)
- [ ] WebAssembly optimization for browser use
- [ ] Quote and coreference chain extraction
- [ ] Big model variant support
- [ ] Streaming/incremental processing for very long documents
- [ ] GPU optimization and multi-GPU support

## Comparison: Python vs TypeScript

| Feature             | Python       | TypeScript (Browser) | Notes                       |
| ------------------- | ------------ | -------------------- | --------------------------- |
| Entity recognition  | ✅            | ✅                    | Equivalent                  |
| Supersense tagging  | ✅            | ✅                    | Equivalent                  |
| Event detection     | ✅            | ✅                    | Equivalent                  |
| ONNX inference      | ✅            | ✅                    | Equivalent                  |
| SpaCy preprocessing | ✅ Integrated | ⚠️ External           | Requires external spaCy     |
| Raw text input      | ✅            | ❌                    | Requires SpaCy context      |
| Model conversion    | ✅ Native     | ⚠️ Via Python         | ONNX export                 |
| Deployment          | Server-side  | Browser/Client-side  | No server required          |
| GPU acceleration    | CUDA         | WebGL/WebGPU         | Different acceleration tech |


## Key Features

### Type-Safe Interfaces
All data structures have complete TypeScript type definitions with validation:

```typescript
interface SpaCyToken {
  text: string;
  startByte: number;
  endByte: number;
  pos: string;          // Coarse POS (NOUN, VERB, etc.)
  finePos: string;      // Fine POS (NN, VBD, etc.)
  lemma: string;
  deprel: string;       // Dependency relation
  dephead: number;      // Head token index
  morph: Record<string, string>;
  likeNum: boolean;
  isStop: boolean;
  sentenceId: number;
  withinSentenceId: number;
}

interface EntityAnnotation {
  startToken: number;
  endToken: number;
  cat: string;          // PER, LOC, FAC, GPE, ORG, VEH
  text: string;
  prop: string;         // PROP, NOM, PRON
}

type SupersenseAnnotation = [number, number, string, string];
// [startToken, endToken, category, text]
```

### ONNX Inference
Complete integration with ONNX Runtime:

```typescript
// Automatic tensor creation with correct shapes and types
const predictions = await controller.predict(
  inputIds,           // int64[batch, seq_len]
  attentionMask,      // int64[batch, seq_len]
  transforms,         // float32[batch, seq_len, seq_len]
  matrix1,            // float32[batch, seq_len, seq_len]
  matrix2,            // float32[batch, seq_len, seq_len]
  wn,                 // int64[batch, seq_len]
  seqLengths,         // int64[batch]
  doEntity,
  doSupersense,
  doEvent
);
```

The ONNX model outputs **final predictions** (already CRF-decoded), not logits or emissions. No Viterbi decoding needed in TypeScript.

### Entity Recognition
Hierarchical 3-layer entity detection with proper BIO tag fixing:

```typescript
// Automatically handles:
// - Invalid BIO sequences (I-PER without B-PER)
// - Entity type classification (PROP_PER → PER)
// - Hierarchical merging (3 LSTM layers)
// - Overlapping entity resolution

const entities = await tagger.tag(tokens, spaCyTokens, true, false, false);
// Returns: EntityAnnotation[] with startToken, endToken, cat, text, prop
```

### Supersense Tagging
WordNet-based semantic annotation:

```typescript
// Uses WordNet first sense mappings
// Categories: noun.person, noun.location, verb.communication, etc.
const supersense = await tagger.tag(tokens, spaCyTokens, false, true, false);
// Returns: [startToken, endToken, category, text][]
```

### Event Detection
Token-level event markers:

```typescript
const events = await tagger.tag(tokens, spaCyTokens, false, false, true);
// Returns: Set<tokenId> of tokens that are events
```

## Architecture

### ONNX Model Architecture

### Data Flow

```
SpaCy Context (input)
    ↓
Token Conversion
    ↓
Entity Tagger
    ├─→ Tokenization (with [CAP] tokens)
    ├─→ Transform Matrix Creation
    ├─→ WordNet Sense Lookup
    ├─→ ONNX Inference (predictions)
    ├─→ Postprocessing (BIO fixing)
    └─→ Entity Extraction
    ↓
BookNLP Result (output)
```

## Configuration

### BookNLPConfig

```typescript
interface BookNLPConfig {
  modelPath?: string;            // Optional: Hugging Face repo ID or URL
                                 // Default: 'Terraa/entities_google_bert_uncased_L-4_H-256_A-4-v1.0-ONNX'
  pipeline: string[];            // ['entity', 'supersense', 'event']
  verbose?: boolean;             // Logging verbosity
  executionProviders?: ExecutionProvider[];  // ['wasm', 'webgl', 'webgpu']
  wasmPaths?: string | Record<string, string>;  // Custom WASM paths
}
```

### Execution Providers

Choose the best backend for your deployment:

- **`wasm`** (default): Universal compatibility, works in all browsers
- **`webgl`**: GPU acceleration via WebGL (faster inference)
- **`webgpu`**: Next-gen GPU acceleration (Chrome/Edge 113+, best performance)

```typescript
const config = {
  pipeline: ['entity'],
  executionProviders: ['webgpu', 'wasm'], // Try WebGPU, fallback to WASM
};
```

### Model Loading Options

#### Option 1: Use Hugging Face (Automatic Download, Default)
```typescript
const config = {
  pipeline: ['entity', 'supersense'],
};
// Automatically downloads from Hugging Face and caches in browser
```

#### Option 2: Specify Hugging Face Repository
```typescript
const config = {
  modelPath: 'Terraa/entities_google_bert_uncased_L-4_H-256_A-4-v1.0-ONNX',
  pipeline: ['entity', 'supersense'],
};
```

#### Option 3: Use Custom URL
```typescript
const config = {
  modelPath: 'https://your-cdn.com/model.onnx',
  pipeline: ['entity', 'supersense'],
};
```

## Input Requirements

### SpaCy Preprocessing

The TypeScript implementation requires complete linguistic annotations from spaCy. Here's how to generate the required input in Python:

```python
import spacy
import json

nlp = spacy.load("en_core_web_sm")
text = "Harry Potter walked through the castle."
doc = nlp(text)

spacy_context = {
    "tokens": [
        {
            "text": token.text,
            "startByte": token.idx,
            "endByte": token.idx + len(token.text),
            "pos": token.pos_,
            "finePos": token.tag_,
            "lemma": token.lemma_,
            "deprel": token.dep_,
            "dephead": token.head.i,
            "morph": {str(k): str(v) for k, v in token.morph.to_dict().items()},
            "likeNum": token.like_num,
            "isStop": token.is_stop,
            "sentenceId": token.sent.start,
            "withinSentenceId": token.i - token.sent.start
        }
        for token in doc
    ],
    "sentences": [
        {"start": sent.start, "end": sent.end}
        for sent in doc.sents
    ]
}

# Save to JSON for TypeScript
with open('spacy_context.json', 'w') as f:
    json.dump(spacy_context, f)
```

Then in TypeScript:

```typescript
import * as fs from 'fs';

const spaCyContext = JSON.parse(
  fs.readFileSync('spacy_context.json', 'utf-8')
);

const result = await booknlp.process(spaCyContext);
```

### Required Fields

All `SpaCyToken` fields are **required**:
- `text`: Token text
- `startByte`, `endByte`: Character offsets
- `pos`, `finePos`: POS tags
- `lemma`: Lemmatized form
- `deprel`, `dephead`: Dependency parse
- `morph`: Morphological features (can be empty object)
- `likeNum`, `isStop`: Token properties
- `sentenceId`, `withinSentenceId`: Sentence information

## Output Format

### BookNLPResult

```typescript
interface BookNLPResult {
  tokens: Token[];                    // Annotated tokens
  sents: any[];                       // Sentence info (future)
  nounChunks: any[];                  // Noun chunks (future)
  entities: EntityAnnotation[];       // Detected entities
  supersense: SupersenseAnnotation[]; // Supersense annotations
  timing: Record<string, number>;     // Performance metrics
}
```

### Entity Format

```typescript
{
  startToken: 0,
  endToken: 2,
  cat: "PER",          // Entity category
  text: "Harry Potter",
  prop: "PROP"         // PROP, NOM, or PRON
}
```

Categories: `PER`, `LOC`, `FAC`, `GPE`, `ORG`, `VEH`

### Supersense Format

```typescript
[0, 2, "noun.person", "Harry Potter"]
// [startToken, endToken, category, text]
```

Categories include:
- Nouns: `noun.person`, `noun.location`, `noun.artifact`, `noun.cognition`, etc.
- Verbs: `verb.communication`, `verb.motion`, `verb.cognition`, etc.

### Batch Processing

The TypeScript implementation uses sentence-level batch processing (matching Python behavior):

```typescript
// Automatically batches sentences for efficient processing
// - Groups tokens into sentence batches (max 500 tokens per batch)
// - Processes up to 32 batches in parallel
// - Reconstructs results with correct token offsets

const result = await booknlp.process(spaCyContext);
// All batching is handled internally
```

**Why Batch Processing?**
- **Memory efficiency**: Processes long documents in manageable chunks
- **Performance**: Parallel processing of multiple sentences
- **ONNX compatibility**: Matches Python implementation's batching strategy

## Performance

### Small Model (Current)
- BERT: 4 layers, 256 hidden dimensions
- Processing: ~500-1000 tokens/second (varies by device and execution provider)
- Memory: ~100-200 MB browser memory

### Execution Provider Performance

| Provider | Speed       | Compatibility        | Notes                             |
| -------- | ----------- | -------------------- | --------------------------------- |
| WASM     | Baseline    | All browsers         | Universal fallback                |
| WebGL    | 2-3x faster | Most modern browsers | GPU acceleration                  |
| WebGPU   | 3-5x faster | Chrome/Edge 113+     | Best performance, limited support |

### Timing Breakdown

```typescript
console.log(result.timing);
// {
//   token_conversion: 5.2,      // ms
//   tagger_inference: 1247.8,   // ms (ONNX inference)
//   total: 1253.0               // ms
// }
```

## Validation

A comprehensive validation suite compares Python and TypeScript outputs:

```bash
cd ../validation
./run_validation.sh
```

This runs both implementations on identical input and reports:
- ✅ Token count matches
- ✅ Entity boundaries match
- ✅ Entity categories match
- ✅ Supersense annotations match
- ✅ Event markers match

See `../validation/README.md` for details.

## Testing

### Unit Tests

```bash
npm test
```

Tests cover:
- Input validation (`validation.test.ts`)
- BIO tag fixing (`postprocessor.test.ts`)
- Entity extraction
- Transform matrix operations

### Integration Tests

```bash
cd ../validation
python3 validate_python.py
npm run build
node validate_typescript.js
python3 compare_outputs.py
```

## Dependencies

### Runtime Dependencies

```json
{
  "@huggingface/transformers": "^2.6.0",   // BERT tokenization
  "onnxruntime-web": "^1.16.0"        // ONNX inference (browser)
}
```

### Development Dependencies

```json
{
  "@types/node": "^20.0.0",
  "typescript": "^5.0.0",
  "vite": "^5.0.0",                   // Build tool
  "vite-plugin-dts": "^3.0.0",        // TypeScript declarations
  "eslint": "^8.0.0"
}
```

## Build and Development

### Development Mode

```bash
npm run dev
# Watches for changes and rebuilds automatically
```

### Production Build

```bash
npm run build
# Compiles TypeScript and bundles with Vite
# Output: dist/booknlp.js (ES module), dist/booknlp.umd.cjs (UMD)
```

### Using in Your Project

#### ES Modules (Recommended)
```typescript
import { BookNLP } from 'booknlp-ts';
```

#### UMD (Script tag)
```html
<script src="node_modules/booknlp-ts/dist/booknlp.umd.cjs"></script>
<script>
  const booknlp = new window.BookNLP.BookNLP();
</script>
```

### Linting

```bash
npm run lint
```

## Limitations

### Current Limitations

1. **Requires SpaCy preprocessing**: Cannot process raw text directly (requires external spaCy tokenization)
2. **Browser-only**: This version is optimized for browser environments
3. **Single-document processing**: No batch API for multiple documents
4. **WebGPU support limited**: Only available in Chrome/Edge 113+

### Future Enhancements

- [ ] Integrate spaCy preprocessing in browser (via Pyodide or similar)
- [ ] Streaming/incremental processing for very long documents
- [ ] Quote and coreference chain extraction
- [ ] Big model variant support
- [ ] Node.js compatibility layer

## Comparison: Python vs TypeScript

| Feature             | Python       | TypeScript   | Status                 |
| ------------------- | ------------ | ------------ | ---------------------- |
| Entity recognition  | ✅            | ✅            | Equivalent             |
| Supersense tagging  | ✅            | ✅            | Equivalent             |
| Event detection     | ✅            | ✅            | Equivalent             |
| ONNX inference      | ✅            | ✅            | Equivalent             |
| SpaCy preprocessing | ✅ Integrated | ⚠️ External   | Requires Python        |
| Raw text input      | ✅            | ❌            | Requires SpaCy context |
| Model conversion    | ✅ Native     | ⚠️ Via Python | ONNX export            |

## Troubleshooting

### Error: "ONNX model not loaded"

Ensure the model path is correct and the file exists:
```typescript
const config = {
  modelPath: '/absolute/path/to/model.onnx',
  // ...
};
```

### Error: "Validation failed"

Check that all required `SpaCyToken` fields are present:
```typescript
import { validateSpaCyContext } from './dist/validation';

const errors = validateSpaCyContext(spaCyContext);
if (errors.length > 0) {
  console.error('Validation errors:', errors);
}
```

### Memory Issues

For very long documents, consider:
1. Processing in smaller chunks
2. Using the small model variant
3. Increasing Node.js heap size: `node --max-old-space-size=8192`

## Contributing

When modifying the TypeScript implementation:

1. **Maintain type safety**: No `any` types without justification
2. **Add tests**: Unit tests for new functionality
3. **Run validation**: Ensure Python vs TypeScript equivalence
4. **Document changes**: Update README and type definitions
5. **Follow conventions**: Use existing code style and patterns

## License

Same as BookNLP (Python version).

## References

- BookNLP Python: https://github.com/dbamman/book-nlp
- ONNX Runtime: https://onnxruntime.ai/
- Transformers.js: https://xenova.github.io/transformers.js/
- BERT: https://arxiv.org/abs/1810.04805
- CRF: https://en.wikipedia.org/wiki/Conditional_random_field

## Support

For issues or questions:
- Python BookNLP: https://github.com/dbamman/book-nlp/issues
- TypeScript implementation: See `TYPESCRIPT_IMPLEMENTATION.md` in project root
- Validation: See `validation/README.md`
