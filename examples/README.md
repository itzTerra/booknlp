# BookNLP Validation Suite

This directory (examples/) contains validation scripts to compare Python and TypeScript implementations of BookNLP using real book data.

## Overview

The validation suite ensures that the TypeScript browser-side library produces equivalent results to the original Python implementation. It processes identical input through both implementations and compares:

- Token-level annotations (POS tags, NER labels, event markers)
- Entity recognition results
- Supersense tagging annotations
- Processing timing and performance

## Architecture

### Key Difference: Browser-Based Library

The TypeScript implementation (`booknlp-ts/`) is now a **browser-side library**, not a Node.js server. This means:

- Runs in web browsers (Chrome, Firefox, Safari, Edge, etc.)
- Uses `onnxruntime-web` for in-browser model inference
- Can process documents client-side without server communication
- Validation requires either:
  1. A test framework with browser environment (Vitest, Jest with jsdom)
  2. A headless browser runner (Puppeteer, Playwright)
  3. Node.js with proper ES module support

### Validation Flow

```
Input Text
    ↓
SpaCy Preprocessing (Python, shared baseline)
    ↓
    ├─→ Python BookNLP → python_output.json
    └─→ TypeScript BookNLP → typescript_output.json
         (via Node.js test harness)
    ↓
Compare Outputs → Validation Report
```

## Prerequisites

### Python Requirements
```bash
pip install spacy booknlp
python -m spacy download en_core_web_sm
```

### TypeScript/Browser Requirements
```bash
cd ../booknlp-ts
pnpm install  # or npm install
pnpm run build
```

### Optional: Test Harness for Browser Testing

For running the TypeScript validation in Node.js, install one of:

```bash
# Option 1: Using tsx for TypeScript execution
pnpm add -D tsx

# Option 2: Using Vitest (recommended for browser library testing)
pnpm add -D vitest jsdom

# Option 3: Using Jest with jsdom
npm add -D jest-environment-jsdom
```

## Running Validation

### Quick Start (Automated)
```bash
chmod +x run_validation.sh
./run_validation.sh
```

This runs all validation steps automatically:
1. Checks dependencies
2. Builds TypeScript library
3. Runs Python BookNLP
4. Runs TypeScript BookNLP via test harness
5. Compares outputs

### Manual Validation

#### Step 1: Run Python validation
```bash
python3 validate_python.py
```
This generates `python_output.json` containing:
- SpaCy preprocessing context (shared baseline)
- BookNLP token annotations
- Entity recognition results
- Supersense annotations
- Timing information

#### Step 2: Build TypeScript library
```bash
cd ../booknlp-ts
pnpm run build
```

#### Step 3: Run TypeScript validation

**Using tsx (simplest for Node.js):**
```bash
cd examples/
npx tsx validate_typescript.ts
```

**Using Vitest (recommended):**
```bash
cd ../booknlp-ts
pnpm add -D vitest jsdom
pnpm run test ../examples/validate_typescript.ts
```

**Using Puppeteer (true browser environment):**
Create a test runner (example: `browser-test.js`):
```javascript
const puppeteer = require('puppeteer');
const { BookNLP } = require('./booknlp-ts/dist/booknlp.js');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  // Load and run BookNLP in browser context
  // ... test code here ...
  
  await browser.close();
})();
```

#### Step 4: Compare outputs
```bash
python3 compare_outputs.py
```
This compares the two outputs and reports:
- ✅ Exact matches
- ⚠️  Minor differences (acceptable tolerance)
- ❌ Significant mismatches (requires investigation)

## Output Files

- **`python_output.json`**: Complete Python BookNLP results
- **`typescript_output.json`**: Complete TypeScript BookNLP results
- **`output_python/`**: Python intermediate files
- **`output_typescript/`**: TypeScript intermediate files

## Test Input

The validation suite uses `158_emma.txt` (Jane Austen's Emma) as the input text. This full-length novel provides comprehensive testing:

- **Person entities**: Emma Woodhouse, Mr. Knightley, Harriet Smith, and many more characters
- **Location entities**: Highbury, Hartfield, Donwell Abbey
- **Organizations**: Various social groups and establishments
- **Events**: Complex narrative events throughout the book
- **Supersense**: Rich variety of semantic categories across dialogue and narration

This real-world text ensures that the validation covers edge cases and long-document processing.

## Custom Test Input

By default, the validation uses `158_emma.txt`. To validate with a different text file:

1. Place your text file in the `examples/` directory
2. Update the input file path in `validate_python.py`:

```python
input_file = Path(__file__).parent / "your_text_file.txt"
```

3. Re-run the validation suite

## Expected Differences

Minor differences between implementations may occur due to:

1. **Floating-point precision**: Different runtimes (PyTorch vs ONNX-Web) may have slight numerical differences
2. **Browser environment**: JavaScript number handling differs slightly from Python
3. **Tokenization**: Edge cases in WordPiece tokenization
4. **Timing**: Different platforms affect timing measurements

These differences are typically acceptable if:
- Entity boundaries match
- Entity types match
- Event markers match
- Supersense categories match

## Troubleshooting

### Error: "Python output not found"
Run `validate_python.py` first to generate the Python baseline.

### Error: "Cannot find module 'booknlp-ts'"
Ensure the TypeScript library is built:
```bash
cd ../booknlp-ts
pnpm install
pnpm run build
```

### Error: "ERR_MODULE_NOT_FOUND" when running validate_typescript.ts
The browser library requires a proper ES module environment. Try:
```bash
npx tsx validate_typescript.ts
```

Or use a test framework like Vitest:
```bash
pnpm add -D vitest
# Create vitest config and test file
```

### Mismatch in validation results
1. Check that both implementations use the same input text
2. Verify the SpaCy context is identical between runs
3. Review specific token/entity differences in the output JSON
4. Check for browser compatibility issues if using browser environment

### TypeScript compilation errors
Ensure TypeScript dependencies are updated:
```bash
cd ../booknlp-ts
pnpm install
pnpm run build
```

## Continuous Validation

For CI/CD integration, use the automated script:
```bash
./run_validation.sh
```

Exit codes:
- `0`: Validation passed (outputs match or have acceptable differences)
- `1`: Validation failed (missing dependencies or significant mismatches)

## Browser-Specific Testing

For validating behavior in specific browsers, consider:

1. **Unit Testing**: Use Vitest with jsdom for fast validation
2. **Browser Testing**: Use Playwright for cross-browser validation
3. **Performance Testing**: Benchmark in different browsers using Lighthouse

Example Vitest setup:
```typescript
import { describe, it, expect } from 'vitest';
import { BookNLP } from '../booknlp-ts/dist/index';

describe('BookNLP TypeScript Browser Library', () => {
  it('should process SpaCy context correctly', async () => {
    const booknlp = new BookNLP();
    const result = await booknlp.process(spaCyContext);
    expect(result.tokens).toBeDefined();
  });
});
```

## Contributing

When modifying the TypeScript browser library:
1. Rebuild the library after changes
2. Run validation suite to ensure equivalence with Python
3. Investigate any new mismatches
4. Test in target browser environments (or use Vitest/Playwright)
5. Update validation scripts if architecture changes

## License

Same as BookNLP project.
