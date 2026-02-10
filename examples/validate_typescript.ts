/**
 * Validation script for TypeScript BookNLP implementation (browser-side library).
 * This script uses Puppeteer to run the validation in a headless browser.
 * It processes test input using the BookNLP browser library and outputs results to JSON.
 */

import puppeteer from 'puppeteer';
import * as fs from 'fs';
import * as path from 'path';
import * as http from 'http';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function processWithTypescriptBookNLP(
  spaCyContext: any,
  inputText: string,
  outputFile: string
): Promise<void> {
  const serverPort = 3001;
  // Start local HTTP server to serve ONNX runtime files
  const server = http.createServer((req, res) => {
    const url = req.url || '/';
    let filePath = path.join(__dirname, '..', 'booknlp-ts', url);
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const ext = path.extname(filePath);
      let contentType = 'text/plain';
      if (ext === '.js') contentType = 'application/javascript';
      else if (ext === '.wasm') contentType = 'application/wasm';
      else if (ext === '.mjs') contentType = 'application/javascript';
      res.writeHead(200, { 'Content-Type': contentType });
      fs.createReadStream(filePath).pipe(res);
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
  });
  server.listen(serverPort, 'localhost');

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-web-security', '--allow-running-insecure-content'],
    protocolTimeout: 3000000
  });

  try {
    const page = await browser.newPage();

    // Set up console logging
    page.on('console', (msg) => console.log('PAGE LOG:', msg.text()));

    // Create HTML content with local ONNX runtime
    const htmlContent = `
<!DOCTYPE html>
<html>
<head>
  <title>BookNLP Validation</title>
</head>
<body>
  <script>
    window.testData = ${JSON.stringify({ spaCyContext, inputText })};
    window.wasmPaths = 'http://localhost:${serverPort}/node_modules/onnxruntime-web/dist/';
  </script>
  <script src="http://localhost:${serverPort}/node_modules/onnxruntime-web/dist/ort.webgpu.min.js"></script>
  <script src="http://localhost:${serverPort}/node_modules/onnxruntime-web/dist/ort.min.js"></script>
  <script>
    ort.env.wasm.wasmPaths = window.wasmPaths;
  </script>
  <script src="http://localhost:${serverPort}/dist/booknlp.umd.cjs"></script>
  <script>
    ${fs.readFileSync(path.join(__dirname, 'browser-validation.js'), 'utf-8')}
  </script>
  <script>
    // Run validation and post result
    runValidation().then(result => {
      window.validationResult = result;
    }).catch(error => {
      window.validationResult = { error: error.message, stack: error.stack };
    });
  </script>
</body>
</html>`;

    // Set the page content directly
    await page.setContent(htmlContent);

    // Wait for validation to complete
    await page.waitForFunction(() => window.validationResult !== undefined, { timeout: 3000000 });

    const outputData = await page.evaluate(() => window.validationResult);

    // Check for errors
    if (outputData && outputData.error) {
      console.error('Validation error:', outputData.error);
      if (outputData.stack) {
        console.error('Stack trace:', outputData.stack);
      }
      throw new Error(`Validation failed: ${outputData.error}`);
    }

    fs.writeFileSync(outputFile, JSON.stringify(outputData, null, 2), 'utf-8');

    console.log(`TypeScript BookNLP results saved to ${outputFile}`);
  } finally {
    await browser.close();
    server.close();
  }
}

function loadPythonOutput(pythonOutputFile: string): any {
  const content = fs.readFileSync(pythonOutputFile, 'utf-8');
  return JSON.parse(content);
}

async function main(): Promise<void> {
  const pythonOutputFile = path.join(__dirname, '..', 'examples', 'python_output.json');

  if (!fs.existsSync(pythonOutputFile)) {
    console.error('ERROR: Python output not found!');
    console.error('Please run validate_python.py first to generate python_output.json');
    process.exit(1);
  }

  const pythonOutput = loadPythonOutput(pythonOutputFile);

  const spaCyContext = pythonOutput.spacy_context;
  const inputText = pythonOutput.input_text;

  const outputDir = path.join(__dirname, '..', 'examples', 'output_typescript');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const outputFile = path.join(__dirname, '..', 'examples', 'typescript_output.json');

  await processWithTypescriptBookNLP(spaCyContext, inputText, outputFile);

  console.log('\n✓ TypeScript validation complete!');
}

main().catch((error) => {
  console.error('ERROR during TypeScript validation:', error);
  process.exit(1);
});
