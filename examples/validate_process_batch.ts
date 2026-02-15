/**
 * Puppeteer runner to validate process_batch in browser environment.
 * Similar to validate_typescript.ts but for the batch test.
 */

import puppeteer from 'puppeteer';
import * as fs from 'fs';
import * as path from 'path';
import * as http from 'http';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function processInBrowser(pythonOutput: any, outputFile: string): Promise<void> {
  const serverPort = 3002;
  const server = http.createServer((req, res) => {
    const url = req.url || '/';
    // Check in booknlp-ts first, then fall back to repository root (examples etc.)
    const candidates = [
      path.join(__dirname, '..', 'booknlp-ts', url),
      path.join(__dirname, '..', url),
    ];

    let found: string | null = null;
    for (const p of candidates) {
      if (fs.existsSync(p) && fs.statSync(p).isFile()) {
        found = p;
        break;
      }
    }

    if (found) {
      const ext = path.extname(found);
      let contentType = 'text/plain';
      if (ext === '.js') contentType = 'application/javascript';
      else if (ext === '.wasm') contentType = 'application/wasm';
      else if (ext === '.mjs') contentType = 'application/javascript';
      res.writeHead(200, { 'Content-Type': contentType });
      fs.createReadStream(found).pipe(res);
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
  });
  server.listen(serverPort, 'localhost');

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-web-security', '--allow-running-insecure-content'],
    protocolTimeout: 3000000,
  });

  try {
    const page = await browser.newPage();
    const pageLogs: string[] = [];
    page.on('console', async (msg) => {
      try {
        const args = msg.args();
        const parts: string[] = [];
        for (const a of args) {
          try {
            const val = await a.jsonValue();
            parts.push(typeof val === 'string' ? val : JSON.stringify(val));
          } catch (e) {
            try {
              parts.push(String(await a.toString()));
            } catch {
              parts.push('[unserializable]');
            }
          }
        }
        const text = parts.length > 0 ? parts.join(' ') : msg.text();
        const entry = `${msg.type().toUpperCase()}: ${text}`;
        pageLogs.push(entry);
        console.log('PAGE:', entry);
      } catch (e) {
        // best-effort
        try { console.log('PAGE: (console handler error)', e); } catch {}
      }
    });

    page.on('pageerror', (err) => {
      try {
        const entry = `PAGE ERROR: ${err && err.message ? err.message : String(err)}\n${err && err.stack ? err.stack : ''}`;
        pageLogs.push(entry);
        console.error(entry);
      } catch (e) {
        try { console.error('PAGE: (pageerror handler error)', e); } catch {}
      }
    });

    const spaCyContext = pythonOutput.spacy_context || pythonOutput.spacyContext || pythonOutput;
    const htmlContent = `<!DOCTYPE html>
<html>
<head><title>Process Batch Validation</title></head>
<body>
  <script>
    window.testData = ${JSON.stringify({ spaCyContext })};
    window.wasmPaths = 'http://localhost:${serverPort}/node_modules/onnxruntime-web/dist/';
  </script>
  <script src="http://localhost:${serverPort}/node_modules/onnxruntime-web/dist/ort.webgpu.min.js"></script>
  <script src="http://localhost:${serverPort}/node_modules/onnxruntime-web/dist/ort.min.js"></script>
  <script>
    ort.env.wasm.wasmPaths = window.wasmPaths;
  </script>
  <script src="http://localhost:${serverPort}/dist/booknlp.umd.cjs"></script>
  <script src="http://localhost:${serverPort}/examples/process-batch-validation.js"></script>
  <script>
    window.runProcessBatchValidation().then(result => { window.validationResult = result; }).catch(err => { window.validationResult = { error: err.message, stack: err.stack }; });
  </script>
</body>
</html>`;

    await page.setContent(htmlContent);

    await page.waitForFunction(() => (window as any).validationResult !== undefined, { timeout: 3000000 });

    const outputData = await page.evaluate(() => (window as any).validationResult);

    if (outputData && outputData.error) {
      throw new Error(`Validation failed: ${outputData.error}`);
    }

    try {
      outputData._debug = outputData._debug || {};
      outputData._debug.page_console = pageLogs;
    } catch (e) {}

    fs.writeFileSync(outputFile, JSON.stringify(outputData, null, 2), 'utf-8');
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
  const args = process.argv.slice(2);
  let pythonOutputFileArg: string | undefined;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--python-output' && args[i + 1]) { pythonOutputFileArg = args[i + 1]; break; }
    if (args[i].startsWith('--python-output=')) { pythonOutputFileArg = args[i].split('=')[1]; break; }
  }

  const fullPath = path.join(__dirname, '..', 'examples', 'output', 'python_output.json');
  const minimalPath = path.join(__dirname, '..', 'examples', 'output', 'python_minimal.json');

  const pythonOutputFile = pythonOutputFileArg || process.env.PYTHON_OUTPUT || (fs.existsSync(minimalPath) ? minimalPath : fullPath);
  if (!fs.existsSync(pythonOutputFile)) {
    throw new Error(`Python output not found at ${pythonOutputFile}. Run validate_python.py to generate python_output.json or provide --python-output`);
  }

  const pythonOutput = loadPythonOutput(pythonOutputFile);
  const outputFile = path.join(__dirname, '..', 'examples', 'output', 'process_batch_output.json');

  // Attempt to load spacy contexts from 1.json, 2.json, 3.json and merge their tokens.
  function extractSpaCyContext(obj: any): any | null {
    if (!obj) return null;
    if (Array.isArray(obj)) {
      for (const item of obj) {
        if (item && Array.isArray(item.tokens) && item.tokens.length > 0) return item.spacy_context || item.spacyContext || item;
      }
    }
    if (obj.spacy_context || obj.spacyContext) return obj.spacy_context || obj.spacyContext;
    if (Array.isArray(obj.tokens)) return obj;
    return null;
  }

  const outDir = path.join(__dirname, '..', 'examples', 'output');
  const parts = ['1.json', '2.json', '3.json'];
  const contexts: any[] = [];
  for (const p of parts) {
    const fp = path.join(outDir, p);
    if (!fs.existsSync(fp)) continue;
    try {
      const data = loadPythonOutput(fp);
      const ctx = extractSpaCyContext(data);
      if (ctx && Array.isArray(ctx.tokens) && ctx.tokens.length > 0) contexts.push(ctx);
    } catch (e) {
      // ignore individual file errors
    }
  }

  if (contexts.length > 0) {
    // Keep the contexts separate (do NOT merge). Pass them as a list so
    // the browser-side script can process them sequentially.
    pythonOutput.spacy_context = contexts;
  }

  await processInBrowser(pythonOutput, outputFile);
}

main().catch((error) => {
  console.error('ERROR during process_batch validation:', error);
  process.exit(1);
});
