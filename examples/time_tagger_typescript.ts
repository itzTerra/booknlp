#!/usr/bin/env ts-node
import puppeteer from 'puppeteer';
import * as fs from 'fs';
import * as path from 'path';
import * as http from 'http';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function processWithTypescriptBookNLP(texts: string[], outputFile: string): Promise<void> {
  const serverPort = 3001;
  const server = http.createServer((req, res) => {
    const url = req.url || '/';
    let filePath = path.join(__dirname, '..', 'booknlp-ts', url);
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const ext = path.extname(filePath);
      let contentType = 'text/plain';
      if (ext === '.js' || ext === '.mjs') contentType = 'application/javascript';
      else if (ext === '.wasm') contentType = 'application/wasm';
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
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
  });

  try {
    const page = await browser.newPage();

    // Collect console logs for debugging
    const pageLogs: string[] = [];
    page.on('console', (msg) => {
      try {
        pageLogs.push(msg.text());
        console.log('PAGE LOG:', msg.text());
      } catch (e) {
        // ignore
      }
    });

    const htmlContent = `
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>BookNLP Timing</title></head>
<body>
  <script>
    window.spacy_contexts = ${JSON.stringify(texts)};
    window.wasmPaths = 'http://localhost:${serverPort}/node_modules/onnxruntime-web/dist/';
  </script>
  <script src="http://localhost:${serverPort}/node_modules/onnxruntime-web/dist/ort.webgpu.min.js"></script>
  <script src="http://localhost:${serverPort}/node_modules/onnxruntime-web/dist/ort.min.js"></script>
  <script src="http://localhost:${serverPort}/dist/booknlp.umd.cjs"></script>
  <script>
    async function runTiming() {
      try {
        const BookNLP = window.BookNLP && window.BookNLP.BookNLP;
        if (!BookNLP) throw new Error('BookNLP not found on window');

        const booknlp = new BookNLP();
        await booknlp.initialize({ pipeline: ['entity', 'supersense', 'event'], wasmPaths: window.wasmPaths });

        const durations = [];
        const resultsSummary = [];
        for (let i = 0; i < window.spacy_contexts.length; i++) {
          const spacyContext = window.spacy_contexts[i];
          const t0 = performance.now();
          try {
            const res = await booknlp.process(spacyContext);
            resultsSummary.push({ tokens: (res.tokens || []).length, entities: (res.entities || []).length });
          } catch (e) {
            resultsSummary.push({ error: String(e) });
          }
          const t1 = performance.now();
          durations.push(t1 - t0);
        }
        console.log('Elapsed:', Math.round(durations.reduce((a, b) => a + b, 0)), 'ms');
        const t0_ = performance.now();
        const res = await booknlp.process_batch(window.spacy_contexts);
        const t1_ = performance.now();
        console.log('Batch elapsed:', Math.round(t1_ - t0_), 'ms');

        return { durations, resultsSummary, page_console: [] };
      } catch (err) {
        return { error: String(err), stack: err && err.stack };
      }
    }

    window.timingPromise = runTiming();
  </script>
</body>
</html>`;

    await page.setContent(htmlContent, { waitUntil: 'load' });

    await page.waitForFunction(() => (window as any).timingPromise !== undefined, { timeout: 0 });

    const outputData = await page.evaluate(async () => {
      return await (window as any).timingPromise;
    });

    if (outputData && outputData.error) {
      throw new Error(`Timing failed: ${outputData.error}`);
    }

    // attach collected console logs
    try {
      outputData._debug = outputData._debug || {};
      outputData._debug.page_console = pageLogs;
    } catch (e) {
      // ignore
    }

    fs.writeFileSync(outputFile, JSON.stringify(outputData, null, 2), 'utf-8');
    console.log(`Wrote timing output to ${outputFile}`);
  } finally {
    await browser.close();
    server.close();
  }
}

function loadTexts(jsonPath: string): any[] {
  const content = fs.readFileSync(jsonPath, 'utf-8');
  const data = JSON.parse(content);
  return data.map((item: any) => item.spacy_context || {});
}

async function main(): Promise<void> {
  const arg = process.argv[2];
  const jsonPath = arg ? path.resolve(arg) : path.join(__dirname, 'speed.json');
  if (!fs.existsSync(jsonPath)) {
    console.error('speed.json not found. Provide path as first arg.');
    process.exit(1);
  }
  const texts = loadTexts(jsonPath);
  console.log(`Loaded ${texts.length} spacy_contexts from ${jsonPath}`);

  const outDir = path.join(__dirname, 'output');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, 'typescript_speed.json');

  await processWithTypescriptBookNLP(texts, outFile);
}

main().catch((err) => {
  console.error('ERROR:', err);
  process.exit(1);
});
