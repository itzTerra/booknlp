/*
 * Browser-side validation script for process_batch test.
 * Runs in a headless browser via Puppeteer.
 */

async function clearBrowserCache() {
    if ('caches' in window) {
        const cacheNames = await caches.keys();
        for (const name of cacheNames) {
            await caches.delete(name);
        }
    }
    console.log('Browser cache cleared');
}

function findDocWithTokens(pythonOutput) {
    if (Array.isArray(pythonOutput)) {
        for (const item of pythonOutput) {
            if (item && Array.isArray(item.tokens) && item.tokens.length > 0) return item;
        }
    } else if (pythonOutput && Array.isArray(pythonOutput.tokens)) {
        return pythonOutput;
    }
    return null;
}

async function runValidation() {
    try {
        await clearBrowserCache();

        const BookNLP = window.BookNLP && (window.BookNLP.BookNLP || window.BookNLP.BookNLP?.default || window.BookNLP.default);
        if (!BookNLP) {
            throw new Error('BookNLP.BookNLP not found on window');
        }

        const spaCyContext = window.testData.spaCyContext;
        if (!spaCyContext) throw new Error('No spaCyContext provided');

        console.log('spaCyContext received');

        const ctxs = [];
        for (const c of spaCyContext) {
            const tokens = c.tokens || [];
            const sentences = c.sentences || (tokens.length > 0 ? [{ root: tokens[0] }] : []);
            ctxs.push({ tokens, sentences, nounChunks: c.nounChunks || [] });
        }
        const config = {
            pipeline: ['entity', 'supersense', 'event'],
            wasmPaths: window.wasmPaths,
        };

        const booknlp = new BookNLP();
        await booknlp.initialize(config);

        // Run sequential processing
        console.log('Running sequential processing...');
        const seq = [];
        for (const ctx of ctxs) {
            try {
                const out = await booknlp.process(ctx);
                seq.push(out);
            } catch (error) {
                console.error('Error processing context:', error.message, error.stack);
            }
        }

        // Run batch processing
        console.log('Running batch processing...');
        let batch = [];
        try {
            batch = await booknlp.process_batch(ctxs);
        } catch (error) {
            console.error('Error processing batch:', error.message, error.stack);
        }

        const mismatches = [];
        for (let i = 0; i < ctxs.length; i++) {
            const a = JSON.stringify(seq[i] || {});
            const b = JSON.stringify(batch[i] || {});
            if (a !== b) {
                mismatches.push({ index: i, sequential: a, batch: b });
            }
        }

        const outputData = {
            pass: mismatches.length === 0,
            mismatches,
            seq_count: seq.length,
            batch_count: batch.length,
        };

        if (!outputData.pass) {
            return outputData;
        }

        return outputData;
    } catch (error) {
        return { error: error.message, stack: error.stack };
    }
}

window.runProcessBatchValidation = runValidation;
