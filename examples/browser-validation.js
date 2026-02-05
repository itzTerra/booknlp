/**
 * Browser-side validation script for BookNLP TypeScript implementation.
 * This runs in a headless browser via Puppeteer.
 */

async function clearBrowserCache() {
    // Clear ServiceWorker cache if available
    if ('caches' in window) {
        const cacheNames = await caches.keys();
        for (const name of cacheNames) {
            await caches.delete(name);
        }
    }

    console.log('Browser cache cleared');
}

async function runValidation() {
    try {
        // Clear all caches before running validation
        await clearBrowserCache();

        // The BookNLP library is already loaded via script tag
        const BookNLP = window.BookNLP.BookNLP;

        if (!BookNLP) {
            throw new Error('BookNLP.BookNLP not found on window');
        }

        // Load test data (this would be injected by Puppeteer)
        const spaCyContext = window.testData.spaCyContext;
        const inputText = window.testData.inputText;

        const config = {
            pipeline: ['entity', 'supersense', 'event'],
            wasmPaths: window.wasmPaths,
        };

        const booknlp = new BookNLP();
        await booknlp.initialize(config);

        const result = await booknlp.process(spaCyContext);

        const outputData = {
            input_text: inputText,
            spacy_context: spaCyContext,
            tokens: result.tokens.map((token) => ({
                text: token.text,
                pos: token.pos,
                ner: token.ner,
                event: token.event,
                tokenId: token.tokenId,
                sentenceId: token.sentenceId,
            })),
            entities: result.entities,
            supersense: result.supersense,
            timing: result.timing,
        };

        // Return result to Puppeteer
        return outputData;
    } catch (error) {
        // Return error information
        return {
            error: error.message,
            stack: error.stack
        };
    }
}

// Expose the function globally
window.runValidation = runValidation;
