import { type NounChunk } from 'booknlp-ts';

// const logEl = document.getElementById('log') as HTMLPreElement;
// const progressEl = document.getElementById('progress') as HTMLElement;
// const pctEl = document.getElementById('pct') as HTMLElement;
// const initBtn = document.getElementById('init') as HTMLButtonElement;
// const clearBtn = document.getElementById('clear') as HTMLButtonElement;

// function log(...args: any[]) {
//   console.log(...args);
//   logEl.textContent = `${new Date().toISOString()} - ${args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ')}\n` + logEl.textContent;
// }

// function setProgress(p: number) {
//   const pct = Math.round(p * 100);
//   progressEl.style.width = `${pct}%`;
//   pctEl.textContent = `${pct}%`;
// }

// initBtn.addEventListener('click', async () => {
//   initBtn.disabled = true;
//   log('Initializing BookNLP pipeline...');

//   const b = new BookNLP();
//   try {
//     await b.initialize({ pipeline: ['entity','supersense','event'], cacheName: "my-cache" }, (progress) => {
//       setProgress(progress);
//     });
//     log('Initialization complete');
//     setProgress(1);
//   } catch (e) {
//     log('Initialization failed', e);
//     initBtn.disabled = false;
//   }
// });

// // expose helpers for manual testing in console
// (window as any).booknlp_demo = { setProgress, log };
