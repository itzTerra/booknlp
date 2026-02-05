PROMPT TEMPLATES

### ANALYSIS

Go through `booknlp/` Python files (including `common/` dir) and `booknlp-ts/` TS files. They should compute equivalent book context, aside from that the typescript version uses onnx model instead of the Tagger class and spacy context as input, without computing it itself. Some conversions in the Tagger tag_all function after the underlying model prediction are done in typescript and not in onnx. Python model before conversion to onnx can be found in `conversion/convert_tagger_to_onnx.py`. Logging is not needed in TS version.

Give me an overview of the 1:1 mapping of files. Start in logical order from the entrypoints to the util functions. Any inconsitency must be reported. Any naming that could be clearer on the typescript side (its a rewrite of the python version) should be recommended to me. There ARE errors. It will be your fault if you dont find them. Be ruthless.

### Fix

`booknlp/` Python files (including `common/` dir) and `booknlp-ts/` TS files should compute equivalent book context, aside from that the typescript version uses onnx model instead of the Tagger class and spacy context as input, without computing it itself. Some things of the Tagger tag_all function after the underlying model prediction is done in typescript and not in onnx.

The error analysis brough up the following errors, please fix them so that the logic is as identical to the Python source of truth as possible:

### Focused fix

`booknlp/` Python files (including `common/` dir) and `booknlp-ts/` TS files should compute equivalent book context, aside from that the typescript version uses onnx model instead of the Tagger class and spacy context as input, without computing it itself. Some things of the Tagger tag_all function after the underlying model prediction is done in typescript and not in onnx.

The comparison function revealed a lot of mismatches, both event and NER. Find the issue in the typescript version, which should mirror the python version as much as possible.

Sample of the mismatches:
⚠️  Token 48476 event mismatch: 'triumph' - Python=False, TypeScript=True
⚠️  Token 48482 event mismatch: 'admired' - Python=False, TypeScript=True
⚠️  Token 48570 NER mismatch: 'common' - Python=None, TypeScript=LOC
⚠️  Token 48675 event mismatch: 'wished' - Python=False, TypeScript=True
⚠️  Token 48776 event mismatch: 'exclamation' - Python=True, TypeScript=False
⚠️  Token 48791 event mismatch: 'looked' - Python=True, TypeScript=False
⚠️  Token 49196 event mismatch: 'gone' - Python=True, TypeScript=False
⚠️  Token 49486 event mismatch: 'came' - Python=True, TypeScript=False
⚠️  Token 49539 event mismatch: 'discovery' - Python=False, TypeScript=True
⚠️  Token 49844 event mismatch: 'cut' - Python=True, TypeScript=False
⚠️  Token 49872 event mismatch: 'declaring' - Python=True, TypeScript=False
⚠️  Token 50627 event mismatch: 'sensations' - Python=False, TypeScript=True
⚠️  Token 50647 event mismatch: 'silence' - Python=False, TypeScript=True
⚠️  Token 50673 NER mismatch: 'Charming' - Python=None, TypeScript=PER
⚠️  Token 50686 event mismatch: 'confesses' - Python=False, TypeScript=True
⚠️  Token 50794 event mismatch: 'wishing' - Python=False, TypeScript=True
⚠️  Token 50878 NER mismatch: 'Miss' - Python=None, TypeScript=PER
⚠️  Token 51278 event mismatch: 'welcomed' - Python=True, TypeScript=False
⚠️  Token 51283 event mismatch: 'delight' - Python=True, TypeScript=False
⚠️  Token 51348 NER mismatch: 'Mr.' - Python=None, TypeScript=PER
⚠️  Token 51851 NER mismatch: 'Mr.' - Python=None, TypeScript=PER
⚠️  Token 51907 event mismatch: 'professed' - Python=True, TypeScript=False
⚠️  Token 52146 NER mismatch: 'Woodhouse' - Python=None, TypeScript=FAC
⚠️  Token 52363 NER mismatch: 'Donwell' - Python=None, TypeScript=FAC
⚠️  Token 52979 event mismatch: 'went' - Python=True, TypeScript=False
⚠️  445 token mismatches found
