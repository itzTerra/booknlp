## ANALYSIS

`booknlp-ts/` is a Typescript port of the `booknlp/` Python library and should compute equivalent book context. Allowed and planned differences between the versions are: 
- TS version takes spaCy context as input, without computing it itself. 
- TS version uses ONNX model instead of the Tagger class. Some postprocessing of the Tagger tag_all function after the underlying model prediction is done in TS and not in the ONNX model. Python model before conversion to ONNX can be found in `conversion/convert_tagger_to_onnx.py`.
- TS version does not do logging
  
1. Run a subagent to go through `booknlp/` Python files (including `common/` dir) and `booknlp-ts/` TS files to get an overview of the 1:1 mapping of files. Start in logical order from the entrypoints all the way to the util functions. Any inconsitency must be reported. Any naming that could be clearer on the typescript side (its a rewrite of the python version) should be recommended to me. There ARE errors. It will be your fault if you dont find them. Be ruthless.

2. The comparison script revealed a lot of mismatches. The token mismatches are only events. Find the issues in the TS version, which should mirror the python version as much as possible. Fix the issues and add documenting comments with reasoning of the change and location of the corresponding code in the Python files.

Sample of the compare script output:
⚠️  Token 34355 event mismatch: 'entreat' - Python=False, TypeScript=True
⚠️  Token 34488 event mismatch: 'endeavouring' - Python=False, TypeScript=True
⚠️  Token 34617 event mismatch: 'hear' - Python=False, TypeScript=True
⚠️  Token 34640 event mismatch: 'appearance' - Python=True, TypeScript=False
⚠️  Token 34668 event mismatch: 'felt' - Python=True, TypeScript=False
⚠️  Token 34706 event mismatch: 'seen' - Python=True, TypeScript=False
⚠️  59 token mismatches found

=== Comparing Entities ===
⚠️  Entity count mismatch: Python=5073, TypeScript=5043

=== Comparing Supersense ===
⚠️  Supersense count mismatch: Python=9237, TypeScript=9236

=== Comparing Debug/Intermediate Values ===
Python Debug Info:
  batches_count: 10
  extracted_entities_count: 5073
  extracted_supersense_count: 9237
  raw_batch_sample: dict (length=1652)
  raw_tokens_count: 35085
  raw_tokens_sample: list (length=5)

TypeScript Debug Info:
  batches_count: 3
  extracted_entities_count: 5043
  extracted_supersense_count: 9236
  raw_tokens_count: 35085
  raw_tokens_sample: list (length=5)

Debug Info Comparison:
  ⚠️  batches_count mismatch: Python=10, TypeScript=3
  ⚠️  extracted_entities_count mismatch: Python=5073, TypeScript=5043
  ⚠️  extracted_supersense_count mismatch: Python=9237, TypeScript=9236
  ✓ raw_tokens_count matches: 35085

⚠️  3 debug info mismatches found
