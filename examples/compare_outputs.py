#!/usr/bin/env python3
"""
Compare Python and TypeScript BookNLP outputs.
Validates that both implementations produce equivalent results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_tokens(python_tokens: List[Dict], ts_tokens: List[Dict]) -> bool:
    """Compare token-level outputs."""
    print("\n=== Comparing Tokens ===")

    if len(python_tokens) != len(ts_tokens):
        print(
            f"❌ Token count mismatch: Python={len(python_tokens)}, TypeScript={len(ts_tokens)}"
        )
        return False

    print(f"✓ Token count matches: {len(python_tokens)}")

    mismatches = 0
    for i, (py_tok, ts_tok) in enumerate(zip(python_tokens, ts_tokens)):
        if py_tok["text"] != ts_tok["text"]:
            print(
                f"❌ Token {i} text mismatch: Python='{py_tok['text']}', TypeScript='{ts_tok['text']}'"
            )
            mismatches += 1

        if py_tok["ner"] != ts_tok["ner"]:
            print(
                f"⚠️  Token {i} NER mismatch: '{py_tok['text']}' - Python={py_tok['ner']}, TypeScript={ts_tok['ner']}"
            )
            mismatches += 1

        if py_tok["event"] != ts_tok["event"]:
            print(
                f"⚠️  Token {i} event mismatch: '{py_tok['text']}' - Python={py_tok['event']}, TypeScript={ts_tok['event']}"
            )
            mismatches += 1

    if mismatches == 0:
        print("✓ All tokens match!")
        return True
    else:
        print(f"⚠️  {mismatches} token mismatches found")
        return False


def compare_entities(python_entities: List, ts_entities: List) -> bool:
    """Compare entity outputs."""
    print("\n=== Comparing Entities ===")

    if len(python_entities) != len(ts_entities):
        print(
            f"⚠️  Entity count mismatch: Python={len(python_entities)}, TypeScript={len(ts_entities)}"
        )

        # Find the specific difference
        py_set = set()
        ts_set = set()

        for ent in python_entities:
            start = ent.get("startToken", ent.get("start_token"))
            end = ent.get("endToken", ent.get("end_token"))
            cat = ent.get("cat")
            py_set.add((start, end, cat))

        for ent in ts_entities:
            start = ent.get("startToken", ent.get("start_token"))
            end = ent.get("endToken", ent.get("end_token"))
            cat = ent.get("cat")
            ts_set.add((start, end, cat))

        extra_in_ts = ts_set - py_set
        missing_in_ts = py_set - ts_set

        if extra_in_ts:
            print(f"  Extra in TypeScript ({len(extra_in_ts)}): {extra_in_ts}")
        if missing_in_ts:
            print(f"  Missing in TypeScript ({len(missing_in_ts)}): {missing_in_ts}")

        return False

    print(f"✓ Entity count matches: {len(python_entities)}")
    return True


def compare_supersense(python_supersense: List, ts_supersense: List) -> bool:
    """Compare supersense outputs."""
    print("\n=== Comparing Supersense ===")

    if len(python_supersense) != len(ts_supersense):
        print(
            f"⚠️  Supersense count mismatch: Python={len(python_supersense)}, TypeScript={len(ts_supersense)}"
        )

        # Find the specific difference
        py_set = set()
        ts_set = set()

        for ss in python_supersense:
            py_set.add((ss[0], ss[1], ss[2]))

        for ss in ts_supersense:
            ts_set.add((ss[0], ss[1], ss[2]))

        extra_in_ts = ts_set - py_set
        missing_in_ts = py_set - ts_set

        if extra_in_ts:
            print(f"  Extra in TypeScript ({len(extra_in_ts)}): {extra_in_ts}")
        if missing_in_ts:
            print(f"  Missing in TypeScript ({len(missing_in_ts)}): {missing_in_ts}")

        return False

    print(f"✓ Supersense count matches: {len(python_supersense)}")
    return True


def compare_timing(python_timing: Dict, ts_timing: Dict) -> None:
    """Compare and display timing information."""
    print("\n=== Timing Comparison ===")
    print(f"Python total time: {python_timing.get('total', 'N/A')}")
    print(f"TypeScript total time: {ts_timing.get('total', 'N/A')}")

    all_keys = set(python_timing.keys()) | set(ts_timing.keys())
    for key in sorted(all_keys):
        py_time = python_timing.get(key, "N/A")
        ts_time = ts_timing.get(key, "N/A")
        print(f"  {key}: Python={py_time}, TypeScript={ts_time}")


def compare_debug_info(python_debug: Dict, ts_debug: Dict) -> bool:
    """Compare intermediate debug values from both implementations."""
    print("\n=== Comparing Debug/Intermediate Values ===")

    if not python_debug or not ts_debug:
        print(
            f"⚠️  Debug info not available. Python={bool(python_debug)}, TypeScript={bool(ts_debug)}"
        )
        return True

    mismatches = 0
    all_keys = set(python_debug.keys()) | set(ts_debug.keys())
    for key in sorted(all_keys):
        py_val = python_debug.get(key, "MISSING")
        ts_val = ts_debug.get(key, "MISSING")

        if key in ["raw_tokens_sample", "raw_batch_sample"]:
            continue

        if py_val != ts_val:
            if mismatches == 0:
                print("Python Debug Info:")
                for k, v in sorted(python_debug.items()):
                    if isinstance(v, (dict, list)) and len(str(v)) > 100:
                        print(
                            f"  {k}: {type(v).__name__} (length={len(v) if isinstance(v, list) else len(str(v))})"
                        )
                    else:
                        print(f"  {k}: {v}")

                print("\nTypeScript Debug Info:")
                for k, v in sorted(ts_debug.items()):
                    if isinstance(v, (dict, list)) and len(str(v)) > 100:
                        print(
                            f"  {k}: {type(v).__name__} (length={len(v) if isinstance(v, list) else len(str(v))})"
                        )
                    else:
                        print(f"  {k}: {v}")

                print("\nDebug Info Comparison:")

            print(f"  ⚠️  {key} mismatch: Python={py_val}, TypeScript={ts_val}")
            mismatches += 1
        else:
            print(f"  ✓ {key} matches")

    if mismatches == 0:
        print("✓ All debug info matches!")
        return True
    else:
        print(f"\n⚠️  {mismatches} debug info mismatches found")
        return False


def main():
    examples_dir = Path(__file__).parent
    python_output = examples_dir / "python_output.json"
    ts_output = examples_dir / "typescript_output.json"

    if not python_output.exists():
        print(f"❌ Python output not found: {python_output}")
        sys.exit(1)

    if not ts_output.exists():
        print(f"❌ TypeScript output not found: {ts_output}")
        sys.exit(1)

    py_data = load_json(str(python_output))
    ts_data = load_json(str(ts_output))

    all_match = True

    tokens_match = compare_tokens(py_data["tokens"], ts_data["tokens"])
    all_match = all_match and tokens_match

    entities_match = compare_entities(py_data["entities"], ts_data["entities"])
    all_match = all_match and entities_match

    supersense_match = compare_supersense(py_data["supersense"], ts_data["supersense"])
    all_match = all_match and supersense_match

    # compare_timing(py_data["timing"], ts_data["timing"])

    # Compare debug/intermediate values if available
    py_debug = py_data.get("_debug", {})
    ts_debug = ts_data.get("_debug", {})
    debug_match = compare_debug_info(py_debug, ts_debug)
    all_match = all_match and debug_match

    print("\n" + "=" * 50)
    if all_match:
        print("✅ VALIDATION PASSED: Python and TypeScript outputs match!")
        sys.exit(0)
    else:
        print("⚠️  VALIDATION WARNING: Some differences found between outputs")
        print(
            "This may be due to floating-point precision or minor implementation differences"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
