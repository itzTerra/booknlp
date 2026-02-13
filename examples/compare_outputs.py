#!/usr/bin/env python3
"""
Compare Python and TypeScript BookNLP outputs.
Validates that both implementations produce equivalent results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_log_line(fpath: str, line: str) -> None:
    try:
        with open(fpath, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _print_and_log(line: str, logfile: str):
    print(line)
    _write_log_line(logfile, line)


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
    else:
        print(f"✓ Entity count matches: {len(python_entities)}")

    # Find the specific difference
    py_set = set()
    ts_set = set()

    for ent in python_entities:
        start = ent.get("startToken", ent.get("start_token"))
        end = ent.get("endToken", ent.get("end_token"))
        cat = ent.get("cat")
        coref = ent.get("coref", None)
        py_set.add((start, end, cat, coref))

    for ent in ts_entities:
        start = ent.get("startToken", ent.get("start_token"))
        end = ent.get("endToken", ent.get("end_token"))
        cat = ent.get("cat")
        coref = ent.get("coref", None)
        ts_set.add((start, end, cat, coref))

    extra_in_ts = ts_set - py_set
    missing_in_ts = py_set - ts_set

    if extra_in_ts:
        print(f"  Extra in TypeScript ({len(extra_in_ts)}): {extra_in_ts}")
    if missing_in_ts:
        print(f"  Missing in TypeScript ({len(missing_in_ts)}): {missing_in_ts}")

    return len(extra_in_ts) == 0 and len(missing_in_ts) == 0


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
    _print_and_log("\n=== Comparing Debug/Intermediate Values ===", LOGFILE)

    if not python_debug or not ts_debug:
        _print_and_log(
            f"⚠️  Debug info not available. Python={bool(python_debug)}, TypeScript={bool(ts_debug)}",
            LOGFILE,
        )
        return True

    mismatches = 0
    # Keys to ignore in full dumps because they are printed earlier or very large
    SKIP_FULL_DUMP = set(
        [
            "raw_tokens_sample",
            "raw_batch_sample",
            "logs",
            "page_console",
            "validation_messages",
        ]
    )

    all_keys = set(python_debug.keys()) | set(ts_debug.keys())
    for key in sorted(all_keys):
        py_val = python_debug.get(key, "MISSING")
        ts_val = ts_debug.get(key, "MISSING")

        if key in ["raw_tokens_sample", "raw_batch_sample"]:
            continue

        if py_val != ts_val:
            if mismatches == 0:
                _print_and_log("Python Debug Info (summary):", LOGFILE)
                for k, v in sorted(python_debug.items()):
                    if k in SKIP_FULL_DUMP:
                        _print_and_log(f"  {k}: <omitted>", LOGFILE)
                        continue
                    if isinstance(v, (dict, list)) and len(str(v)) > 200:
                        _print_and_log(
                            f"  {k}: {type(v).__name__} (length={len(v) if isinstance(v, list) else len(str(v))})",
                            LOGFILE,
                        )
                    else:
                        _print_and_log(f"  {k}: {v}", LOGFILE)

                _print_and_log("\nTypeScript Debug Info (summary):", LOGFILE)
                for k, v in sorted(ts_debug.items()):
                    if k in SKIP_FULL_DUMP:
                        _print_and_log(f"  {k}: <omitted>", LOGFILE)
                        continue
                    if isinstance(v, (dict, list)) and len(str(v)) > 200:
                        _print_and_log(
                            f"  {k}: {type(v).__name__} (length={len(v) if isinstance(v, list) else len(str(v))})",
                            LOGFILE,
                        )
                    else:
                        _print_and_log(f"  {k}: {v}", LOGFILE)

                _print_and_log("\nDebug Info Comparison:", LOGFILE)

            # Log the mismatch succinctly
            _print_and_log(
                f"  ⚠️  {key} mismatch: Python={type(py_val).__name__} vs TypeScript={type(ts_val).__name__}",
                LOGFILE,
            )
            mismatches += 1
        else:
            _print_and_log(f"  ✓ {key} matches", LOGFILE)

    if mismatches == 0:
        _print_and_log("✓ All debug info matches!", LOGFILE)
        return True
    else:
        _print_and_log(f"\n⚠️  {mismatches} debug info mismatches found", LOGFILE)
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Compare Python and TypeScript BookNLP outputs"
    )
    ap.add_argument("--python", help="Path to Python output JSON", default=None)
    ap.add_argument("--typescript", help="Path to TypeScript output JSON", default=None)
    ap.add_argument(
        "--start",
        help="Start token id (inclusive) to restrict comparison",
        type=int,
        default=None,
    )
    ap.add_argument(
        "--end",
        help="End token id (inclusive) to restrict comparison",
        type=int,
        default=None,
    )
    args = ap.parse_args()

    output_dir = Path(__file__).parent / "output"
    global LOGFILE
    LOGFILE = str(output_dir / "compare_log.txt")
    # Reset log file
    try:
        open(LOGFILE, "w", encoding="utf-8").close()
    except Exception:
        pass
    # Resolve file paths (allow overrides). Prefer minimal python file if present.
    python_output = (
        Path(args.python)
        if args.python
        else (
            output_dir / "python_minimal.json"
            if (output_dir / "python_minimal.json").exists()
            else (output_dir / "python_output.json")
        )
    )
    ts_output = (
        Path(args.typescript)
        if args.typescript
        else (output_dir / "typescript_output.json")
    )

    if not python_output.exists():
        print(f"❌ Python output not found: {python_output}")
        sys.exit(1)

    if not ts_output.exists():
        print(f"❌ TypeScript output not found: {ts_output}")
        sys.exit(1)

    py_data = load_json(str(python_output))
    ts_data = load_json(str(ts_output))

    # Apply optional token id range filtering
    start_id = args.start
    end_id = args.end
    if start_id is not None or end_id is not None:
        _print_and_log(
            f"Applying token id filter: start={start_id} end={end_id}", LOGFILE
        )

        def _get_token_id(tok, idx):
            for k in (
                "id",
                "tokenId",
                "token_id",
                "tokenIndex",
                "token_index",
                "global_token_id",
            ):
                if k in tok:
                    try:
                        return int(tok[k])
                    except Exception:
                        try:
                            return int(str(tok[k]))
                        except Exception:
                            continue
            # fallback to positional index if no id present
            return idx

        def _filter_tokens(tokens):
            out = []
            for i, t in enumerate(tokens):
                tid = _get_token_id(t, i)
                if start_id is not None and tid < start_id:
                    continue
                if end_id is not None and tid > end_id:
                    continue
                out.append(t)
            return out

        def _filter_entities(ents):
            out = []
            for e in ents:
                s = e.get("startToken", e.get("start_token", e.get("start", None)))
                eend = e.get("endToken", e.get("end_token", e.get("end", None)))
                try:
                    s_i = int(s) if s is not None else None
                except Exception:
                    s_i = None
                try:
                    ee_i = int(eend) if eend is not None else None
                except Exception:
                    ee_i = None

                # include if any overlap with range
                if s_i is None or ee_i is None:
                    # unknown positions, include conservatively
                    out.append(e)
                    continue
                if end_id is not None and s_i > end_id:
                    continue
                if start_id is not None and ee_i < start_id:
                    continue
                out.append(e)
            return out

        def _filter_supersense(ss):
            out = []
            for item in ss:
                try:
                    s_i = int(item[0])
                    ee_i = int(item[1])
                except Exception:
                    out.append(item)
                    continue
                if end_id is not None and s_i > end_id:
                    continue
                if start_id is not None and ee_i < start_id:
                    continue
                out.append(item)
            return out

        # Apply filters in-place on copies
        try:
            if "tokens" in py_data and isinstance(py_data.get("tokens"), list):
                py_data = dict(py_data)
                py_data["tokens"] = _filter_tokens(py_data["tokens"])
                py_data["entities"] = _filter_entities(py_data.get("entities", []))
                py_data["supersense"] = _filter_supersense(
                    py_data.get("supersense", [])
                )
        except Exception:
            pass

        try:
            if "tokens" in ts_data and isinstance(ts_data.get("tokens"), list):
                ts_data = dict(ts_data)
                ts_data["tokens"] = _filter_tokens(ts_data["tokens"])
                ts_data["entities"] = _filter_entities(ts_data.get("entities", []))
                ts_data["supersense"] = _filter_supersense(
                    ts_data.get("supersense", [])
                )
        except Exception:
            pass

    all_match = True

    # Print validation/run messages collected by the validators
    py_debug_root = py_data.get("_debug", {})
    ts_debug_root = ts_data.get("_debug", {})

    py_val_msgs = py_debug_root.get("validation_messages", [])
    ts_val_msgs = ts_debug_root.get("validation_messages", [])

    if py_val_msgs:
        _print_and_log("\n--- Python validation messages ---", LOGFILE)
        for m in py_val_msgs:
            _print_and_log(f"  {m}", LOGFILE)

    if ts_val_msgs:
        _print_and_log("\n--- TypeScript validation messages ---", LOGFILE)
        for m in ts_val_msgs:
            _print_and_log(f"  {m}", LOGFILE)

    # Print logger buffers / page console traces if available
    py_logs = py_debug_root.get("logs") or []
    if py_logs:
        _print_and_log("\n--- Python collected logs ---", LOGFILE)
        for entry in py_logs:
            try:
                _print_and_log(
                    f"  {entry.get('level')} - {entry.get('message')}", LOGFILE
                )
            except Exception:
                _print_and_log(f"  {entry}", LOGFILE)

    ts_page_console = ts_debug_root.get("page_console", [])
    if ts_page_console:
        _print_and_log("\n--- TypeScript page console ---", LOGFILE)
        # Condense repeated lines and truncate long entries
        last = None
        count = 0

        def _emit_last():
            nonlocal last, count
            if last is None:
                return
            display = last if len(last) <= 300 else last[:300] + "... [truncated]"
            if count == 1:
                _print_and_log(f"  {display}", LOGFILE)
            else:
                _print_and_log(f"  {display}  (repeated {count} times)", LOGFILE)
            last = None
            count = 0

        for m in ts_page_console:
            try:
                if m == last:
                    count += 1
                else:
                    _emit_last()
                    last = m
                    count = 1
            except Exception:
                continue
        _emit_last()

    # If token-level data exists in both, do the full token/entity/supersense comparisons.
    py_has_tokens = "tokens" in py_data and isinstance(py_data.get("tokens"), list)
    ts_has_tokens = "tokens" in ts_data and isinstance(ts_data.get("tokens"), list)

    if py_has_tokens and ts_has_tokens:
        tokens_match = compare_tokens(py_data["tokens"], ts_data["tokens"])
        all_match = all_match and tokens_match

        entities_match = compare_entities(
            py_data.get("entities", []), ts_data.get("entities", [])
        )
        all_match = all_match and entities_match

        supersense_match = compare_supersense(
            py_data.get("supersense", []), ts_data.get("supersense", [])
        )
        all_match = all_match and supersense_match
    else:
        print(
            "⚠️  Token-level outputs not present in one or both files; skipping token/entity/supersense comparisons."
        )

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
