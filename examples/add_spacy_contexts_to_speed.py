#!/usr/bin/env python3
"""
Add SpaCy contexts to the entries in examples/speed.json.

Usage:
  python examples/add_spacy_contexts_to_speed.py \
      --input examples/speed.json --output examples/speed_with_spacy.json

The script uses the `SpacyPipeline` from `booknlp.common.pipelines` and
the same `serialize_tag_result` logic used in `validate_python.py`.
"""

from pathlib import Path
import json
from typing import List, Sequence, Any, Dict
import argparse

import spacy
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from booknlp.common.pipelines import SpacyPipeline, Token


def serialize_tag_result(
    toks: List[Token], sents: Sequence, noun_chunks: Sequence
) -> Dict[str, Any]:
    """Serialize spaCy tagging output to the frontend-friendly shape.

    This mirrors the implementation in `examples/validate_python.py`.
    """
    mapped_tokens = []
    for t in toks:
        morph = {}
        if t.morph is not None:
            if hasattr(t.morph, "to_dict"):
                morph = t.morph.to_dict()
            else:
                morph = dict(t.morph)

        mapped_tokens.append(
            {
                "paragraphId": t.paragraph_id,
                "sentenceId": t.sentence_id,
                "withinSentenceId": t.within_sentence_id,
                "tokenId": t.token_id,
                "text": t.text,
                "pos": t.pos,
                "finePos": t.fine_pos,
                "lemma": t.lemma,
                "deprel": t.deprel,
                "dephead": t.dephead,
                "ner": t.ner if hasattr(t, "ner") else None,
                "startByte": t.startByte,
                "endByte": t.endByte,
                "morph": morph,
                "likeNum": t.like_num,
                "isStop": t.is_stop,
                "itext": getattr(t, "itext", t.text.casefold()),
                "inQuote": getattr(t, "inQuote", False),
                "event": getattr(t, "event", False),
            }
        )

    def build_sent_token(tok, sent_start, sent_end, visited=None):
        if visited is None:
            visited = set()
        if tok.i in visited:
            return None
        visited.add(tok.i)

        children = []
        for child in tok.children:
            if child.i >= sent_start and child.i < sent_end:
                child_node = build_sent_token(child, sent_start, sent_end, visited)
                if child_node is not None:
                    children.append(child_node)

        return {
            "text": tok.text,
            "pos_": tok.pos_,
            "dep_": tok.dep_,
            "children": children,
        }

    sentences_out = []
    for s in sents:
        root = s.root
        root_node = build_sent_token(root, s.start, s.end)
        sentences_out.append({"root": root_node, "start": s.start, "end": s.end})

    noun_chunks_out = [
        {"start": nc.start, "end": nc.end, "text": nc.text} for nc in noun_chunks
    ]

    result = {
        "tokens": mapped_tokens,
        "sentences": sentences_out,
        "nounChunks": noun_chunks_out,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Add spaCy contexts to speed.json entries."
    )
    parser.add_argument("--input", "-i", default="speed.json", help="Input JSON file")
    parser.add_argument(
        "--output",
        "-o",
        default="speed_with_spacy.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--model", "-m", default="en_core_web_sm", help="spaCy model name to load"
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nlp = spacy.load(args.model)
    spacy_pipeline = SpacyPipeline(nlp)

    for idx, entry in enumerate(data):
        text = entry.get("text")
        if not text:
            entry["spacy_context"] = None
            continue

        toks, sents, noun_chunks = spacy_pipeline.tag(text)
        spacy_context = serialize_tag_result(toks, sents, noun_chunks)
        entry["spacy_context"] = spacy_context

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} entries with spaCy contexts to {out_path}")


if __name__ == "__main__":
    main()
