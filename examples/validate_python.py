#!/usr/bin/env python3
"""
Validation script for Python BookNLP implementation.
Processes test input using Python BookNLP and outputs results to JSON for comparison.
"""

import sys
import json
from typing import List, Sequence
import argparse
import spacy
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from booknlp.booknlp import BookNLP
from booknlp.common.pipelines import SpacyPipeline, Token


def serialize_tag_result(toks: List[Token], sents: Sequence, noun_chunks: Sequence):
    # Map internal Token -> SpaCyToken shape
    mapped_tokens = []
    for t in toks:
        # Normalize morph to a plain dict
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

    # Build hierarchical sentence representation expected by frontend types.
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


def process_with_python_booknlp(text: str, output_file: str):
    """Process text with Python BookNLP and save results to JSON."""

    nlp = spacy.load("en_core_web_sm")
    spacy_pipeline = SpacyPipeline(nlp)
    tokens, sentences, noun_chunks = spacy_pipeline.tag(text)

    spacy_context = serialize_tag_result(tokens, sentences, noun_chunks)

    model_params = {"pipeline": "entity,supersense,event", "model": "small"}
    booknlp = BookNLP("en", model_params)

    result = booknlp.process(text=text, doc_id="test")

    output_data = {
        "input_text": text,
        "spacy_context": spacy_context,
        "tokens": [
            {
                "text": token.text,
                "pos": token.pos,
                "ner": token.ner,
                "event": token.event,
                "tokenId": token.token_id,
                "sentenceId": token.sentence_id,
            }
            for token in result.tokens
        ],
        "entities": result.entities,
        "supersense": result.supersense,
        "timing": getattr(result, "timing", None),
        "_debug": getattr(result, "_debug", None),
    }

    # Record validation messages in the debug output instead of printing
    try:
        if output_data.get("_debug") is None:
            output_data["_debug"] = {}
        output_data["_debug"].setdefault("validation_messages", []).append(
            f"Python BookNLP results saved to {output_file}"
        )
    except Exception:
        pass

    # Write the output JSON (including debug/validation messages)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Python BookNLP on sample text."
    )
    default_input = Path(__file__).parent / "158_emma_cut.txt"
    parser.add_argument(
        "-i",
        "--input",
        default=str(default_input),
        help="Input text file path",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/python_output.json",
        help="Output JSON path",
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    with open(input_file, "r", encoding="utf-8") as f:
        test_text = f.read()

    output_file = args.output
    # Ensure output directory exists
    out_dir = Path(output_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    process_with_python_booknlp(test_text.strip(), output_file)
