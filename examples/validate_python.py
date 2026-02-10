#!/usr/bin/env python3
"""
Validation script for Python BookNLP implementation.
Processes test input using Python BookNLP and outputs results to JSON for comparison.
"""

import sys
import json
import spacy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from booknlp.booknlp import BookNLP


def process_with_python_booknlp(text: str, output_file: str):
    """Process text with Python BookNLP and save results to JSON."""

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    spacy_context = {"tokens": [], "sentences": []}

    for token in doc:
        token_dict = {
            "text": token.text,
            "startByte": token.idx,
            "endByte": token.idx + len(token.text),
            "pos": token.pos_,
            "finePos": token.tag_,
            "lemma": token.lemma_,
            "deprel": token.dep_,
            "dephead": token.head.i,
            "morph": {str(k): str(v) for k, v in token.morph.to_dict().items()},
            "likeNum": token.like_num,
            "isStop": token.is_stop,
            "sentenceId": token.sent.start,
            "withinSentenceId": token.i - token.sent.start,
        }
        spacy_context["tokens"].append(token_dict)

    for sent in doc.sents:
        spacy_context["sentences"].append({"start": sent.start, "end": sent.end})

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
        "timing": result.timing,
        "_debug": result._debug,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Python BookNLP results saved to {output_file}")
    return output_data


if __name__ == "__main__":
    input_file = Path(__file__).parent / "158_emma.txt"

    with open(input_file, "r", encoding="utf-8") as f:
        test_text = f.read()

    output_file = "python_output.json"

    process_with_python_booknlp(test_text.strip(), output_file)
    print("\n✓ Python validation complete!")
