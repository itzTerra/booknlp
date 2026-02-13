from dataclasses import dataclass, field
from typing import List, Dict, Any
from booknlp.common.pipelines import Token


@dataclass
class BookNLPResult:
    tokens: List[Token]
    sents: List[Any]
    noun_chunks: List[Any]
    entities: List[Dict[str, Any]]
    supersense: List[Any]
    timing: Dict[str, Any]
    # debug collection removed
