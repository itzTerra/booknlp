from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from booknlp.common.pipelines import Token


@dataclass
class BookNLPResult:
    tokens: List[Token]
    sents: List[Any]
    noun_chunks: List[Any]
    entities: List[Dict[str, Any]]
    supersense: List[Any]
    quotes: List[Any]
    attributed_quotes: List[Optional[int]]
    coref: Optional[List[int]]
    characters: List[Dict[str, Any]]
    timing: Dict[str, Any]
