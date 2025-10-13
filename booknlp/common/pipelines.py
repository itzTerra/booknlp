from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from spacy.tokens import Doc


class Entity:
    def __init__(
        self,
        start: int,
        end: int,
        entity_id: Optional[int] = None,
        quote_id: Optional[int] = None,
        quote_eid: Optional[int] = None,
        proper: Optional[bool] = None,
        ner_cat: Optional[str] = None,
        in_quote: Optional[bool] = None,
        text: Optional[str] = None,
    ) -> None:
        """Lightweight container for an entity span.

        Parameters
        ----------
        start, end: Character offsets (token-based indices externally) marking the span.
        entity_id: Canonical entity identifier.
        quote_id: Identifier of quote the entity appears in, if any.
        quote_eid: Identifier of the quoted entity mention if applicable.
        proper: Whether this span is a proper noun mention.
        ner_cat: Named entity category label.
        in_quote: Whether the mention is inside a quotation span.
        text: Raw surface text.
        """
        self.start: int = start
        self.end: int = end
        self.entity_id: Optional[int] = entity_id
        self.quote_id: Optional[int] = quote_id
        self.proper: Optional[bool] = proper
        self.ner_cat: Optional[str] = ner_cat
        self.in_quote: Optional[bool] = in_quote
        self.quote_eid: Optional[int] = quote_eid
        self.text: Optional[str] = text
        self.quote_mention: Optional[str] = None
        self.global_start: Optional[int] = None
        self.global_end: Optional[int] = None

    def __str__(self) -> str:  # pragma: no cover - convenience representation
        return "%s %s %s %s %s %s %s %s" % (
            self.global_start,
            self.global_end,
            self.entity_id,
            self.proper,
            self.ner_cat,
            self.in_quote,
            self.quote_eid,
            self.text,
        )


@dataclass
class Token:
    paragraph_id: int
    sentence_id: int
    within_sentence_id: int
    token_id: int
    text: str
    pos: Optional[str]
    fine_pos: Optional[str]
    lemma: Optional[str]
    deprel: Optional[str]
    dephead: Optional[int]
    ner: Optional[str]
    startByte: int
    morph: Optional[Any]
    like_num: bool = False
    is_stop: bool = False

    itext: str = field(init=False)
    endByte: int = field(init=False)
    inQuote: bool = field(init=False)
    event: bool = field(init=False)

    def __post_init__(self) -> None:
        self.itext = self.text.casefold()
        self.endByte = self.startByte + len(self.text)
        self.inQuote = False
        self.event = False

    def __str__(self) -> str:
        return "\t".join(
            [
                str(x)
                for x in [
                    self.paragraph_id,
                    self.sentence_id,
                    self.within_sentence_id,
                    self.token_id,
                    self.text,
                    self.lemma,
                    self.startByte,
                    self.endByte,
                    self.pos,
                    self.fine_pos,
                    self.deprel,
                    self.dephead,
                    self.event,
                ]
            ]
        )

    @classmethod
    def convert(cls, sents: Sequence[Sequence[str]]) -> List[Token]:
        toks: List[Token] = []
        i = 0
        cur = 0
        for sidx, sent in enumerate(sents):
            for widx, word in enumerate(sent):
                token = Token(
                    0,
                    sidx,
                    widx,
                    i,
                    word,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    cur,
                    None,
                )
                toks.append(token)
                i += 1
                cur += len(word) + 1
        return toks

    @classmethod
    def deconvert(cls, toks: Sequence[Token]) -> List[List[Token]]:
        sents: List[List[Token]] = []
        sent: List[Token] = []
        lastSid: Optional[int] = None
        for tok in toks:
            if lastSid is not None and tok.sentence_id != lastSid:
                sents.append(sent)
                sent = []
            sent.append(tok)
            lastSid = tok.sentence_id

        if len(sent) > 0:
            sents.append(sent)

        return sents


class SpacyPipeline:
    def __init__(self, spacy_nlp) -> None:  # type: ignore[no-untyped-def]
        # We intentionally keep spacy_nlp untyped to avoid importing large spaCy types.
        self.spacy_nlp = spacy_nlp
        self.spacy_nlp.max_length = 10000000

    def filter_ws(self, text: str) -> str:
        text = re.sub(" ", "S", text)
        text = re.sub("[\n\r]", "N", text)
        text = re.sub("\t", "T", text)
        return text

    def tag_pretokenized(
        self, toks: Sequence[str], sents: Sequence[bool], spaces: Sequence[bool]
    ) -> List[Token]:
        doc = Doc(self.spacy_nlp.vocab, words=list(toks), spaces=list(spaces))
        for idx, token in enumerate(doc):
            token.sent_start = sents[idx]

        for _name, proc in self.spacy_nlp.pipeline:  # type: ignore[attr-defined]
            doc = proc(doc)

        return self.process_doc(doc)

    def tag(self, text: str) -> Tuple[List[Token], Sequence, Sequence]:
        doc = self.spacy_nlp(text)
        return self.process_doc(doc), doc.sents, doc.noun_chunks

    def process_doc(self, doc: Doc) -> List[Token]:
        tokens: List[Token] = []
        skipped_global = 0
        paragraph_id = 0
        current_whitespace = ""
        sentence_id = 0
        for _sid, sent in enumerate(doc.sents):
            skipped_in_sentence = 0
            skips_in_sentence: List[int] = []
            curSkips = 0
            for _w_idx, tok in enumerate(sent):
                if tok.is_space:
                    curSkips += 1
                skips_in_sentence.append(curSkips)

            hasWord = False

            for w_idx, tok in enumerate(sent):
                if tok.is_space:
                    skipped_global += 1
                    skipped_in_sentence += 1
                    current_whitespace += tok.text
                else:
                    if re.search("\n\n", current_whitespace) is not None:
                        paragraph_id += 1

                    hasWord = True

                    head_in_sentence = tok.head.i - sent.start
                    skips_between_token_and_head = (
                        skips_in_sentence[head_in_sentence] - skips_in_sentence[w_idx]
                    )
                    token = Token(
                        paragraph_id,
                        sentence_id,
                        w_idx - skipped_in_sentence,
                        tok.i - skipped_global,
                        self.filter_ws(tok.text),
                        tok.pos_,
                        tok.tag_,
                        tok.lemma_,
                        tok.dep_,
                        tok.head.i - skipped_global - skips_between_token_and_head,
                        None,
                        tok.idx,
                        tok.morph,
                        tok.like_num,
                        tok.is_stop,
                    )
                    tokens.append(token)
                    current_whitespace = ""

            if hasWord:
                sentence_id += 1

        return tokens
