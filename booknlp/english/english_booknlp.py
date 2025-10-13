from __future__ import annotations

import sys
import spacy
import copy
from dataclasses import dataclass, asdict
from booknlp.common.pipelines import SpacyPipeline
from booknlp.common.logger import get_logger
from booknlp.english.entity_tagger import LitBankEntityTagger
from booknlp.english.gender_inference_model_1 import GenderEM
from booknlp.english.name_coref import NameCoref
from booknlp.english.litbank_coref import LitBankCoref
from booknlp.english.litbank_quote import QuoteTagger
from booknlp.english.bert_qa import QuotationAttribution
from os.path import join
import os
import json
from collections import Counter
from html import escape
import time
from pathlib import Path
import urllib.request
import pkg_resources
import torch
from booknlp.common.pipelines import Token
from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Any,
    Sequence,
)
from booknlp.common.core import BookNLPResult

# Type aliases for clarity
Entity = Tuple[int, int, str, str]  # start_token, end_token, category, phrase
QuoteSpan = Tuple[int, int]


@dataclass
class EnglishBookNLPConfig:
    """Configuration for EnglishBookNLP.

    This mirrors the former loose model_params dict. New fields should have
    explicit types and sensible defaults. Optional paths may be None when
    not required for the chosen model/pipeline.
    """

    # core required params
    model: str = "small"  # one of: "small", "big", "custom"
    pipeline: str = "entity,coref,quote"  # comma-separated steps

    # spacy
    spacy_model: str = "en_core_web_sm"

    # model storage
    model_path: str | None = None

    # custom model paths (used when model == "custom")
    entity_model_path: str | None = None
    coref_model_path: str | None = None
    quote_attribution_model_path: str | None = None

    # gender / referential config
    referential_gender_cats: List[List[str]] | None = None
    referential_gender_hyperparameterFile: str | None = None
    pronominalCorefOnly: bool = True

    # runtime
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EnglishBookNLPConfig":
        # Allow unknown keys (they will be ignored) and override defaults
        field_names = {
            f.name for f in EnglishBookNLPConfig.__dataclass_fields__.values()
        }  # type: ignore
        init_args = {}
        for k, v in d.items():
            if k in field_names:
                init_args[k] = v
        return EnglishBookNLPConfig(**init_args)


class EnglishBookNLP:
    def __init__(self, model_params: Dict[str, Any] | EnglishBookNLPConfig):
        """Initialize EnglishBookNLP.

        Args:
            model_params: Either a legacy dict of parameters or an
                EnglishBookNLPConfig instance. Dicts are converted to the
                strongly typed config for internal use.

        Example:
            # Preferred strongly-typed usage
            config = EnglishBookNLPConfig(
                model="small",
                pipeline="entity,coref,quote",
                verbose=True,
            )
            nlp = EnglishBookNLP(config)

            # Backwards-compatible legacy dict
            nlp = EnglishBookNLP({
                "model": "small",
                "pipeline": "entity,coref,quote",
                "verbose": True,
            })
        """
        if isinstance(model_params, EnglishBookNLPConfig):
            self.config = model_params
        else:
            self.config = EnglishBookNLPConfig.from_dict(model_params)

        # Backwards compatibility: retain original dict for any external
        # code that may introspect (minimal risk). New code should use
        # self.config.
        model_params = self.config.to_dict()
        with torch.no_grad():
            start_time = time.time()
            self.logger = get_logger(enabled=self.config.verbose)
            self.logger.info(model_params)

            spacy_model = self.config.spacy_model

            spacy_nlp = spacy.load(spacy_model, disable=["ner"])

            valid_keys = set("entity,event,supersense,quote,coref".split(","))

            pipes = self.config.pipeline.split(",")

            self.gender_cats = [
                ["he", "him", "his"],
                ["she", "her"],
                ["they", "them", "their"],
                ["xe", "xem", "xyr", "xir"],
                ["ze", "zem", "zir", "hir"],
            ]

            if self.config.referential_gender_cats is not None:
                self.gender_cats = self.config.referential_gender_cats

            home = str(Path.home())
            modelPath = self.config.model_path or os.path.join(home, "booknlp_models")

            if not Path(modelPath).is_dir():
                Path(modelPath).mkdir(parents=True, exist_ok=True)

            if self.config.model == "big":
                entityName = "entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model"
                corefName = "coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model"
                quoteAttribName = (
                    "speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model"
                )

                self.entityPath = os.path.join(modelPath, entityName)
                if not Path(self.entityPath).is_file():
                    self.logger.info("downloading %s" % entityName)
                    urllib.request.urlretrieve(
                        "http://people.ischool.berkeley.edu/~dbamman/booknlp_models/%s"
                        % entityName,
                        self.entityPath,
                    )

                self.coref_model = os.path.join(modelPath, corefName)
                if not Path(self.coref_model).is_file():
                    self.logger.info("downloading %s" % corefName)
                    urllib.request.urlretrieve(
                        "http://people.ischool.berkeley.edu/~dbamman/booknlp_models/%s"
                        % corefName,
                        self.coref_model,
                    )

                self.quoteAttribModel = os.path.join(modelPath, quoteAttribName)
                if not Path(self.quoteAttribModel).is_file():
                    self.logger.info("downloading %s" % quoteAttribName)
                    urllib.request.urlretrieve(
                        "http://people.ischool.berkeley.edu/~dbamman/booknlp_models/%s"
                        % quoteAttribName,
                        self.quoteAttribModel,
                    )

            elif self.config.model == "small":
                entityName = "entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model"
                corefName = "coref_google_bert_uncased_L-2_H-256_A-4-v1.0.model"
                quoteAttribName = (
                    "speaker_google_bert_uncased_L-8_H-256_A-4-v1.0.1.model"
                )

                self.entityPath = os.path.join(modelPath, entityName)
                if not Path(self.entityPath).is_file():
                    self.logger.info("downloading %s" % entityName)
                    urllib.request.urlretrieve(
                        "http://people.ischool.berkeley.edu/~dbamman/booknlp_models/%s"
                        % entityName,
                        self.entityPath,
                    )

                self.coref_model = os.path.join(modelPath, corefName)
                if not Path(self.coref_model).is_file():
                    self.logger.info("downloading %s" % corefName)
                    urllib.request.urlretrieve(
                        "http://people.ischool.berkeley.edu/~dbamman/booknlp_models/%s"
                        % corefName,
                        self.coref_model,
                    )

                self.quoteAttribModel = os.path.join(modelPath, quoteAttribName)
                if not Path(self.quoteAttribModel).is_file():
                    self.logger.info("downloading %s" % quoteAttribName)
                    urllib.request.urlretrieve(
                        "http://people.ischool.berkeley.edu/~dbamman/booknlp_models/%s"
                        % quoteAttribName,
                        self.quoteAttribModel,
                    )

            elif self.config.model == "custom":
                assert (
                    self.config.entity_model_path
                    and self.config.coref_model_path
                    and self.config.quote_attribution_model_path
                ), "Custom model requires all custom model path fields to be set"
                self.entityPath = self.config.entity_model_path
                self.coref_model = self.config.coref_model_path
                self.quoteAttribModel = self.config.quote_attribution_model_path

            self.doEntities = self.doCoref = self.doQuoteAttrib = self.doSS = (
                self.doEvent
            ) = False

            for pipe in pipes:
                if pipe not in valid_keys:
                    self.logger.info("unknown pipe: %s" % pipe)
                    sys.exit(1)
                if pipe == "entity":
                    self.doEntities = True
                elif pipe == "event":
                    self.doEvent = True
                elif pipe == "coref":
                    self.doCoref = True
                elif pipe == "supersense":
                    self.doSS = True
                elif pipe == "quote":
                    self.doQuoteAttrib = True

            tagsetPath = "data/entity_cat.tagset"
            tagsetPath = pkg_resources.resource_filename(__name__, tagsetPath)

            if self.config.referential_gender_hyperparameterFile is not None:
                self.gender_hyperparameterFile = (
                    self.config.referential_gender_hyperparameterFile
                )
            else:
                self.gender_hyperparameterFile = pkg_resources.resource_filename(
                    __name__, "data/gutenberg_prop_gender_terms.txt"
                )

            pronominalCorefOnly = self.config.pronominalCorefOnly

            if not self.doEntities and self.doCoref:
                self.logger.info("coref requires entity tagging")
                sys.exit(1)

            if not self.doQuoteAttrib and self.doCoref:
                self.logger.info("coref requires quotation attribution")
                sys.exit(1)
            if not self.doEntities and self.doQuoteAttrib:
                self.logger.info("quotation attribution requires entity tagging")
                sys.exit(1)

            if self.doQuoteAttrib or self.doCoref:
                self.quoteTagger = QuoteTagger()

            if self.doEntities:
                self.entityTagger = LitBankEntityTagger(self.entityPath, tagsetPath)
                aliasPath = pkg_resources.resource_filename(
                    __name__, "data/aliases.txt"
                )
                self.name_resolver = NameCoref(aliasPath)

            if self.doQuoteAttrib:
                self.quote_attrib = QuotationAttribution(self.quoteAttribModel)

            if self.doCoref:
                self.litbank_coref = LitBankCoref(
                    self.coref_model,
                    self.gender_cats,
                    pronominalCorefOnly=pronominalCorefOnly,
                )

            self.tagger = SpacyPipeline(spacy_nlp)

            self.logger.info(
                "--- startup: %.3f seconds ---" % (time.time() - start_time)
            )

    def get_syntax(
        self,
        tokens: Sequence[Token],
        entities: List[Entity],
        assignments: Sequence[int],
        genders: Dict[int, Optional[str]],
    ) -> Dict[str, Any]:
        def check_conj(tok: Token, tokens: Sequence[Token]) -> Token:
            if tok.deprel == "conj" and tok.dephead != tok.token_id:
                return tokens[tok.dephead]
            return tok

        def get_head_in_range(
            start: int, end: int, tokens: Sequence[Token]
        ) -> Optional[Token]:
            for i in range(start, end + 1):
                if tokens[i].dephead < start or tokens[i].dephead > end:
                    return tokens[i]
            return None

        agents: Dict[int, List[Dict[str, Any]]] = {}
        patients: Dict[int, List[Dict[str, Any]]] = {}
        poss: Dict[int, List[Dict[str, Any]]] = {}
        mods: Dict[int, List[Dict[str, Any]]] = {}
        prop_mentions: Dict[int, Counter[str]] = {}
        pron_mentions: Dict[int, Counter[str]] = {}
        nom_mentions: Dict[int, Counter[str]] = {}
        keys: Counter[int] = Counter()

        toks_by_children: Dict[int, Dict[Token, int]] = {}
        for tok in tokens:
            if tok.dephead not in toks_by_children:
                toks_by_children[tok.dephead] = {}
            toks_by_children[tok.dephead][tok] = 1

        for idx, (start_token, end_token, cat, phrase) in enumerate(entities):
            ner_prop = cat.split("_")[0]
            ner_type = cat.split("_")[1]

            if ner_type != "PER":
                continue

            coref = assignments[idx]

            keys[coref] += 1
            if coref not in agents:
                agents[coref] = []
                patients[coref] = []
                poss[coref] = []
                mods[coref] = []
                prop_mentions[coref] = Counter()
                pron_mentions[coref] = Counter()
                nom_mentions[coref] = Counter()

            if ner_prop == "PROP":
                prop_mentions[coref][phrase] += 1
            elif ner_prop == "PRON":
                pron_mentions[coref][phrase] += 1
            elif ner_prop == "NOM":
                nom_mentions[coref][phrase] += 1

            tok = get_head_in_range(start_token, end_token, tokens)
            if tok is not None:
                tok = check_conj(tok, tokens)
                head = tokens[tok.dephead]

                # nsubj
                # mod
                if tok.deprel == "nsubj" and head.lemma == "be":
                    for sibling in toks_by_children[head.token_id]:
                        # "he was strong and happy", where happy -> conj -> strong -> attr/acomp -> be
                        sibling_id = sibling.token_id
                        sibling_tok = tokens[sibling_id]
                        if (
                            sibling_tok.deprel == "attr"
                            or sibling_tok.deprel == "acomp"
                        ) and (sibling_tok.pos == "NOUN" or sibling_tok.pos == "ADJ"):
                            mods[coref].append(
                                {"w": sibling_tok.text, "i": sibling_tok.token_id}
                            )

                            if sibling.token_id in toks_by_children:
                                for grandsibling in toks_by_children[sibling.token_id]:
                                    grandsibling_id = grandsibling.token_id
                                    grandsibling_tok = tokens[grandsibling_id]

                                    if grandsibling_tok.deprel == "conj" and (
                                        grandsibling_tok.pos == "NOUN"
                                        or grandsibling_tok.pos == "ADJ"
                                    ):
                                        mods[coref].append(
                                            {
                                                "w": grandsibling_tok.text,
                                                "i": grandsibling_tok.token_id,
                                            }
                                        )

                # ("Bill and Ted ran" conj captured by check_conj above)
                elif tok.deprel == "nsubj" and head.pos == ("VERB"):
                    agents[coref].append({"w": head.text, "i": head.token_id})

                    # "Bill ducked and ran", where ran -> conj -> ducked
                    for sibling in toks_by_children[head.token_id]:
                        sibling_id = sibling.token_id
                        sibling_tok = tokens[sibling_id]
                        if sibling_tok.deprel == "conj" and sibling_tok.pos == "VERB":
                            agents[coref].append(
                                {"w": sibling_tok.text, "i": sibling_tok.token_id}
                            )

                # "Jack was hit by John and William" conj captured by check_conj above
                elif tok.deprel == "pobj" and head.deprel == "agent":
                    # not root
                    if head.dephead != head.token_id:
                        grandparent = tokens[head.dephead]
                        if grandparent.pos.startswith("V"):
                            agents[coref].append(
                                {"w": grandparent.text, "i": grandparent.token_id}
                            )

                # patient ("He loved Bill and Ted" conj captured by check_conj above)
                elif (
                    tok.deprel == "dobj" or tok.deprel == "nsubjpass"
                ) and head.pos == "VERB":
                    patients[coref].append({"w": head.text, "i": head.token_id})

                # poss

                elif tok.deprel == "poss":
                    poss[coref].append({"w": head.text, "i": head.token_id})

                    # "her house and car", where car -> conj -> house
                    for sibling in toks_by_children[head.token_id]:
                        sibling_id = sibling.token_id
                        sibling_tok = tokens[sibling_id]
                        if sibling_tok.deprel == "conj":
                            poss[coref].append(
                                {"w": sibling_tok.text, "i": sibling_tok.token_id}
                            )

        data: Dict[str, Any] = {}
        data["characters"] = []

        for coref, total_count in keys.most_common():
            # must observe a character at least *twice*

            if total_count > 1:
                chardata: Dict[str, Any] = {}
                chardata["agent"] = agents[coref]
                chardata["patient"] = patients[coref]
                chardata["mod"] = mods[coref]
                chardata["poss"] = poss[coref]
                chardata["id"] = coref
                chardata["g"] = genders.get(coref)
                chardata["count"] = total_count

                mentions: Dict[str, Any] = {}

                pnames: List[Dict[str, Any]] = []
                for k, v in prop_mentions[coref].most_common():
                    pnames.append({"c": v, "n": k})
                mentions["proper"] = pnames

                nnames: List[Dict[str, Any]] = []
                for k, v in nom_mentions[coref].most_common():
                    nnames.append({"c": v, "n": k})
                mentions["common"] = nnames

                prnames: List[Dict[str, Any]] = []
                for k, v in pron_mentions[coref].most_common():
                    prnames.append({"c": v, "n": k})
                mentions["pronoun"] = prnames

                chardata["mentions"] = mentions

                data["characters"].append(chardata)

        return data

    def process(
        self,
        filename: Optional[str] = None,
        text: Optional[str] = None,
        out_folder: Optional[str] = None,
        doc_id: str = "doc",
    ) -> BookNLPResult:
        """Run the pipeline on either a filename or raw text.

        Exactly one of filename or text must be provided. If out_folder is
        supplied, all side-effect files (tokens, entities, quotes, etc.)
        will be written there using doc_id as the prefix. The structured
        result is always returned.

        Args:
            filename: Path to an input text file (mutually exclusive with text).
            text: Raw text content to process (mutually exclusive with filename).
            out_folder: Optional directory to write output artifact files.
            doc_id: Identifier used as filename prefix when writing outputs.

        Returns:
            Dictionary containing extracted data. Keys may include tokens,
            entities, quotes, attributed_quotes, coref, characters, timing.
        """
        with torch.no_grad():
            # Validate input sources
            if (filename is None and text is None) or (filename and text):
                raise ValueError("Provide exactly one of filename or text")

            if filename:
                with open(filename) as file:
                    data = file.read()
            else:
                data = text or ""

            if len(data) == 0:
                self.logger.info(
                    "Input is empty" + (f": {filename}" if filename else "")
                )
                return BookNLPResult(
                    tokens=[],
                    sents=[],
                    noun_chunks=[],
                    entities=[],
                    supersense=[],
                    quotes=[],
                    attributed_quotes=[],
                    coref=[],
                    characters=[],
                    timing={},
                )

            # Timer setup
            if self.config.verbose:
                start_time = time.time()
                originalTime = start_time
            else:
                start_time = None  # type: ignore
                originalTime = None  # type: ignore

            # Prepare output directory if requested
            if out_folder is not None:
                try:
                    os.makedirs(out_folder, exist_ok=True)
                except Exception as e:
                    self.logger.info(f"Could not create out_folder {out_folder}: {e}")

            tokens, sents, noun_chunks = self.tagger.tag(data)
            # Initialize optional outputs to avoid NameError when features disabled
            entity_vals: Dict[str, Any] = {"entities": []}
            quotes: List[QuoteSpan] = []
            attributed_quotations: List[Optional[int]] = []
            chardata: Dict[str, Any] | None = None
            genders: Dict[int, Optional[str]] | None = None

            if self.config.verbose:
                self.logger.info(
                    "--- spacy: %.3f seconds ---" % (time.time() - start_time)
                )
                start_time = time.time()

            if self.doEvent or self.doEntities or self.doSS:
                entity_vals = self.entityTagger.tag(
                    tokens,
                    doEvent=self.doEvent,
                    doEntities=self.doEntities,
                    doSS=self.doSS,
                )
                entity_vals["entities"] = sorted(entity_vals["entities"])
                if self.doSS and out_folder is not None:
                    supersense_entities = entity_vals["supersense"]
                    with open(
                        join(out_folder, f"{doc_id}.supersense"),
                        "w",
                        encoding="utf-8",
                    ) as out:
                        out.write("start_token\tend_token\tsupersense_category\ttext\n")
                        for start, end, cat, text in supersense_entities:
                            out.write("%s\t%s\t%s\t%s\n" % (start, end, cat, text))

                if self.doEvent:
                    events = entity_vals["events"]
                    for token in tokens:
                        if token.token_id in events:
                            token.event = True

                if out_folder is not None:
                    with open(
                        join(out_folder, f"{doc_id}.tokens"), "w", encoding="utf-8"
                    ) as out:
                        out.write(
                            "%s\n"
                            % "\t".join(
                                [
                                    "paragraph_ID",
                                    "sentence_ID",
                                    "token_ID_within_sentence",
                                    "token_ID_within_document",
                                    "word",
                                    "lemma",
                                    "byte_onset",
                                    "byte_offset",
                                    "POS_tag",
                                    "fine_POS_tag",
                                    "dependency_relation",
                                    "syntactic_head_ID",
                                    "event",
                                ]
                            )
                        )
                        for token in tokens:
                            out.write("%s\n" % token)

                if self.config.verbose:
                    self.logger.info(
                        "--- entities: %.3f seconds ---" % (time.time() - start_time)
                    )
                    start_time = time.time()

            if self.doQuoteAttrib or self.doCoref:
                in_quotes = []
                quotes = self.quoteTagger.tag(tokens)

                if self.config.verbose:
                    self.logger.info(
                        "--- quotes: %.3f seconds ---" % (time.time() - start_time)
                    )
                    start_time = time.time()

            if self.doQuoteAttrib:
                entities = entity_vals["entities"]
                attributed_quotations = self.quote_attrib.tag(quotes, entities, tokens)

                if self.config.verbose:
                    self.logger.info(
                        "--- attribution: %.3f seconds ---" % (time.time() - start_time)
                    )
                    start_time = time.time()

            if self.doEntities:
                entities = entity_vals["entities"]

                in_quotes = []

                for start, end, cat, text in entities:
                    if tokens[start].inQuote or tokens[end].inQuote:
                        in_quotes.append(1)
                    else:
                        in_quotes.append(0)

                # Create entity for first-person narrator, if present
                refs = self.name_resolver.cluster_narrator(entities, in_quotes, tokens)

                # Cluster non-PER PROP mentions that are identical
                refs = self.name_resolver.cluster_identical_propers(entities, refs)

                # Cluster mentions of named people
                refs = self.name_resolver.cluster_only_nouns(entities, refs, tokens)

                if self.config.verbose:
                    self.logger.info(
                        "--- name coref: %.3f seconds ---" % (time.time() - start_time)
                    )

                if self.doCoref:
                    # Infer referential gender from he/she/they mentions around characters
                    start_time = time.time()
                    genderEM = GenderEM(
                        tokens=tokens,
                        entities=entities,
                        refs=refs,
                        genders=self.gender_cats,
                        hyperparameterFile=self.gender_hyperparameterFile,
                    )
                    genders = genderEM.tag(entities, tokens, refs)
                    if self.config.verbose:
                        self.logger.info(
                            "--- gender: %.3f seconds ---" % (time.time() - start_time)
                        )

            assignments = None
            if self.doEntities:
                assignments = copy.deepcopy(refs)

            if self.doCoref:
                start_time = time.time()
                torch.cuda.empty_cache()
                assignments = self.litbank_coref.tag(
                    tokens, entities, refs, genders, attributed_quotations, quotes
                )

                if self.config.verbose:
                    self.logger.info(
                        "--- coref: %.3f seconds ---" % (time.time() - start_time)
                    )
                    start_time = time.time()

                ent_names = {}
                for a, e in zip(assignments, entities):
                    if a not in ent_names:
                        ent_names[a] = Counter()
                    ent_names[a][e[3]] += 1

                # Update gender estimates from coref data
                genders = genderEM.update_gender_from_coref(
                    genders, entities, assignments
                )

                chardata = self.get_syntax(tokens, entities, assignments, genders)
                if out_folder is not None:
                    with open(
                        join(out_folder, f"{doc_id}.book"), "w", encoding="utf-8"
                    ) as out:
                        json.dump(chardata, out)

            start_time = time.time()
            if self.doEntities:
                for i, (start, end, cat, text) in enumerate(entity_vals["entities"]):
                    coref = assignments[i] if assignments is not None else -1
                    ent_type = cat.split("_")[1]
                    ent_prop = cat.split("_")[0]
                    entity_vals["entities"][i] = {
                        "start_token": start,
                        "end_token": end,
                        "cat": ent_type,
                        "text": text,
                        "coref": coref,
                        "prop": ent_prop,
                    }
            if self.doEntities and out_folder is not None:
                # Write entities and coref
                with open(
                    join(out_folder, f"{doc_id}.entities"), "w", encoding="utf-8"
                ) as out:
                    out.write("COREF\tstart_token\tend_token\tprop\tcat\ttext\n")
                    for ent in entity_vals["entities"]:
                        out.write(
                            "%s\t%s\t%s\t%s\t%s\t%s\n"
                            % (
                                ent["coref"],
                                ent["start_token"],
                                ent["end_token"],
                                ent["prop"],
                                ent["cat"],
                                ent["text"],
                            )
                        )
                if self.config.verbose:
                    self.logger.info(
                        "--- entity write: %.3f seconds ---"
                        % (time.time() - start_time)
                    )

            if self.doQuoteAttrib and out_folder is not None:
                with open(
                    join(out_folder, f"{doc_id}.quotes"), "w", encoding="utf-8"
                ) as out:
                    out.write(
                        "\t".join(
                            [
                                "quote_start",
                                "quote_end",
                                "mention_start",
                                "mention_end",
                                "mention_phrase",
                                "char_id",
                                "quote",
                            ]
                        )
                        + "\n"
                    )

                    for idx, line in enumerate(attributed_quotations):
                        q_start, q_end = quotes[idx]
                        mention = attributed_quotations[idx]
                        if mention is not None:
                            entity = entities[mention]
                            speaker_id = assignments[mention]
                            e_start = entity[0]
                            e_end = entity[1]
                            cat = entity[3]
                            speak = speaker_id
                        else:
                            e_start = None
                            e_end = None
                            cat = None
                            speak = None
                        quote = [tok.text for tok in tokens[q_start : q_end + 1]]
                        out.write(
                            "%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                            % (
                                q_start,
                                q_end,
                                e_start,
                                e_end,
                                cat,
                                speak,
                                " ".join(quote),
                            )
                        )

            if self.doQuoteAttrib and self.doCoref and out_folder is not None:
                # get canonical name for character
                names = {}
                for idx, (start, end, cat, text) in enumerate(entities):
                    coref = assignments[idx]
                    if coref not in names:
                        names[coref] = Counter()
                    ner_prop = cat.split("_")[0]
                    ner_type = cat.split("_")[1]
                    if ner_prop == "PROP":
                        names[coref][text.lower()] += 10
                    elif ner_prop == "NOM":
                        names[coref][text.lower()] += 1
                    else:
                        names[coref][text.lower()] += 0.001

                with open(
                    join(out_folder, f"{doc_id}.book.html"), "w", encoding="utf-8"
                ) as out:
                    out.write("<html>")
                    out.write("""<head>
          <meta charset="UTF-8">
        </head>""")
                    out.write("<h2>Named characters</h2>\n")
                    for character in chardata["characters"]:
                        char_id = character["id"]

                        proper_names = character["mentions"]["proper"]
                        if len(proper_names) > 0 or char_id == 0:  # 0=narrator
                            proper_name_list = "/".join(
                                [
                                    "%s (%s)" % (name["n"], name["c"])
                                    for name in proper_names
                                ]
                            )

                            common_names = character["mentions"]["common"]
                            common_name_list = "/".join(
                                [
                                    "%s (%s)" % (name["n"], name["c"])
                                    for name in common_names
                                ]
                            )

                            char_count = character["count"]

                            if char_id == 0:
                                if len(proper_name_list) == 0:
                                    proper_name_list = "[NARRATOR]"
                                else:
                                    proper_name_list += "/[NARRATOR]"
                            out.write(
                                "%s %s %s <br />\n"
                                % (char_count, proper_name_list, common_name_list)
                            )

                    out.write("<p>\n")

                    out.write("<h2>Major entities (proper, common)</h2>")

                    major_places = {}
                    for prop in ["PROP", "NOM"]:
                        major_places[prop] = {}
                        for cat in ["FAC", "GPE", "LOC", "PER", "ORG", "VEH"]:
                            major_places[prop][cat] = {}

                    for idx, (start, end, cat, text) in enumerate(entities):
                        coref = assignments[idx]

                        ner_prop = cat.split("_")[0]
                        ner_type = cat.split("_")[1]
                        if ner_prop != "PRON":
                            if coref not in major_places[ner_prop][ner_type]:
                                major_places[ner_prop][ner_type][coref] = Counter()
                            major_places[ner_prop][ner_type][coref][text] += 1

                    max_entities_to_display = 10
                    for cat in ["FAC", "GPE", "LOC", "PER", "ORG", "VEH"]:
                        out.write("<h3>%s</h3>" % cat)
                        for prop in ["PROP", "NOM"]:
                            freqs = {}
                            for coref in major_places[prop][cat]:
                                freqs[coref] = sum(
                                    major_places[prop][cat][coref].values()
                                )

                            sorted_freqs = sorted(
                                freqs.items(), key=lambda x: x[1], reverse=True
                            )
                            for k, v in sorted_freqs[:max_entities_to_display]:
                                ent_names = []
                                for name, count in major_places[prop][cat][
                                    k
                                ].most_common():
                                    ent_names.append("%s" % (name))
                                out.write("%s %s <br />" % (v, "/".join(ent_names)))
                            out.write("<p>")

                    out.write("<h2>Text</h2>\n")

                    beforeToks = [""] * len(tokens)
                    afterToks = [""] * len(tokens)

                    lastP = None

                    for idx, (start, end, cat, text) in enumerate(entities):
                        coref = assignments[idx]
                        name = names[coref].most_common(1)[0][0]
                        beforeToks[start] += '<font color="#D0D0D0">[</font>'
                        afterToks[end] = (
                            '<font color="#D0D0D0">]</font><font color="#FF00FF"><sub>%s-%s</sub></font>'
                            % (coref, name)
                            + afterToks[end]
                        )

                    for idx, (start, end) in enumerate(quotes):
                        mention_id = attributed_quotations[idx]
                        if mention_id is not None:
                            speaker_id = assignments[mention_id]
                            name = names[speaker_id].most_common(1)[0][0]
                        else:
                            speaker_id = "None"
                            name = "None"
                        beforeToks[start] += '<font color="#666699">'
                        afterToks[end] += "</font><sub>[%s-%s]</sub>" % (
                            speaker_id,
                            name,
                        )

                    for idx in range(len(tokens)):
                        if tokens[idx].paragraph_id != lastP:
                            out.write("<p />")
                        out.write(
                            "%s%s%s "
                            % (
                                beforeToks[idx],
                                escape(tokens[idx].text),
                                afterToks[idx],
                            )
                        )
                        lastP = tokens[idx].paragraph_id

                    out.write("</html>")

            elapsed = None
            if self.config.verbose:
                elapsed = time.time() - originalTime
                self.logger.info(
                    "--- TOTAL (excl. startup): %.3f seconds ---, %s words"
                    % (elapsed, len(tokens))
                )

            return BookNLPResult(
                tokens=list(tokens),
                sents=list(sents),
                noun_chunks=noun_chunks,
                entities=entity_vals["entities"] if self.doEntities else [],
                supersense=entity_vals["supersense"] if self.doSS else [],
                quotes=quotes if (self.doQuoteAttrib or self.doCoref) else [],
                attributed_quotes=attributed_quotations if self.doQuoteAttrib else [],
                coref=assignments if self.doCoref else [],
                characters=chardata["characters"]
                if (chardata and self.doCoref)
                else [],
                timing={"elapsed": elapsed} if elapsed is not None else {},
            )
