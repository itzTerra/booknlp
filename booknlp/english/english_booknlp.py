from __future__ import annotations

import sys
import spacy
from dataclasses import dataclass, asdict
from booknlp.common.pipelines import SpacyPipeline
from booknlp.common.logger import get_logger
from booknlp.english.entity_tagger import LitBankEntityTagger
from booknlp.english.name_coref import NameCoref
from os.path import join
import os
import time
from pathlib import Path
import urllib.request
import pkg_resources
import torch
from typing import (
    Dict,
    Optional,
    Any,
)
from booknlp.common.core import BookNLPResult


@dataclass
class EnglishBookNLPConfig:
    """Configuration for EnglishBookNLP.

    This mirrors the former loose model_params dict. New fields should have
    explicit types and sensible defaults. Optional paths may be None when
    not required for the chosen model/pipeline.
    """

    # core required params
    model: str = "small"  # one of: "small", "big", "custom"
    pipeline: str = "entity"  # comma-separated steps

    # spacy
    spacy_model: str = "en_core_web_sm"

    # model storage
    model_path: str | None = None

    # custom model paths (used when model == "custom")
    entity_model_path: str | None = None

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
                pipeline="entity",
                verbose=True,
            )
            nlp = EnglishBookNLP(config)

            # Backwards-compatible legacy dict
            nlp = EnglishBookNLP({
                "model": "small",
                "pipeline": "entity",
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

            valid_keys = set("entity,event,supersense".split(","))

            pipes = self.config.pipeline.split(",")

            self.doEntities = self.doSS = self.doEvent = False

            for pipe in pipes:
                if pipe not in valid_keys:
                    self.logger.info("unknown pipe: %s" % pipe)
                    sys.exit(1)
                if pipe == "entity":
                    self.doEntities = True
                elif pipe == "event":
                    self.doEvent = True
                elif pipe == "supersense":
                    self.doSS = True

            home = str(Path.home())
            modelPath = self.config.model_path or os.path.join(home, "booknlp_models")

            if not Path(modelPath).is_dir():
                Path(modelPath).mkdir(parents=True, exist_ok=True)

            if self.config.model == "big":
                entityName = "entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model"

                self.entityPath = os.path.join(modelPath, entityName)
                if not Path(self.entityPath).is_file():
                    self.logger.info("downloading %s" % entityName)
                    urllib.request.urlretrieve(
                        "http://people.ischool.berkeley.edu/~dbamman/booknlp_models/%s"
                        % entityName,
                        self.entityPath,
                    )

            elif self.config.model == "small":
                entityName = "entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model"

                if self.doEntities:
                    self.entityPath = os.path.join(modelPath, entityName)
                    if not Path(self.entityPath).is_file():
                        self.logger.info("downloading %s" % entityName)
                        urllib.request.urlretrieve(
                            "http://people.ischool.berkeley.edu/~dbamman/booknlp_models/%s"
                            % entityName,
                            self.entityPath,
                        )

            elif self.config.model == "custom":
                assert self.config.entity_model_path, (
                    "Custom model requires entity_model_path to be set"
                )
                self.entityPath = self.config.entity_model_path

            tagsetPath = "data/entity_cat.tagset"
            tagsetPath = pkg_resources.resource_filename(__name__, tagsetPath)

            if self.doEntities:
                self.entityTagger = LitBankEntityTagger(self.entityPath, tagsetPath)
                aliasPath = pkg_resources.resource_filename(
                    __name__, "data/aliases.txt"
                )
                self.name_resolver = NameCoref(aliasPath)

            self.tagger = SpacyPipeline(spacy_nlp)

            self.logger.info(
                "--- startup: %.3f seconds ---" % (time.time() - start_time)
            )

    def process(
        self,
        filename: Optional[str] = None,
        text: Optional[str] = None,
        out_folder: Optional[str] = None,
        doc_id: str = "doc",
    ) -> BookNLPResult:
        """Run the pipeline on either a filename or raw text.

        Exactly one of filename or text must be provided. If out_folder is
        supplied, all side-effect files (tokens, entities, etc.)
        will be written there using doc_id as the prefix. The structured
        result is always returned.

        Args:
            filename: Path to an input text file (mutually exclusive with text).
            text: Raw text content to process (mutually exclusive with filename).
            out_folder: Optional directory to write output artifact files.
            doc_id: Identifier used as filename prefix when writing outputs.

        Returns:
            Dictionary containing extracted data. Keys may include tokens,
            entities, supersense, timing.
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

                for i, (start, end, cat, text) in enumerate(entities):
                    ent_type = cat.split("_")[1]
                    ent_prop = cat.split("_")[0]
                    entities[i] = {
                        "start_token": start,
                        "end_token": end,
                        "cat": ent_type,
                        "text": text,
                        "prop": ent_prop,
                        "coref": refs[i],
                    }
            if self.doEntities and out_folder is not None:
                # Write entities
                with open(
                    join(out_folder, f"{doc_id}.entities"), "w", encoding="utf-8"
                ) as out:
                    out.write("start_token\tend_token\tprop\tcat\ttext\n")
                    for ent in entity_vals["entities"]:
                        out.write(
                            "%s\t%s\t%s\t%s\t%s\n"
                            % (
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

            elapsed = None
            if self.config.verbose:
                elapsed = time.time() - originalTime
                self.logger.info(
                    "--- TOTAL (excl. startup): %.3f seconds ---, %s words"
                    % (elapsed, len(tokens))
                )

            result = BookNLPResult(
                tokens=list(tokens),
                sents=list(sents),
                noun_chunks=noun_chunks,
                entities=entity_vals["entities"] if self.doEntities else [],
                supersense=entity_vals["supersense"] if self.doSS else [],
                timing={"elapsed": elapsed} if elapsed is not None else {},
            )
            # debug collection removed; no logs attached to results
            return result
