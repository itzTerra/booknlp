from booknlp.english.tagger import Tagger
import torch
import re
import booknlp.common.layered_reader as layered_reader
import booknlp.common.sequence_layered_reader as sequence_layered_reader
import pkg_resources


class LitBankEntityTagger:
    def __init__(self, model_file, model_tagset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tagset = sequence_layered_reader.read_tagset(model_tagset)
        supersenseTagset = pkg_resources.resource_filename(
            __name__, "data/supersense.tagset"
        )

        self.supersense_tagset = sequence_layered_reader.read_tagset(supersenseTagset)
        base_model = re.sub("google_bert", "google/bert", model_file.split("/")[-1])
        base_model = re.sub(".model", "", base_model)

        self.model = Tagger(
            freeze_bert=False,
            base_model=base_model,
            tagset_flat={"EVENT": 1, "O": 1},
            supersense_tagset=self.supersense_tagset,
            tagset=self.tagset,
            device=device,
        )

        self.model.to(device)

        state_dict = torch.load(model_file, map_location=device)
        del state_dict["bert.embeddings.position_ids"]
        self.model.load_state_dict(state_dict)
        wnsFile = pkg_resources.resource_filename(__name__, "data/wordnet.first.sense")
        self.wns = self._read_wn(wnsFile)

    def _read_wn(self, filename):
        wns = {}
        with open(filename) as file:
            for line in file:
                cols = line.rstrip().split("\t")
                word = cols[0]
                pos = cols[1]
                wn = int(cols[2].split(" ")[0])
                wns["%s.%s" % (word, pos)] = wn
        return wns

    def _get_wn(self, supersense_batched_sents):
        wn_batches = []

        for idx, b_sent in enumerate(supersense_batched_sents):
            max_len = 0
            for sent in b_sent:
                if sent is not None:
                    if len(sent) > max_len:
                        max_len = len(sent)

            wn_senses = []

            for sent in b_sent:
                wn = []
                if sent is None:
                    continue

                for word in sent:
                    if word is None:
                        wn.append(0)
                    else:
                        text = word.text
                        pos = word.pos
                        if pos == "NOUN":
                            pos = "n"
                        elif pos == "VERB":
                            pos = "v"
                        term = text.split(" ")[-1].lower()
                        key = "%s.%s" % (term, pos)
                        if key in self.wns:
                            wn.append(self.wns[key])
                        else:
                            wn.append(1)

                for val in range(len(sent), max_len):
                    wn.append(0)
                wn_senses.append(wn)

            wn_senses = torch.LongTensor(wn_senses)
            wn_batches.append(wn_senses)
        return wn_batches

    def tag(self, toks, doEvent=True, doEntities=True, doSS=True, debug_info=None):
        if debug_info is None:
            debug_info = {}
        max_sentence_length = 500

        entities = []
        supersense_entities = []

        batch_size = 32

        sents = []
        o_sents = []
        sent = []
        o_sent = []
        lastSid = None

        length = 0
        split_count = 0

        for tok in toks:
            wptok = tok.text
            # working with uncased BERT models, so add a special tag to denote capitalization
            if wptok[0].lower() != wptok[0]:
                wptok = "[CAP] " + wptok.lower()

            toks = self.model.tokenizer.tokenize(wptok)
            if lastSid is not None and (
                tok.sentence_id != lastSid or length + len(toks) > max_sentence_length
            ):
                sents.append(sent)
                o_sents.append(o_sent)
                split_count += 1
                sent = []
                o_sent = []
                length = 0

            sent.append(toks)
            o_sent.append(tok)

            lastSid = tok.sentence_id
            length += len(toks)

        sents.append(sent)
        o_sents.append(o_sent)
        # Debug: sents should equal split_count + 1 (splits + final append)
        # print(f"DEBUG PY: Created {len(sents)} sentence chunks after {split_count} splits")
        first_phase_count = len(sents)

        # Calculate wordpiece lengths for each first-phase chunk for debugging
        chunk_wordpiece_lengths = []
        # Prepare a compact per-chunk sample (first 150 chunks) with token-level wordpiece details
        chunk_wordpiece_samples = []
        for sent in sents:
            chunk_len = 0
            for toks in sent:
                chunk_len += len(toks)
            chunk_wordpiece_lengths.append(chunk_len)

        try:
            for ci, sent in enumerate(sents[:150]):
                token_details = []
                for ti, wp_tokens in enumerate(sent):
                    try:
                        orig_tok = o_sents[ci][ti]
                        text = orig_tok.text
                        prepared = text
                        if text and text[0].lower() != text[0]:
                            prepared = "[CAP] " + text.lower()
                        wp_list = list(wp_tokens)
                        wp_ids = []
                        for wp in wp_list:
                            try:
                                wid = self.model.tokenizer.convert_tokens_to_ids(wp)
                            except Exception:
                                try:
                                    wid = self.model.tokenizer.vocab.get(wp)
                                except Exception:
                                    wid = None
                            wp_ids.append(wid)

                        wp_len = len(wp_list)
                        token_details.append(
                            {
                                "idx": ti,
                                "text": text,
                                "prepared": prepared,
                                "wp_ids": wp_ids,
                                "wp_tokens": wp_list,
                                "wp_len": wp_len,
                                "token_id": getattr(orig_tok, "token_id", None),
                            }
                        )
                    except Exception:
                        continue
                chunk_wordpiece_samples.append(
                    {"chunkIdx": ci, "tokenDetails": token_details}
                )
        except Exception:
            chunk_wordpiece_samples = []

        # Prepare a compact sample of the first few sents with per-token wordpiece info
        # This helps compare Python tokenizer outputs to the TypeScript port.
        sents_sample = []
        cap_token_id = None
        try:
            try:
                cap_token_id = self.model.tokenizer.convert_tokens_to_ids("[CAP]")
            except Exception:
                try:
                    cap_token_id = self.model.tokenizer.vocab.get("[CAP]")
                except Exception:
                    cap_token_id = None

            for si, sent_wp in enumerate(sents[:5]):
                sample = []
                # o_sents mirrors sents but holds original Token objects
                for ti, wp_tokens in enumerate(sent_wp):
                    try:
                        orig_tok = o_sents[si][ti]
                        text = orig_tok.text
                        prepared = text
                        if text and text[0].lower() != text[0]:
                            prepared = "[CAP] " + text.lower()
                        wp_list = list(wp_tokens)
                        wp_len = len(wp_list)
                        sample.append(
                            {
                                "text": text,
                                "prepared": prepared,
                                "wordpieces": wp_list,
                                "wp_len": wp_len,
                                "token_id": getattr(orig_tok, "token_id", None),
                            }
                        )
                    except Exception:
                        # best-effort: skip malformed entries
                        continue
                sents_sample.append(sample)
        except Exception:
            sents_sample = []

        sentences = []
        o_sentences = []

        sentence = [["[CLS]"]]

        o_sent = []

        cur_length = 0
        group_chunk_count = 0
        group_lengths = []
        group_chunk_counts = []

        for idx, sent in enumerate(sents):
            sent_len = 0
            for toks in sent:
                sent_len += len(toks)

            if sent_len + cur_length >= max_sentence_length:
                sentence.append(["[SEP]"])
                sentences.append(sentence)
                o_sentences.append(o_sent)
                group_lengths.append(cur_length)
                group_chunk_counts.append(group_chunk_count)
                sentence = [["[CLS]"]]
                o_sent = []
                cur_length = 0
                group_chunk_count = 0

            cur_length += sent_len
            group_chunk_count += 1

            sentence.extend(sent)
            o_sent.extend(o_sents[idx])

        if len(sentence) > 1:
            sentence.append(["[SEP]"])
            o_sentences.append(o_sent)
            sentences.append(sentence)
            group_lengths.append(cur_length)
            group_chunk_counts.append(group_chunk_count)

        sents = o_sentences

        sentence_lengths = []
        for sent in sentences:
            sent_len = 0
            for toks in sent:
                sent_len += len(toks)
            sentence_lengths.append(sent_len)

        # Debug: Track sentence grouping before batching
        sentence_groups_count = len(sentences)
        first_phase_chunks = first_phase_count
        second_phase_groups = sentence_groups_count

        (
            batched_sents,
            batched_data,
            batched_mask,
            batched_transforms,
            batched_orig_token_lens,
            ordering,
            order_to_batch_map,
        ) = layered_reader.get_batches(
            self.model, sentences, batch_size, self.tagset, training=False
        )

        # Request collection of batch-level debug details for known problematic ranges
        # so tagger can optionally attach CRF/viterbi internals and we can emit
        # per-batch tokenization/transforms for easier side-by-side comparison.
        try:
            debug_info.setdefault("collect_batch_debug_ranges", [])
        except Exception:
            pass
        # Add the ranges we care about (matches TypeScript debugRanges)
        try:
            debug_info["collect_batch_debug_ranges"].extend(
                [
                    {"start": 26960, "end": 26990},
                    {"start": 27030, "end": 27060},
                    {"start": 27180, "end": 27300},
                    {"start": 30350, "end": 30365},
                    {"start": 34805, "end": 34820},
                ]
            )
        except Exception:
            pass

        batch_pos = {}
        for idx, ind in enumerate(ordering):
            batch_id, batch_s, batch_position = order_to_batch_map[idx]
            if batch_id not in batch_pos:
                batch_pos[batch_id] = [None] * batch_s
            batch_pos[batch_id][batch_position] = [None]
            for tok in sents[ind]:
                # print(tok)
                batch_pos[batch_id][batch_position].append(tok)
            batch_pos[batch_id][batch_position].append(None)

        batched_pos = [None] * len(batch_pos)
        for i in range(len(batch_pos)):
            batched_pos[i] = batch_pos[i]

        wn_batches = self._get_wn(batched_pos)

        # Debug: Log basic WordNet batch structure for analysis
        if debug_info is not None:
            debug_info["wn_batches_count"] = len(wn_batches)
            debug_info["wn_batches_shapes"] = [
                (
                    wn_batch.shape[0] if hasattr(wn_batch, "shape") else len(wn_batch),
                    wn_batch.shape[1]
                    if hasattr(wn_batch, "shape")
                    else (len(wn_batch[0]) if len(wn_batch) > 0 else 0),
                )
                for wn_batch in wn_batches
            ]

            wn_batches_details = []
            for i, (batch_pos_item, wn_batch) in enumerate(
                zip(batched_pos, wn_batches)
            ):
                batch_token_count = 0
                for sent in batch_pos_item:
                    if sent is not None:
                        for tok in sent:
                            if tok is not None:
                                batch_token_count += 1

                wn_senses_length = (
                    wn_batch.shape[1]
                    if hasattr(wn_batch, "shape") and len(wn_batch.shape) > 1
                    else (
                        len(wn_batch[0])
                        if len(wn_batch) > 0 and hasattr(wn_batch[0], "__len__")
                        else 0
                    )
                )

                wn_batches_details.append(
                    {
                        "batchTokenCount": batch_token_count,
                        "batchSpaCyTokenCount": batch_token_count,
                        "wnSensesLength": wn_senses_length,
                    }
                )

            debug_info["wn_batches_details"] = wn_batches_details
            debug_info["debug_batch_sizes"] = [len(batch) for batch in batched_pos]
            debug_info["debug_batch_sizes_sum"] = sum(debug_info["debug_batch_sizes"])
            debug_info["sentence_lengths"] = sentence_lengths
            debug_info["ordering"] = ordering.tolist()

        # Prepare compact per-batch debug dumps for any batches overlapping our ranges
        batch_debug_details = []
        try:
            ranges = (
                debug_info.get("collect_batch_debug_ranges", []) if debug_info else []
            )
            for b in range(len(batched_sents)):
                # collect token ids in this batch
                try:
                    batch_tokens = []
                    for sent in batched_pos[b]:
                        if sent is None:
                            continue
                        for tok in sent:
                            if tok is None:
                                continue
                            batch_tokens.append(getattr(tok, "token_id", None))
                    if len(batch_tokens) == 0:
                        continue
                    gstart = min([t for t in batch_tokens if t is not None])
                    gend = max([t for t in batch_tokens if t is not None])
                except Exception:
                    continue

                overlaps = False
                for r in ranges:
                    if gstart <= r["end"] and gend >= r["start"]:
                        overlaps = True
                        break

                if not overlaps:
                    continue

                # include tokenization and transform details (convert tensors to lists)
                try:
                    input_ids = [
                        x.tolist() if hasattr(x, "tolist") else list(x)
                        for x in batched_data[b]
                    ]
                except Exception:
                    try:
                        input_ids = [list(x) for x in batched_data[b]]
                    except Exception:
                        input_ids = None

                try:
                    mask = [
                        x.tolist() if hasattr(x, "tolist") else list(x)
                        for x in batched_mask[b]
                    ]
                except Exception:
                    try:
                        mask = [list(x) for x in batched_mask[b]]
                    except Exception:
                        mask = None

                try:
                    transforms_list = (
                        batched_transforms[b].tolist()
                        if hasattr(batched_transforms[b], "tolist")
                        else list(batched_transforms[b])
                    )
                except Exception:
                    transforms_list = None

                try:
                    orig_lens = (
                        batched_orig_token_lens[b].tolist()
                        if hasattr(batched_orig_token_lens[b], "tolist")
                        else list(batched_orig_token_lens[b])
                    )
                except Exception:
                    orig_lens = None

                batch_debug_details.append(
                    {
                        "batch_idx": b,
                        "global_start_token": int(gstart),
                        "global_end_token": int(gend),
                        "input_ids": input_ids,
                        "attention_mask": mask,
                        "transforms": transforms_list,
                        "orig_token_lens": orig_lens,
                    }
                )
        except Exception:
            batch_debug_details = []

        if debug_info is not None and batch_debug_details:
            debug_info["batch_debug_details"] = batch_debug_details
            debug_info["ordered_sentence_lengths"] = [
                sentence_lengths[idx] for idx in ordering
            ]

        preds_in_order, events_in_order, supersense_preds_in_order = self.model.tag_all(
            wn_batches,
            batched_sents,
            batched_data,
            batched_mask,
            batched_transforms,
            batched_orig_token_lens,
            ordering,
            doEvent=doEvent,
            doEntities=doEntities,
            doSS=doSS,
        )

        return_vals = {}

        if doEntities:
            for idx, preds in enumerate(preds_in_order):
                for tmp, label, start, end in preds:
                    start_token = sents[idx][start].token_id
                    end_token = sents[idx][end - 1].token_id
                    phrase = " ".join([x.text for x in sents[idx][start:end]])
                    phraseEndToken = int(end_token)
                    if phraseEndToken == -2:
                        phraseEndToken = start_token
                    entities.append((start_token, phraseEndToken, label, phrase))
            return_vals["entities"] = entities

        if doSS:
            # Track supersense annotations at problematic positions for debugging
            supersense_debug_positions = []

            for idx, preds in enumerate(supersense_preds_in_order):
                for tmp, label, start, end in preds:
                    start_token = sents[idx][start].token_id
                    end_token = sents[idx][end - 1].token_id
                    phrase = " ".join([x.text for x in sents[idx][start:end]])
                    phraseEndToken = int(end_token)
                    if phraseEndToken == -2:
                        phraseEndToken = start_token
                    supersense_entities.append(
                        (start_token, phraseEndToken, label, phrase)
                    )

                    # Debug: Track problematic token ranges
                    # Reference: TypeScript entity-tagger.ts debug ranges
                    if (
                        start_token
                        in [
                            26968,
                            26969,
                            26979,
                            27033,
                            27049,
                            27186,
                            27266,
                            27286,
                            30357,
                            34811,
                        ]
                        or (26960 <= start_token <= 26990)
                        or (27030 <= start_token <= 27060)
                        or (27180 <= start_token <= 27300)
                        or (30350 <= start_token <= 30365)
                        or (34805 <= start_token <= 34820)
                    ):
                        supersense_debug_positions.append(
                            {
                                "start": start_token,
                                "end": phraseEndToken,
                                "category": label,
                                "text": phrase,
                            }
                        )

            debug_info["supersense_debug_positions"] = supersense_debug_positions
            return_vals["supersense"] = supersense_entities

        if doEvent:
            events = {}
            for idx, preds in enumerate(events_in_order):
                for start in preds:
                    start_token = sents[idx][start].token_id
                    phrase = sents[idx][start].text
                    events[start_token] = 1
                return_vals["events"] = events

        debug_info["batches_count"] = len(batched_sents)
        debug_info["first_phase_chunks"] = first_phase_chunks
        debug_info["second_phase_groups"] = second_phase_groups
        debug_info["sentence_chunks_count"] = first_phase_chunks
        debug_info["sentence_groups_before_sort"] = sentence_groups_count
        debug_info["ordered_sentences_count"] = len(ordering)
        debug_info["chunk_wordpiece_lengths"] = (
            chunk_wordpiece_lengths  # First-phase chunk lengths for debugging
        )
        debug_info["chunk_wordpiece_samples"] = chunk_wordpiece_samples
        # Include a small sample of per-token wordpiece splits so TS and Python can be diffed
        debug_info["sents_sample"] = sents_sample
        debug_info["tokenizer_cap_token_id"] = cap_token_id
        debug_info["second_phase_group_lengths"] = group_lengths
        debug_info["second_phase_group_chunk_counts"] = group_chunk_counts
        debug_info["raw_batch_sample"] = {
            "batched_sents_count": len(batched_sents[:1]),
            "entity_predictions_sample": [
                str(p) for p in preds_in_order[:1] if doEntities
            ],
        }
        debug_info["extracted_entities_count"] = len(entities) if doEntities else 0
        debug_info["extracted_supersense_count"] = (
            len(supersense_entities) if doSS else 0
        )

        return return_vals
