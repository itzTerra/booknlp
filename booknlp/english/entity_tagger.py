from booknlp.english.tagger import Tagger
import torch
import re
import booknlp.common.layered_reader as layered_reader
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
        if "bert.embeddings.position_ids" in state_dict:
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
                if sent is not None and len(sent) > max_len:
                    max_len = len(sent)

            wn_senses = []

            for sent in b_sent:
                if sent is None:
                    continue

                wn = []
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

                for _ in range(len(sent), max_len):
                    wn.append(0)
                wn_senses.append(wn)

            wn_senses = torch.LongTensor(wn_senses)
            wn_batches.append(wn_senses)
        return wn_batches

    def tag(self, toks, doEvent=True, doEntities=True, doSS=True):
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
            if wptok and wptok[0].lower() != wptok[0]:
                wptok = "[CAP] " + wptok.lower()

            wp_tokens = self.model.tokenizer.tokenize(wptok)
            if lastSid is not None and (
                tok.sentence_id != lastSid
                or length + len(wp_tokens) > max_sentence_length
            ):
                sents.append(sent)
                o_sents.append(o_sent)
                split_count += 1
                sent = []
                o_sent = []
                length = 0

            sent.append(wp_tokens)
            o_sent.append(tok)

            lastSid = tok.sentence_id
            length += len(wp_tokens)

        if sent:
            sents.append(sent)
            o_sents.append(o_sent)

        first_phase_count = len(sents)

        # Calculate wordpiece lengths for each first-phase chunk
        chunk_wordpiece_lengths = []
        chunk_wordpiece_samples = []
        for sent in sents:
            chunk_len = sum(len(wp) for wp in sent)
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
            sent_len = sum(len(wp) for wp in sent)

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

        if group_chunk_count > 0:
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

        batch_pos = {}
        for idx, ind in enumerate(ordering):
            batch_id, batch_s, batch_position = order_to_batch_map[idx]
            if batch_id not in batch_pos:
                batch_pos[batch_id] = [None] * batch_s
            batch_pos[batch_id][batch_position] = [None]
            for tok in sents[ind]:
                batch_pos[batch_id][batch_position].append(tok)
            batch_pos[batch_id][batch_position].append(None)

        batched_pos = [None] * len(batch_pos)
        for i in range(len(batch_pos)):
            batched_pos[i] = batch_pos[i]

        wn_batches = self._get_wn(batched_pos)

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

            return_vals["supersense"] = supersense_entities

        if doEvent:
            events = {}
            for idx, preds in enumerate(events_in_order):
                for start in preds:
                    start_token = sents[idx][start].token_id
                    phrase = sents[idx][start].text
                    events[start_token] = 1
            return_vals["events"] = events

        return return_vals
