"""
Generates minimal pairs from the spaCy-enriched CS-only CSV.
Invalid sentences have the switch moved to an EC-violating position,
with the relabelled token span translated so the surface text genuinely changes.

Usage:
    pip install transformers torch sentencepiece sacremoses
    python build_pairs.py --input miami_preprocessed_cs_only.csv --output pairs.csv
    python build_pairs.py --input miami_preprocessed_cs_only.csv --output pairs.csv --limit 50
"""

import csv
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from collections import Counter


# ── EC rules 

EC_FREE = {
    ('NOUN','VERB'), ('PRON','VERB'), ('PROPN','VERB'),
    ('NOUN','AUX'),  ('PRON','AUX'),
    ('VERB','NOUN'), ('VERB','PRON'), ('VERB','DET'),
    ('VERB','ADP'),  ('VERB','ADV'),  ('VERB','ADJ'),
    ('CCONJ','NOUN'),('CCONJ','PRON'),('CCONJ','VERB'),
    ('SCONJ','NOUN'),('SCONJ','PRON'),('SCONJ','VERB'),
    ('ADV','VERB'),  ('ADV','NOUN'),
    ('INTJ','VERB'), ('INTJ','NOUN'), ('INTJ','PRON'),
    ('NOUN','ADJ'),
}

EC_TIGHT = {
    ('DET','NOUN'):  'DET-NOUN dependency',
    ('DET','PROPN'): 'DET-PROPN dependency',
    ('DET','ADJ'):   'DET-ADJ (pre-nominal)',
    ('ADJ','NOUN'):  'ADJ-NOUN dependency',
    ('ADJ','PROPN'): 'ADJ-PROPN dependency',
    ('AUX','VERB'):  'AUX-VERB dependency',
    ('AUX','AUX'):   'AUX-AUX (modal+have)',
    ('ADP','DET'):   'ADP-DET (prep phrase)',
    ('ADP','NOUN'):  'ADP-NOUN (prep phrase)',
    ('ADP','PROPN'): 'ADP-PROPN (prep phrase)',
    ('ADP','PRON'):  'ADP-PRON (prep phrase)',
    ('PART','VERB'): 'PART-VERB dependency',
    ('NUM','NOUN'):  'NUM-NOUN dependency',
}


def boundary_type(lp: str, rp: str) -> str:
    if lp == 'UNK' or rp == 'UNK':
        return 'unknown'
    if (lp, rp) in EC_FREE:
        return 'free'
    if (lp, rp) in EC_TIGHT:
        return 'tight'
    return 'free'


# ── Translator 

class Translator:
    _CACHE: Dict[str, tuple] = {}
    _ISO = {'eng': 'en', 'spa': 'es'}

    def _load(self, src: str, tgt: str):
        s, t = self._ISO.get(src, src), self._ISO.get(tgt, tgt)
        key = f"{s}-{t}"
        if key not in Translator._CACHE:
            from transformers import MarianMTModel, MarianTokenizer
            name = f"Helsinki-NLP/opus-mt-{s}-{t}"
            print(f"  Loading: {name}")
            tok = MarianTokenizer.from_pretrained(name)
            mdl = MarianMTModel.from_pretrained(name)
            mdl.eval()
            Translator._CACHE[key] = (tok, mdl)
        return Translator._CACHE[key]

    def translate(self, texts: List[str], src: str, tgt: str) -> List[str]:
        import torch
        tok, mdl = self._load(src, tgt)
        inputs = tok(texts, return_tensors='pt', padding=True,
                     truncation=True, max_length=64)
        with torch.no_grad():
            out = mdl.generate(**inputs, num_beams=4, max_new_tokens=64)
        return [tok.decode(t, skip_special_tokens=True) for t in out]

    def translate_span(self, tokens: List[str], start: int, end: int,
                       src: str, tgt: str) -> List[str]:
        text = ' '.join(tokens[start:end])
        translated = self.translate([text], src, tgt)[0].split()
        return tokens[:start] + translated + tokens[end:]


# ── Data model 

@dataclass
class Sentence:
    pair_id: int
    sentence_id: int
    label: int
    tokens: List[str]
    lang_tags: List[str]
    pos_tags: List[str]
    switch_pos: int
    violation_type: str
    n_variants: int = 0
    translated_span: str = ''

    @property
    def sentence(self) -> str:
        return ' '.join(self.tokens)


# ── Switch detection 

def find_switches(lang_tags: List[str]) -> List[int]:
    return [
        i for i in range(1, len(lang_tags))
        if lang_tags[i] in ('eng','spa')
        and lang_tags[i-1] in ('eng','spa')
        and lang_tags[i] != lang_tags[i-1]
    ]


# ── Perturbation

def generate_variants(tokens, lang_tags, pos_tags, sw, translator):
    variants = []
    lang_before = lang_tags[sw - 1]
    lang_after  = lang_tags[sw]

    emb_start = sw
    emb_end   = sw
    while emb_end < len(lang_tags) and lang_tags[emb_end] == lang_after:
        emb_end += 1

    valid_surface = ' '.join(tokens)

    for new_pos in range(1, len(tokens)):
        if new_pos == sw:
            continue
        lp = pos_tags[new_pos - 1]
        rp = pos_tags[new_pos]
        if lp == 'UNK' or rp == 'UNK':
            continue
        if (lp, rp) not in EC_TIGHT:
            continue

        if new_pos < sw:
            tr_start, tr_end = new_pos, emb_start
            tr_from,  tr_to  = lang_before, lang_after
            new_lang = lang_tags.copy()
            for k in range(new_pos, emb_end):
                new_lang[k] = lang_after
        else:
            tr_start, tr_end = emb_start, new_pos
            tr_from,  tr_to  = lang_after, lang_before
            new_lang = lang_tags.copy()
            for k in range(emb_start, new_pos):
                new_lang[k] = lang_before

        if tr_start >= tr_end:
            continue
        if len(set(l for l in new_lang if l in ('eng','spa'))) < 2:
            continue
        if new_pos not in find_switches(new_lang):
            continue

        try:
            original_span = tokens[tr_start:tr_end]
            new_tokens = translator.translate_span(
                tokens, tr_start, tr_end, tr_from, tr_to)
        except Exception as e:
            continue

        new_surface = ' '.join(new_tokens)
        if new_surface == valid_surface:
            continue

        delta = len(new_tokens) - len(tokens)
        tr_len = tr_end - tr_start + delta

        new_lang_r = new_lang[:tr_start] + [tr_to]*tr_len + new_lang[tr_end:]
        new_pos_r  = pos_tags[:tr_start] + ['UNK']*tr_len + pos_tags[tr_end:]
        adj_sw     = new_pos if new_pos <= tr_start else new_pos + delta

        span_str = (
            f"'{' '.join(original_span)}'"
            f" -> '{' '.join(new_tokens[tr_start:tr_start+tr_len])}'"
            f" ({tr_from}->{tr_to})"
        )

        variants.append((new_tokens, new_lang_r, new_pos_r,
                         adj_sw, EC_TIGHT[(lp,rp)], span_str))

    return variants


# ── Pipeline 

def process_corpus(input_path, strict=False, limit=None):
    with open(input_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    print(f"Input rows: {len(rows)}")
    translator = Translator()

    all_sentences = []
    pair_id = 0
    stats = dict(too_short=0, no_switch=0, skipped=0, no_invalid=0, pairs=0)

    for idx, row in enumerate(rows):
        if (idx+1) % 200 == 0:
            print(f"  Row {idx+1}/{len(rows)} — {stats['pairs']} pairs so far...")

        tokens    = json.loads(row['tokens'])
        lang_tags = json.loads(row['lang_tags'])
        pos_tags  = json.loads(row['pos_tags'])

        if len(tokens) < 5:
            stats['too_short'] += 1; continue

        switches = find_switches(lang_tags)
        if not switches:
            stats['no_switch'] += 1; continue

        sw = switches[0]
        btype = boundary_type(pos_tags[sw-1], pos_tags[sw])
        if strict and btype == 'unknown':
            stats['skipped'] += 1; continue

        variants = generate_variants(tokens, lang_tags, pos_tags, sw, translator)

        if not variants:
            stats['no_invalid'] += 1; continue

        all_sentences.append(Sentence(
            pair_id=pair_id, sentence_id=idx, label=1,
            tokens=tokens, lang_tags=lang_tags, pos_tags=pos_tags,
            switch_pos=sw, violation_type='', n_variants=len(variants),
        ))

        for vt, vl, vp, vsw, vvtype, vspan in variants:
            all_sentences.append(Sentence(
                pair_id=pair_id, sentence_id=idx, label=0,
                tokens=vt, lang_tags=vl, pos_tags=vp,
                switch_pos=vsw, violation_type=vvtype,
                n_variants=len(variants), translated_span=vspan,
            ))

        pair_id += 1
        stats['pairs'] += 1

    print(f"\nStats:")
    print(f"  Too short:         {stats['too_short']}")
    print(f"  No switch:         {stats['no_switch']}")
    print(f"  Skipped (strict):  {stats['skipped']}")
    print(f"  No invalid found:  {stats['no_invalid']}")
    print(f"  Pairs generated:   {stats['pairs']}")
    print(f"  Total records:     {len(all_sentences)}")
    print(f"    valid   (1):     {sum(1 for s in all_sentences if s.label==1)}")
    print(f"    invalid (0):     {sum(1 for s in all_sentences if s.label==0)}")

    vtypes = Counter(s.violation_type for s in all_sentences if s.label==0)
    print(f"\nViolation breakdown:")
    for vt, c in vtypes.most_common():
        print(f"  {c:5d}  {vt}")

    return all_sentences


def write_output(sentences, output_path):
    fieldnames = ['pair_id','sentence_id','label','sentence',
                  'tokens','lang_tags','pos_tags',
                  'switch_pos','violation_type','n_variants','translated_span']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sentences:
            writer.writerow({
                'pair_id': s.pair_id, 'sentence_id': s.sentence_id,
                'label': s.label, 'sentence': s.sentence,
                'tokens': json.dumps(s.tokens, ensure_ascii=False),
                'lang_tags': json.dumps(s.lang_tags, ensure_ascii=False),
                'pos_tags': json.dumps(s.pos_tags, ensure_ascii=False),
                'switch_pos': s.switch_pos, 'violation_type': s.violation_type,
                'n_variants': s.n_variants, 'translated_span': s.translated_span,
            })
    print(f"\nWritten -> {output_path}")

    # Sample
    # only show invalids that differ from their valid partner
    print("\n-- Sample pairs (first 3 with genuine surface changes) --")
    valid_map = {s.pair_id: s.sentence for s in sentences if s.label == 1}
    shown, seen = 0, set()
    for s in sentences:
        if s.label != 0 or s.pair_id in seen:
            continue
        if s.sentence == valid_map.get(s.pair_id, ''):
            continue  # skip identical pairs in display
        seen.add(s.pair_id)
        valid_s = next(x for x in sentences if x.pair_id == s.pair_id and x.label == 1)
        invs = [x for x in sentences if x.pair_id == s.pair_id and x.label == 0
                and x.sentence != valid_map[s.pair_id]]
        print(f"\nPAIR {s.pair_id}")
        print(f"  VALID:   {valid_s.sentence}")
        for inv in invs[:2]:
            print(f"  INVALID: {inv.sentence}")
            print(f"           [{inv.violation_type}] {inv.translated_span}")
        shown += 1
        if shown >= 3:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  type=Path, default=Path('miami_preprocessed_cs_only.csv'))
    parser.add_argument('--output', type=Path, default=Path('pairs.csv'))
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--limit',  type=int, default=None)
    args = parser.parse_args()
    sentences = process_corpus(args.input, args.strict, args.limit)
    write_output(sentences, args.output)
    print('\nCompleted. Run score_perplexity.py on pairs.csv.')


if __name__ == '__main__':
    main()
