"""
Parses the raw CHAT (.cha) files from the Bangor Miami corpus into a clean,
structured CSV ready for the perturbation pipeline .

CORPUS STRUCTURE
  eng/   English-dominant conversations; Spanish words marked with @s suffix
  spa/   Spanish-dominant conversations; English words marked with @s suffix

Language tagging convention:
  word    matrix language (English in eng/, Spanish in spa/)
  word@s   embedded language (Spanish in eng/, English in spa/)
  word@s:eng   explicitly English (used for isolated tags, loanwords, etc.)
  word@s:spa   explicitly Spanish
  word@s:ita etc  other languages (rare, filtered out)

POS tagging:
  The %mor tier provides POS for matrix-language words.
  Embedded-language (@s) words are tagged L2|xxx (unknown).
  Gets assigned as POS using spaCy for embedded tokens after parsing.

Output CSV columns:
  file, speaker, folder (eng/spa), utterance, tokens (JSON list),
  lang_tags (JSON list: 'matrix'/'embedded'/'eng'/'spa'/'other'),
  pos_tags (JSON list, UD tagset), is_cs, n_switches, n_tokens

Usage:
    pip install spacy
    python -m spacy download en_core_web_sm
    python -m spacy download es_core_news_sm

    python preprocess_miami.py --corpus_dir ./miami_corpus --output miami_preprocessed.csv
    python preprocess_miami.py --corpus_dir ./miami_corpus --output miami_preprocessed.csv --cs_only
"""

import re
import csv
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Optional spaCy import 
try:
    import spacy
    _nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    _nlp_es = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    USE_SPACY = True
    print("spaCy loaded: POS tags will be assigned for all tokens.")
except Exception:
    USE_SPACY = False
    print("spaCy not available: POS tags will be taken from %mor tier only "
          "(embedded-language tokens will be UNK).")

# ── Constants 

# CHAT noise patterns to strip from main tier before tokenising
CHAT_NOISE = re.compile(
    r'\x15\d+_\d+\x15'             # sound links
    r'|&[=+\-~]\w+'                 # non-lexical events: &=laughs, &+uh, &~co
    r'|[<>]'                        # overlap angle brackets
    r'|\[[^\]]*\]'                  # editorial comments: [/], [//], [?], [*]
    r'|\+[/\\.,?!\"]{1,2}'          # continuation markers: +/. +//. +"/.
    r'|\+\.\.\.'                    # trailing off: +...
    r'|\bxxx\b|\byyy\b|\bwww\b'     # unintelligible / untranscribed
    r'|\(\.\.*\)'                   # pauses: (.) (..)
    r'|\x00'                        # null bytes
)

# Language tag on a token: @s alone, or @s:xxx
LANG_TAG = re.compile(r'@s(?::([a-z&]+))?')

# %mor POS entry: pos|lemma or pfx~pos|lemma
MOR_POS = re.compile(r'(?:^|~)(\w[\w:]*)\|')

# Bangor/UD POS mapping
BANGOR_UD = {
    "noun": "NOUN", "n": "NOUN", "n:prop": "PROPN", "propn": "PROPN",
    "pro": "PRON", "pro:dem": "PRON", "pro:per": "PRON", "pro:poss": "PRON",
    "pro:refl": "PRON", "pro:rel": "PRON", "pro:int": "PRON",
    "pro:exist": "PRON", "pro:indef": "PRON",
    "verb": "VERB", "v": "VERB", "cop": "AUX", "aux": "AUX",
    "part": "VERB", "adj": "ADJ", "adv": "ADV", "adv:int": "ADV",
    "adv:neg": "ADV", "prep": "ADP", "adp": "ADP",
    "det": "DET", "det:art": "DET",
    "conj": "CCONJ", "cconj": "CCONJ", "sconj": "SCONJ",
    "conj:subor": "SCONJ", "num": "NUM", "int": "INTJ", "intj": "INTJ",
    "ptl": "PART", "neg": "PART", "inf": "PART", "poss": "PART",
    "bab": "INTJ", "l2": "UNK",
}


def mor_to_ud(mor_tag: str) -> str:
    tag = mor_tag.lower().strip()
    return BANGOR_UD.get(tag, tag.upper() or "UNK")


# ── Data model 

@dataclass
class Token:
    form: str
    lang: str        # 'matrix' | 'embedded' | 'eng' | 'spa' | 'other'
    pos: str = "UNK"


@dataclass
class Utterance:
    file: str
    folder: str       # 'eng' or 'spa'
    speaker: str
    raw_text: str
    tokens: List[Token] = field(default_factory=list)

    # Resolved language labels
    @property
    def abs_lang(self, tok: Token) -> str:
        """Return absolute language for a token given the folder context."""
        if tok.lang == "matrix":
            return "eng" if self.folder == "eng" else "spa"
        if tok.lang == "embedded":
            return "spa" if self.folder == "eng" else "eng"
        return tok.lang

    @property
    def lang_sequence(self) -> List[str]:
        result = []
        for t in self.tokens:
            if t.lang == "matrix":
                result.append("eng" if self.folder == "eng" else "spa")
            elif t.lang == "embedded":
                result.append("spa" if self.folder == "eng" else "eng")
            else:
                result.append(t.lang)
        return result

    @property
    def is_cs(self) -> bool:
        langs = set(self.lang_sequence)
        return "eng" in langs and "spa" in langs

    @property
    def n_switches(self) -> int:
        content = [l for l in self.lang_sequence if l in ("eng", "spa")]
        return sum(1 for i in range(1, len(content)) if content[i] != content[i-1])

    @property
    def token_forms(self) -> List[str]:
        return [t.form for t in self.tokens]

    @property
    def pos_sequence(self) -> List[str]:
        return [t.pos for t in self.tokens]


# ── CHAT parsing 

def resolve_lang_tag(raw: str) -> Tuple[str, str]:
    m = LANG_TAG.search(raw)
    if m:
        explicit = m.group(1)  # None if bare @s, else 'eng', 'spa', 'ita', etc.
        clean = LANG_TAG.sub("", raw).strip("@+").strip()
        if explicit is None:
            lang = "embedded"
        elif "eng" in explicit and "spa" in explicit:
            lang = "ambig"
        elif explicit in ("eng", "spa"):
            lang = explicit
        else:
            lang = "other"
    else:
        clean = raw.strip()
        lang = "matrix"
    # Strip stray punctuation from the form
    clean = re.sub(r'^[.?!,;:()\-]+|[.?!,;:()\-]+$', '', clean).strip()
    return clean, lang


def parse_mor_pos(mor_line: str) -> List[str]:
    tags = []
    for entry in mor_line.split():
        if entry in ("+/.", "+//.", "+...", "+\"/.", "."):
            continue
        # Find the last pos|lemma chunk (after any ~ prefix)
        parts = entry.split("~")
        m = MOR_POS.match(parts[-1])
        tags.append(mor_to_ud(m.group(1)) if m else "UNK")
    return tags


def clean_main_tier(raw: str) -> str:
    text = CHAT_NOISE.sub(" ", raw)
    # Remove terminal punctuation added by CHAT
    text = re.sub(r'\s*[.?!]\s*$', '', text)
    return " ".join(text.split())


def merge_continuation(lines: List[str], start: int) -> Tuple[str, int]:
    line = lines[start].rstrip("\n")
    i = start + 1
    while i < len(lines) and lines[i].startswith("\t"):
        line += " " + lines[i].strip()
        i += 1
    return line, i


def parse_cha_file(filepath: Path) -> List[Utterance]:
    folder = filepath.parts[-3]  # 'eng' or 'spa'
    utterances = []
    pending_main: Optional[Tuple[str, str, str]] = None  # (speaker, cleaned, raw)
    pending_mor: Optional[str] = None

    def flush():
        nonlocal pending_main, pending_mor
        if pending_main is None:
            return
        speaker, cleaned, _ = pending_main
        utt = build_utterance(filepath.stem, folder, speaker, cleaned, pending_mor)
        if utt and utt.tokens:
            utterances.append(utt)
        pending_main = None
        pending_mor = None

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line, i = merge_continuation(lines, i)

        if line.startswith("*"):
            flush()
            m = re.match(r'\*(\w+):\s*(.*)', line)
            if m:
                speaker = m.group(1)
                raw_text = m.group(2)
                cleaned = clean_main_tier(raw_text)
                if cleaned:
                    pending_main = (speaker, cleaned, raw_text)

        elif line.startswith("%mor:") and pending_main is not None:
            pending_mor = line[5:].strip()
            flush()

        elif line.startswith("%") and pending_main is not None:
            pass

    flush()
    return utterances


def build_utterance(source: str, folder: str, speaker: str,
                    cleaned: str, mor: Optional[str]) -> Optional[Utterance]:
    raw_tokens = cleaned.split()
    if not raw_tokens:
        return None

    tokens = []
    for raw in raw_tokens:
        form, lang = resolve_lang_tag(raw)
        if form:
            tokens.append(Token(form=form, lang=lang))

    if not tokens:
        return None

    # Assign POS from %mor tier 
    if mor:
        mor_pos = parse_mor_pos(mor)
        # Align by index, skipping mismatches gracefully
        mi = 0
        for tok in tokens:
            if mi < len(mor_pos):
                if tok.lang in ("matrix", "eng", "spa"):
                    tok.pos = mor_pos[mi]
                mi += 1

    return Utterance(
        file=source,
        folder=folder,
        speaker=speaker,
        raw_text=cleaned,
        tokens=tokens,
    )


# ── spaCy POS enrichment 

def enrich_pos_spacy(utterances: List[Utterance]) -> None:

    if not USE_SPACY:
        return

    print("Enriching POS tags with spaCy...")
    # Collect (utt_idx, tok_idx, form, lang) for UNK tokens
    to_tag_en: List[Tuple[int, int, str]] = []
    to_tag_es: List[Tuple[int, int, str]] = []

    for ui, utt in enumerate(utterances):
        for ti, tok in enumerate(utt.tokens):
            if tok.pos == "UNK":
                abs_lang = utt.lang_sequence[ti]
                if abs_lang == "eng":
                    to_tag_en.append((ui, ti, tok.form))
                elif abs_lang == "spa":
                    to_tag_es.append((ui, ti, tok.form))

    def batch_tag(nlp, items):
        if not items:
            return
        texts = [form for _, _, form in items]
        for (ui, ti, _), doc in zip(items, nlp.pipe(texts, batch_size=256)):
            if doc:
                utterances[ui].tokens[ti].pos = doc[0].pos_

    batch_tag(_nlp_en, to_tag_en)
    batch_tag(_nlp_es, to_tag_es)
    print(f"  Tagged {len(to_tag_en)} English tokens, {len(to_tag_es)} Spanish tokens.")


# ── Output 

def write_csv(utterances: List[Utterance], output_path: Path) -> None:
    cs = [u for u in utterances if u.is_cs]
    print(f"\nTotal utterances parsed:            {len(utterances)}")
    print(f"Code-switched (eng+spa):            {len(cs)}")
    print(f"  from eng/ folder:                 {sum(1 for u in cs if u.folder=='eng')}")
    print(f"  from spa/ folder:                 {sum(1 for u in cs if u.folder=='spa')}")

    fieldnames = ["file", "folder", "speaker", "utterance",
                  "tokens", "lang_tags", "pos_tags",
                  "is_cs", "n_switches", "n_tokens"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for u in utterances:
            writer.writerow({
                "file": u.file,
                "folder": u.folder,
                "speaker": u.speaker,
                "utterance": " ".join(u.token_forms),
                "tokens": json.dumps(u.token_forms, ensure_ascii=False),
                "lang_tags": json.dumps(u.lang_sequence, ensure_ascii=False),
                "pos_tags": json.dumps(u.pos_sequence, ensure_ascii=False),
                "is_cs": u.is_cs,
                "n_switches": u.n_switches,
                "n_tokens": len(u.tokens),
            })
    print(f"\nWritten → {output_path}")


# ── CLI 

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Bangor Miami Corpus .cha files into CSV."
    )
    parser.add_argument(
        "--corpus_dir", type=Path, required=True,
        help="Path to the extracted Miami corpus folder (contains eng/ and spa/)."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("miami_preprocessed.csv"),
        help="Output CSV to miami_preprocessed.csv"
    )
    parser.add_argument(
        "--cs_only", action="store_true",
        help="Also write a *_cs_only.csv with only code-switched utterances."
    )
    args = parser.parse_args()

    cha_files = sorted(args.corpus_dir.rglob("*.cha"))
    if not cha_files:
        raise FileNotFoundError(f"No .cha files found in {args.corpus_dir}")
    print(f"Found {len(cha_files)} .cha files across eng/ and spa/.\n")

    all_utterances: List[Utterance] = []
    for f in cha_files:
        utts = parse_cha_file(f)
        all_utterances.extend(utts)
        print(f"  {f.parts[-2]}/{f.name}: {len(utts)} utterances "
              f"({sum(1 for u in utts if u.is_cs)} CS)")

    enrich_pos_spacy(all_utterances)

    write_csv(all_utterances, args.output)

    if args.cs_only:
        cs_path = args.output.with_name(args.output.stem + "_cs_only.csv")
        cs_utts = [u for u in all_utterances if u.is_cs]
        # Temporarily reassign to write only CS
        orig = all_utterances
        write_csv(cs_utts, cs_path)

    print("\nCompleted. Run build_pairs.py on the CS-only CSV.")


if __name__ == "__main__":
    main()
