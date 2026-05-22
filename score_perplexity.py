"""
Computes masked-language-model (MLM) pseudo-perplexity (PPPL) for each
sentence in pairs.csv using mBERT and XLM-RoBERTa.

Two scoring methods are supported via --method:

  salazar (default):
    Salazar et al. (2020)
    Mask one subword token at a time.
    PPPL = exp( -1/N * sum_i log P(subtoken_i | all others visible) )
    N = total subword tokens.

  kauf (corrected):
    Kauf & Ivanova (2023) 
    Mask ALL subtokens of the same word simultaneously,
    sum their log probs, normalise by number of WORDS not subtokens.
    Eliminates bias that inflates scores for multi-subword words under Salazar.
    Especially relevant for code-switched text where Spanish/English words
    tokenize asymmetrically under mBERT/XLM-R.

Run both and compare:
    python score_perplexity.py --input pairs.csv --output pairs_scored_salazar.csv --method salazar
    python score_perplexity.py --input pairs.csv --output pairs_scored_kauf.csv    --method kauf

Usage:
    pip install transformers torch
    python score_perplexity.py --input pairs.csv --output pairs_scored.csv [--method salazar|kauf]
    python score_perplexity.py --input pairs.csv --output pairs_scored.csv --limit 50
    python score_perplexity.py --input pairs.csv --output pairs_scored.csv --models mbert
"""

import csv
import json
import math
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr':  'xlm-roberta-base',
}


# ── Word-to-subtoken grouping 

def get_word_spans(encoding, tokenizer) -> List[List[int]]:

    special_ids = set(tokenizer.all_special_ids)
    ids = encoding['input_ids'][0].tolist()

    # Try word_ids() first (most reliable, works for both tokenizer types)
    try:
        word_ids = encoding.word_ids(batch_index=0)
        groups: Dict[int, List[int]] = {}
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue  # special token
            if ids[pos] in special_ids:
                continue
            groups.setdefault(wid, []).append(pos)
        return list(groups.values())
    except Exception:
        pass

    # Fallback
    tokens = tokenizer.convert_ids_to_tokens(ids)
    groups = []
    current = []
    for pos, tok in enumerate(tokens):
        if ids[pos] in special_ids:
            if current:
                groups.append(current)
                current = []
            continue
        if tok.startswith('##') and current:
            current.append(pos)
        else:
            if current:
                groups.append(current)
            current = [pos]
    if current:
        groups.append(current)
    return groups


# ── Salazar et al. (2020) method 

def compute_pppl_salazar(sentence, tokenizer, model, device) -> float:

    encoding = tokenizer(sentence, return_tensors='pt',
                         truncation=True, max_length=512)
    input_ids     = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    special_ids = set(tokenizer.all_special_ids)
    token_positions = [
        i for i, tid in enumerate(input_ids[0].tolist())
        if tid not in special_ids
    ]
    if not token_positions:
        return float('inf')

    total_log_prob = 0.0
    with torch.no_grad():
        for pos in token_positions:
            masked = input_ids.clone()
            masked[0, pos] = tokenizer.mask_token_id
            logits = model(input_ids=masked,
                           attention_mask=attention_mask).logits[0, pos]
            log_probs = torch.log_softmax(logits, dim=-1)
            total_log_prob += log_probs[input_ids[0, pos].item()].item()

    return math.exp(-total_log_prob / len(token_positions))


# ── Kauf & Ivanova (2023) corrected method 

def compute_pppl_kauf(sentence, tokenizer, model, device) -> float:

    encoding = tokenizer(sentence, return_tensors='pt',
                         truncation=True, max_length=512)
    input_ids      = encoding['input_ids'].to(device)
    attention_mask  = encoding['attention_mask'].to(device)

    word_spans = get_word_spans(encoding, tokenizer)
    if not word_spans:
        return float('inf')

    total_word_log_prob = 0.0
    with torch.no_grad():
        for span in word_spans:
            # Mask all subtokens of this word at once
            masked = input_ids.clone()
            for pos in span:
                masked[0, pos] = tokenizer.mask_token_id

            logits_all = model(input_ids=masked,
                               attention_mask=attention_mask).logits[0]

            # Sum log probs for all subtokens in this word
            word_log_prob = 0.0
            for pos in span:
                log_probs = torch.log_softmax(logits_all[pos], dim=-1)
                word_log_prob += log_probs[input_ids[0, pos].item()].item()

            total_word_log_prob += word_log_prob

    return math.exp(-total_word_log_prob / len(word_spans))


# ── Dispatcher 

def compute_pppl(sentence, tokenizer, model, device, method='salazar') -> float:
    if method == 'kauf':
        return compute_pppl_kauf(sentence, tokenizer, model, device)
    return compute_pppl_salazar(sentence, tokenizer, model, device)


# ── Batch scoring 

def score_all(sentences, model_key, device, method='salazar',
              batch_report_every=100) -> List[float]:
    model_name = MODELS[model_key]
    print(f"\nLoading {model_key} ({model_name})  [method={method}]...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_obj = AutoModelForMaskedLM.from_pretrained(model_name)
    model_obj.to(device)
    model_obj.eval()
    print(f"Scoring {len(sentences)} sentences on {device}...")

    scores = []
    t0 = time.time()
    for i, sent in enumerate(sentences):
        scores.append(compute_pppl(sent, tokenizer, model_obj, device, method))
        if (i + 1) % batch_report_every == 0:
            elapsed = time.time() - t0
            eta = (len(sentences) - i - 1) / ((i + 1) / elapsed) / 60
            print(f"  [{i+1}/{len(sentences)}]  "
                  f"last={scores[-1]:.2f}  "
                  f"avg={sum(scores)/len(scores):.2f}  "
                  f"ETA {eta:.1f} min")

    elapsed = time.time() - t0
    finite = [s for s in scores if s != float('inf')]
    print(f"Done in {elapsed/60:.1f} min.  "
          f"Mean PPPL={sum(finite)/len(finite):.2f}")

    del model_obj
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return scores


# ── Main 

def main():
    parser = argparse.ArgumentParser(
        description='Score sentence pairs with mBERT/XLM-R pseudo-perplexity.')
    parser.add_argument('--input',  type=Path, default=Path('pairs.csv'))
    parser.add_argument('--output', type=Path, default=Path('pairs_scored.csv'))
    parser.add_argument('--models', type=str, default='both',
                        choices=['both', 'mbert', 'xlmr'])
    parser.add_argument('--method', type=str, default='salazar',
                        choices=['salazar', 'kauf'],
                        help='salazar = original (Salazar et al. 2020); '
                             'kauf = corrected (Kauf & Ivanova 2023)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--limit',  type=int, default=None)
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}  |  Method: {args.method}")

    # Load CSV
    with open(args.input, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if args.limit:
        first_n = sorted(set(r['pair_id'] for r in rows))[:args.limit]
        rows = [r for r in rows if r['pair_id'] in set(first_n)]
        print(f"Limited to {args.limit} pairs -> {len(rows)} records.")

    sentences = [r['sentence'] for r in rows]
    print(f"Total sentences: {len(sentences)}")

    # Score
    mbert_scores: Optional[List[float]] = None
    xlmr_scores:  Optional[List[float]] = None

    if args.models in ('both', 'mbert'):
        mbert_scores = score_all(sentences, 'mbert', device, args.method)
    if args.models in ('both', 'xlmr'):
        xlmr_scores  = score_all(sentences, 'xlmr',  device, args.method)

    # Write output 
    new_cols = []
    if mbert_scores is not None and 'mbert_pppl' not in original_fieldnames:
        new_cols.append('mbert_pppl')
    if xlmr_scores  is not None and 'xlmr_pppl'  not in original_fieldnames:
        new_cols.append('xlmr_pppl')
    out_fields = original_fieldnames + new_cols

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            if mbert_scores is not None:
                out['mbert_pppl'] = round(mbert_scores[i], 4)
            if xlmr_scores is not None:
                out['xlmr_pppl'] = round(xlmr_scores[i], 4)
            writer.writerow(out)

    print(f"\nWritten -> {args.output}")

    # Sanity check
    print("\n-- Sanity check --")
    valid_idx   = [i for i, r in enumerate(rows) if r['label'] == '1']
    invalid_idx = [i for i, r in enumerate(rows) if r['label'] == '0']
    for name, scores in [('mBERT', mbert_scores), ('XLM-R', xlmr_scores)]:
        if scores is None:
            continue
        vm = sum(scores[i] for i in valid_idx)   / len(valid_idx)
        im = sum(scores[i] for i in invalid_idx) / len(invalid_idx)
        direction = "✓ valid < invalid" if vm < im else "✗ valid > invalid"
        print(f"{name}  valid={vm:.2f}  invalid={im:.2f}  {direction}")


if __name__ == '__main__':
    main()
