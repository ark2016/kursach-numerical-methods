import os
import numpy as np
from collections import Counter
from typing import Optional
from datasets import load_dataset
from tqdm.auto import tqdm
import json
import re
try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not installed. Install with: pip install nltk")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score")


def simple_tokenize(text: str) -> list[str]:    
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text.split()


def get_tokenizer():
    if NLTK_AVAILABLE:
        return word_tokenize
    return simple_tokenize


def get_ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_bleu_score(reference_texts: list[str], hypothesis_texts: list[str],
                       max_n: int = 4, sample_size: Optional[int] = 1000) -> dict:
    tokenize = get_tokenizer()
    if sample_size and len(hypothesis_texts) > sample_size:
        indices = np.random.choice(len(hypothesis_texts), sample_size, replace=False)
        hypothesis_texts = [hypothesis_texts[i] for i in indices]
    if sample_size and len(reference_texts) > sample_size:
        indices = np.random.choice(len(reference_texts), sample_size, replace=False)
        reference_texts = [reference_texts[i] for i in indices]
    ref_tokenized = [tokenize(text) for text in tqdm(reference_texts, desc="Tokenizing references")]
    hyp_tokenized = [tokenize(text) for text in tqdm(hypothesis_texts, desc="Tokenizing hypotheses")]
    results = {}
    if NLTK_AVAILABLE:
        smoothing = SmoothingFunction().method1
        bleu_scores = []
        for hyp in tqdm(hyp_tokenized, desc="Computing BLEU"):
            refs = ref_tokenized[:100]
            score = sentence_bleu(refs, hyp, smoothing_function=smoothing)
            bleu_scores.append(score)
        results["bleu_mean"] = float(np.mean(bleu_scores))
        results["bleu_std"] = float(np.std(bleu_scores))
        results["bleu_median"] = float(np.median(bleu_scores))
    else:
        results["bleu_mean"] = compute_simple_bleu(ref_tokenized, hyp_tokenized, max_n)

    return results


def compute_simple_bleu(references: list[list[str]], hypotheses: list[list[str]], max_n: int = 4) -> float:
    total_score = 0
    count = 0
    for hyp in hypotheses[:100]:
        if len(hyp) == 0:
            continue
        precision_scores = []
        for n in range(1, min(max_n + 1, len(hyp) + 1)):
            hyp_ngrams = Counter(get_ngrams(hyp, n))
            max_ref_counts = Counter()
            for ref in references[:50]:
                ref_ngrams = Counter(get_ngrams(ref, n))
                for ngram in hyp_ngrams:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
            clipped_count = sum(min(hyp_ngrams[ng], max_ref_counts[ng]) for ng in hyp_ngrams)
            total_count = sum(hyp_ngrams.values())
            if total_count > 0:
                precision_scores.append(clipped_count / total_count)
        if precision_scores:
            log_precision = sum(np.log(p + 1e-10) for p in precision_scores) / len(precision_scores)
            total_score += np.exp(log_precision)
            count += 1

    return total_score / max(count, 1)


def compute_rouge_scores(reference_texts: list[str], hypothesis_texts: list[str],
                         sample_size: Optional[int] = 500) -> dict:
    if not ROUGE_AVAILABLE:
        return {"error": "rouge-score not installed"}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    if sample_size:
        if len(hypothesis_texts) > sample_size:
            hypothesis_texts = np.random.choice(hypothesis_texts, sample_size, replace=False).tolist()
        if len(reference_texts) > sample_size:
            reference_texts = np.random.choice(reference_texts, sample_size, replace=False).tolist()
    results = {
        "rouge1_precision": [], "rouge1_recall": [], "rouge1_fmeasure": [],
        "rouge2_precision": [], "rouge2_recall": [], "rouge2_fmeasure": [],
        "rougeL_precision": [], "rougeL_recall": [], "rougeL_fmeasure": [],
    }
    for hyp in tqdm(hypothesis_texts, desc="Computing ROUGE"):
        best_scores = None
        for ref in reference_texts[:20]:
            scores = scorer.score(ref, hyp)
            if best_scores is None:
                best_scores = scores
            else:
                for key in scores:
                    if scores[key].fmeasure > best_scores[key].fmeasure:
                        best_scores = scores
                        break
        if best_scores:
            results["rouge1_precision"].append(best_scores["rouge1"].precision)
            results["rouge1_recall"].append(best_scores["rouge1"].recall)
            results["rouge1_fmeasure"].append(best_scores["rouge1"].fmeasure)
            results["rouge2_precision"].append(best_scores["rouge2"].precision)
            results["rouge2_recall"].append(best_scores["rouge2"].recall)
            results["rouge2_fmeasure"].append(best_scores["rouge2"].fmeasure)
            results["rougeL_precision"].append(best_scores["rougeL"].precision)
            results["rougeL_recall"].append(best_scores["rougeL"].recall)
            results["rougeL_fmeasure"].append(best_scores["rougeL"].fmeasure)
    return {k: float(np.mean(v)) for k, v in results.items()}


def compute_vocabulary_overlap(texts1: list[str], texts2: list[str]) -> dict:
    tokenize = get_tokenizer()
    vocab1 = set()
    vocab2 = set()
    for text in tqdm(texts1, desc="Building vocab1"):
        vocab1.update(tokenize(text))
    for text in tqdm(texts2, desc="Building vocab2"):
        vocab2.update(tokenize(text))
    intersection = vocab1 & vocab2
    union = vocab1 | vocab2
    return {
        "vocab_size_1": len(vocab1),
        "vocab_size_2": len(vocab2),
        "intersection_size": len(intersection),
        "union_size": len(union),
        "jaccard_similarity": len(intersection) / len(union) if union else 0,
        "overlap_ratio_1": len(intersection) / len(vocab1) if vocab1 else 0,
        "overlap_ratio_2": len(intersection) / len(vocab2) if vocab2 else 0,
    }


def compute_self_bleu(texts: list[str], sample_size: int = 500, num_refs: int = 100) -> float:
    tokenize = get_tokenizer()
    if len(texts) > sample_size:
        texts = np.random.choice(texts, sample_size, replace=False).tolist()
    tokenized = [tokenize(text) for text in texts]
    scores = []
    for i, hyp in enumerate(tqdm(tokenized[:sample_size], desc="Computing Self-BLEU")):
        refs = tokenized[:i] + tokenized[i+1:]
        if len(refs) > num_refs:
            ref_indices = np.random.choice(len(refs), num_refs, replace=False)
            refs = [refs[j] for j in ref_indices]
        if NLTK_AVAILABLE:
            smoothing = SmoothingFunction().method1
            score = sentence_bleu(refs, hyp, smoothing_function=smoothing)
        else:
            score = compute_simple_bleu(refs, [hyp], max_n=4)
        scores.append(score)

    return float(np.mean(scores))


def compute_length_statistics(texts: list[str]) -> dict:
    tokenize = get_tokenizer()
    lengths = [len(tokenize(text)) for text in texts]

    return {
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "median_length": float(np.median(lengths)),
    }


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def analyze_datasets_similarity(
    dataset1_name: str = "seara/ru_go_emotions", dataset2_name: str = "Djacon/ru-izard-emotions", split: str = "train",
    text_column: str = "text", sample_size: int = 2000, output_file: Optional[str] = "dataset_similarity_report.json"
) -> dict:
    print(f"Loading datasets...")
    ds1 = load_dataset(dataset1_name, "simplified" if "go_emotions" in dataset1_name else None)
    ds2 = load_dataset(dataset2_name)
    texts1 = ds1[split][text_column]
    texts2 = ds2[split][text_column]
    print(f"Dataset 1: {len(texts1)} samples")
    print(f"Dataset 2: {len(texts2)} samples")
    results = {
        "dataset1": dataset1_name,
        "dataset2": dataset2_name,
        "split": split,
        "dataset1_size": len(texts1),
        "dataset2_size": len(texts2),
    }
    print("\n--- Length Statistics ---")
    results["dataset1_length_stats"] = compute_length_statistics(texts1)
    results["dataset2_length_stats"] = compute_length_statistics(texts2)
    print(f"Dataset 1 mean length: {results['dataset1_length_stats']['mean_length']:.1f} tokens")
    print(f"Dataset 2 mean length: {results['dataset2_length_stats']['mean_length']:.1f} tokens")
    print("\n--- Vocabulary Overlap ---")
    vocab_stats = compute_vocabulary_overlap(texts1, texts2)
    results["vocabulary"] = vocab_stats
    print(f"Jaccard Similarity: {vocab_stats['jaccard_similarity']:.4f}")
    print(f"Vocab overlap ratio (ds1): {vocab_stats['overlap_ratio_1']:.4f}")
    print(f"Vocab overlap ratio (ds2): {vocab_stats['overlap_ratio_2']:.4f}")
    print("\n--- BLEU Scores ---")
    bleu_1_to_2 = compute_bleu_score(texts1, texts2, sample_size=sample_size)
    bleu_2_to_1 = compute_bleu_score(texts2, texts1, sample_size=sample_size)
    results["bleu_ds1_to_ds2"] = bleu_1_to_2
    results["bleu_ds2_to_ds1"] = bleu_2_to_1
    print(f"BLEU (ds1 -> ds2): {bleu_1_to_2.get('bleu_mean', 0):.4f}")
    print(f"BLEU (ds2 -> ds1): {bleu_2_to_1.get('bleu_mean', 0):.4f}")
    if ROUGE_AVAILABLE:
        print("\n--- ROUGE Scores ---")
        rouge_scores = compute_rouge_scores(texts1, texts2, sample_size=sample_size)
        results["rouge"] = rouge_scores
        print(f"ROUGE-1 F1: {rouge_scores.get('rouge1_fmeasure', 0):.4f}")
        print(f"ROUGE-2 F1: {rouge_scores.get('rouge2_fmeasure', 0):.4f}")
        print(f"ROUGE-L F1: {rouge_scores.get('rougeL_fmeasure', 0):.4f}")
    print("\n--- Self-BLEU (Diversity) ---")
    self_bleu_1 = compute_self_bleu(texts1, sample_size=min(500, len(texts1)))
    self_bleu_2 = compute_self_bleu(texts2, sample_size=min(500, len(texts2)))
    results["self_bleu_ds1"] = self_bleu_1
    results["self_bleu_ds2"] = self_bleu_2
    print(f"Self-BLEU ds1: {self_bleu_1:.4f} (lower = more diverse)")
    print(f"Self-BLEU ds2: {self_bleu_2:.4f} (lower = more diverse)")
    if output_file:
        results_converted = convert_numpy_types(results)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze similarity between two text datasets")
    parser.add_argument("--dataset1", default="seara/ru_go_emotions", help="First dataset name")
    parser.add_argument("--dataset2", default="Djacon/ru-izard-emotions", help="Second dataset name")
    parser.add_argument("--split", default="train", help="Dataset split to analyze")
    parser.add_argument("--sample-size", type=int, default=2000, help="Sample size for analysis")
    parser.add_argument("--output", default="dataset_similarity_report.json", help="Output file path")
    args = parser.parse_args()
    results = analyze_datasets_similarity(
        dataset1_name=args.dataset1,
        dataset2_name=args.dataset2,
        split=args.split,
        sample_size=args.sample_size,
        output_file=args.output
    )
