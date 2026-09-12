"""
Experiment A: Construct Validity of Intrinsic Partition Metrics.
Evaluates ISD, IBI, GPC, CMA, and CDF across 8 chunking strategies / null controls
on multi-genre documents (Scientific and Encyclopedic).
Computes paired differences, effect sizes (Cohen's d), bootstrap 95% CIs, and significance tests.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy import stats

from metrics import compute_isd, compute_ibi, compute_gpc, compute_cma, compute_cdf
from chunking_policies import (
    Document,
    GoldSectionChunker,
    FixedWindowChunker,
    RecursiveChunker,
    SemanticThresholdChunker,
    RandomMatchedChunker,
    ShuffledSentenceNullChunker,
    simple_tokenize
)
from dataset_loader import load_qasper_dataset, load_wikipedia_documents


def bootstrap_ci(data: np.ndarray, num_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for mean."""
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.RandomState(42)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(num_bootstraps)]
    low_pct = (1.0 - ci) / 2.0 * 100
    high_pct = (1.0 + ci) / 2.0 * 100
    return float(np.percentile(boot_means, low_pct)), float(np.percentile(boot_means, high_pct))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size for paired samples."""
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    return float(np.mean(diff) / std_diff)


def run_construct_validity_experiment(
    max_qasper_docs: int = 50,
    evaluator_model_name: str = "all-MiniLM-L6-v2",
    pca_rank: int = 10,
    alpha: float = 0.5,
    beta: float = 0.6
) -> Dict[str, Any]:
    print("=" * 70)
    print(f"RUNNING CONSTRUCT VALIDITY EXPERIMENT (Evaluator: {evaluator_model_name})")
    print("=" * 70)
    
    os.makedirs("experiments/results", exist_ok=True)
    
    # 1. Load data
    qasper_docs, _ = load_qasper_dataset(split="dev", max_docs=max_qasper_docs)
    wiki_docs = load_wikipedia_documents(max_docs=10)
    all_docs = qasper_docs + wiki_docs
    print(f"Total documents for construct validity: {len(all_docs)} ({len(qasper_docs)} scientific, {len(wiki_docs)} encyclopedic)")
    
    # 2. Load encoder
    print(f"Loading sentence encoder: {evaluator_model_name}...")
    encoder = SentenceTransformer(evaluator_model_name)
    
    # Define chunking strategies
    strategies = {
        "GoldSection": GoldSectionChunker(),
        "Fixed-128": FixedWindowChunker(target_tokens=128),
        "Fixed-256": FixedWindowChunker(target_tokens=256),
        "Fixed-512": FixedWindowChunker(target_tokens=512),
        "Recursive-256": RecursiveChunker(target_tokens=256),
        "Semantic-75": SemanticThresholdChunker(percentile_threshold=75.0),
        "RandomMatched": RandomMatchedChunker(),
        "ShuffledNull": ShuffledSentenceNullChunker()
    }
    
    records = []
    
    print("\nEvaluating document partitions across strategies...")
    for doc in tqdm(all_docs, desc="Documents"):
        # Encode all sentences once per document
        sentence_embeddings = encoder.encode(doc.sentences, show_progress_bar=False, normalize_embeddings=True)
        doc_tokens = doc.tokens
        
        # Gold reference chunks
        gold_chunks = strategies["GoldSection"].partition(doc)
        
        for strat_name, chunker in strategies.items():
            if strat_name == "RandomMatched":
                chunks = chunker.partition(doc, reference_chunks=gold_chunks, seed=42)
                curr_sent_embs = sentence_embeddings
                curr_doc_tokens = doc_tokens
                curr_doc_sentences = doc.sentences
            elif strat_name == "ShuffledNull":
                perm, chunks = chunker.partition(doc, reference_chunks=gold_chunks, seed=42)
                curr_sent_embs = sentence_embeddings[perm]
                curr_doc_sentences = [doc.sentences[idx] for idx in perm]
                curr_doc_tokens = simple_tokenize(" ".join(curr_doc_sentences))
            elif strat_name == "Semantic-75":
                chunks = chunker.partition(doc, sentence_embeddings=sentence_embeddings)
                curr_sent_embs = sentence_embeddings
                curr_doc_tokens = doc_tokens
                curr_doc_sentences = doc.sentences
            else:
                chunks = chunker.partition(doc)
                curr_sent_embs = sentence_embeddings
                curr_doc_tokens = doc_tokens
                curr_doc_sentences = doc.sentences
                
            # Extract chunk sentence embeddings and token lists
            chunk_sent_embs_list = [curr_sent_embs[c_idx_list] for c_idx_list in chunks if len(c_idx_list) > 0]
            chunk_tokens_list = [simple_tokenize(" ".join([curr_doc_sentences[idx] for idx in c_idx_list])) for c_idx_list in chunks if len(c_idx_list) > 0]
            
            # Compute metrics
            gpc_res = compute_gpc(chunk_sent_embs_list, alpha=alpha, beta=beta)
            cma_res = compute_cma(curr_sent_embs, chunk_sent_embs_list, pca_rank=pca_rank)
            cdf_res = compute_cdf(curr_doc_tokens, chunk_tokens_list)
            
            records.append({
                "doc_id": doc.doc_id,
                "genre": doc.genre,
                "strategy": strat_name,
                "num_chunks": len(chunks),
                "avg_chunk_sentences": float(np.mean([len(c) for c in chunks])),
                "mean_isd": gpc_res["mean_isd"],
                "mean_ibi": gpc_res["mean_ibi"],
                "document_gpc": gpc_res["document_gpc"],
                "document_cma": cma_res["document_cma"],
                "document_cdf": cdf_res["document_cdf"]
            })
            
    df = pd.DataFrame(records)
    df.to_csv("experiments/results/construct_validity_raw.csv", index=False)
    print(f"\nSaved raw results to experiments/results/construct_validity_raw.csv")
    
    # Statistical analysis: Summary table and paired tests vs Null controls
    summary_rows = []
    gold_df = df[df["strategy"] == "GoldSection"].set_index("doc_id")
    rand_df = df[df["strategy"] == "RandomMatched"].set_index("doc_id")
    shuff_df = df[df["strategy"] == "ShuffledNull"].set_index("doc_id")
    
    for strat in strategies.keys():
        sub_df = df[df["strategy"] == strat].set_index("doc_id")
        
        isd_vals = sub_df["mean_isd"].values
        ibi_vals = sub_df["mean_ibi"].values
        gpc_vals = sub_df["document_gpc"].values
        cma_vals = sub_df["document_cma"].values
        cdf_vals = sub_df["document_cdf"].values
        
        isd_ci = bootstrap_ci(isd_vals)
        ibi_ci = bootstrap_ci(ibi_vals)
        gpc_ci = bootstrap_ci(gpc_vals)
        cma_ci = bootstrap_ci(cma_vals)
        cdf_ci = bootstrap_ci(cdf_vals)
        
        summary_rows.append({
            "Strategy": strat,
            "ISD_Mean": float(np.mean(isd_vals)),
            "ISD_95CI": f"[{isd_ci[0]:.3f}, {isd_ci[1]:.3f}]",
            "IBI_Mean": float(np.mean(ibi_vals)),
            "IBI_95CI": f"[{ibi_ci[0]:.3f}, {ibi_ci[1]:.3f}]",
            "GPC_Mean": float(np.mean(gpc_vals)),
            "GPC_95CI": f"[{gpc_ci[0]:.3f}, {gpc_ci[1]:.3f}]",
            "CMA_Mean": float(np.mean(cma_vals)),
            "CMA_95CI": f"[{cma_ci[0]:.3f}, {cma_ci[1]:.3f}]",
            "CDF_Mean": float(np.mean(cdf_vals)),
            "CDF_95CI": f"[{cdf_ci[0]:.3f}, {cdf_ci[1]:.3f}]",
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("experiments/results/table2_construct_validity_summary.csv", index=False)
    print("\nTable 2: Construct Validity Summary:")
    print(summary_df[["Strategy", "ISD_Mean", "IBI_Mean", "GPC_Mean", "CMA_Mean", "CDF_Mean"]].to_string(index=False))
    
    # Paired hypothesis tests: Gold vs RandomMatched and Gold vs ShuffledNull
    paired_tests = []
    for metric_col in ["mean_isd", "mean_ibi", "document_gpc", "document_cma", "document_cdf"]:
        # Gold vs Random
        g_vals = gold_df[metric_col].values
        r_vals = rand_df[metric_col].values
        s_vals = shuff_df[metric_col].values
        
        # Paired diffs
        diff_r = g_vals - r_vals
        diff_s = g_vals - s_vals
        
        ttest_r = stats.ttest_rel(g_vals, r_vals)
        wilcox_r = stats.wilcoxon(diff_r)
        d_r = cohens_d(g_vals, r_vals)
        ci_r = bootstrap_ci(diff_r)
        
        ttest_s = stats.ttest_rel(g_vals, s_vals)
        wilcox_s = stats.wilcoxon(diff_s)
        d_s = cohens_d(g_vals, s_vals)
        ci_s = bootstrap_ci(diff_s)
        
        paired_tests.append({
            "Metric": metric_col,
            "Comparison": "Gold vs RandomMatched",
            "Mean_Delta": float(np.mean(diff_r)),
            "Delta_95CI": f"[{ci_r[0]:.4f}, {ci_r[1]:.4f}]",
            "Cohens_d": float(d_r),
            "t_stat": float(ttest_r.statistic),
            "p_val_paired_t": float(ttest_r.pvalue),
            "p_val_wilcoxon": float(wilcox_r.pvalue)
        })
        paired_tests.append({
            "Metric": metric_col,
            "Comparison": "Gold vs ShuffledNull",
            "Mean_Delta": float(np.mean(diff_s)),
            "Delta_95CI": f"[{ci_s[0]:.4f}, {ci_s[1]:.4f}]",
            "Cohens_d": float(d_s),
            "t_stat": float(ttest_s.statistic),
            "p_val_paired_t": float(ttest_s.pvalue),
            "p_val_wilcoxon": float(wilcox_s.pvalue)
        })
        
    paired_df = pd.DataFrame(paired_tests)
    paired_df.to_csv("experiments/results/table2_paired_hypothesis_tests.csv", index=False)
    print("\nPaired Hypothesis Tests (Gold vs Null Controls):")
    print(paired_df[["Metric", "Comparison", "Mean_Delta", "Delta_95CI", "Cohens_d", "p_val_paired_t"]].to_string(index=False))
    
    return {
        "raw_df": df,
        "summary_df": summary_df,
        "paired_df": paired_df
    }


if __name__ == "__main__":
    run_construct_validity_experiment()
