"""
Experiment B: Predictive Validity of Intrinsic Metrics on Long-Document Evidence QA.
Primary Corpus: Qasper dataset with verified multi-sentence evidence spans.
Evaluates:
- Retrieval methods: Dense (SentenceTransformer) and Sparse (BM25).
- Matched retrieved-token budgets: 256, 512, 1024 tokens.
- Metrics: Evidence Recall@Budget, Evidence Coverage (Hit), MRR, nDCG@Budget, Relevant-Context Fraction.
- Intrinsic-Extrinsic Predictive Validity: Query-condition paired bootstrap CIs,
  repeated-measures / clustered Pearson & Spearman correlation between GPC/CMA and Evidence Recall.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from scipy import stats

from metrics import compute_isd, compute_ibi, compute_gpc, compute_cma, compute_cdf
from chunking_policies import (
    Document,
    GoldSectionChunker,
    FixedWindowChunker,
    RecursiveChunker,
    SemanticThresholdChunker,
    simple_tokenize
)
from dataset_loader import load_qasper_dataset


def bootstrap_ci(data: np.ndarray, num_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for mean."""
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.RandomState(42)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(num_bootstraps)]
    low_pct = (1.0 - ci) / 2.0 * 100
    high_pct = (1.0 + ci) / 2.0 * 100
    return float(np.percentile(boot_means, low_pct)), float(np.percentile(boot_means, high_pct))


def check_evidence_overlap(chunk_text: str, evidence_spans: List[str]) -> Tuple[float, bool]:
    """
    Computes token-level overlap and exact containment between chunk text and evidence spans.
    Returns:
        overlap_fraction: Fraction of evidence words covered in chunk text.
        exact_hit: Whether at least one complete evidence span is substantially contained.
    """
    chunk_tokens = set(simple_tokenize(chunk_text))
    if not chunk_tokens or not evidence_spans:
        return 0.0, False
        
    total_ev_tokens = set()
    exact_hit = False
    
    for ev in evidence_spans:
        ev_tokens = set(simple_tokenize(ev))
        total_ev_tokens.update(ev_tokens)
        # Check if 80%+ of this evidence span's tokens are inside chunk
        if len(ev_tokens) > 0 and len(ev_tokens.intersection(chunk_tokens)) / len(ev_tokens) >= 0.8:
            exact_hit = True
            
    if not total_ev_tokens:
        return 0.0, False
        
    overlap_frac = len(total_ev_tokens.intersection(chunk_tokens)) / len(total_ev_tokens)
    return float(overlap_frac), exact_hit


def run_predictive_validity_experiment(
    max_docs: int = 50,
    dense_model_name: str = "all-MiniLM-L6-v2",
    token_budgets: List[int] = [256, 512, 1024]
) -> Dict[str, Any]:
    print("=" * 70)
    print(f"RUNNING PREDICTIVE VALIDITY EXPERIMENT ON QASPER (Dense: {dense_model_name})")
    print("=" * 70)
    
    os.makedirs("experiments/results", exist_ok=True)
    
    # 1. Load Qasper dev dataset
    docs, qas = load_qasper_dataset(split="dev", max_docs=max_docs)
    doc_map = {d.doc_id: d for d in docs}
    
    # Filter QAs for loaded docs
    active_qas = [q for q in qas if q["doc_id"] in doc_map]
    print(f"Active documents: {len(docs)}, Active QA queries: {len(active_qas)}")
    
    # 2. Load dense encoder
    encoder = SentenceTransformer(dense_model_name)
    
    # Policies to compare
    policies = {
        "Fixed-128": FixedWindowChunker(target_tokens=128),
        "Fixed-256": FixedWindowChunker(target_tokens=256),
        "Fixed-512": FixedWindowChunker(target_tokens=512),
        "Recursive-256": RecursiveChunker(target_tokens=256),
        "Semantic-75": SemanticThresholdChunker(percentile_threshold=75.0),
        "GoldSection": GoldSectionChunker()
    }
    
    # Pre-partition documents and compute intrinsic metrics
    doc_partitions = {} # (doc_id, policy_name) -> {chunks, chunk_texts, chunk_embs, gpc, cma, cdf}
    
    print("\nPartitioning documents and pre-indexing chunks...")
    for doc in tqdm(docs, desc="Indexing Documents"):
        sent_embs = encoder.encode(doc.sentences, show_progress_bar=False, normalize_embeddings=True)
        doc_tokens = doc.tokens
        
        for pol_name, chunker in policies.items():
            if pol_name == "Semantic-75":
                chunks = chunker.partition(doc, sentence_embeddings=sent_embs)
            else:
                chunks = chunker.partition(doc)
                
            chunk_texts = [" ".join([doc.sentences[idx] for idx in c_idx_list]) for c_idx_list in chunks if len(c_idx_list) > 0]
            chunk_token_lists = [simple_tokenize(txt) for txt in chunk_texts]
            chunk_sent_embs_list = [sent_embs[c_idx_list] for c_idx_list in chunks if len(c_idx_list) > 0]
            
            # Compute chunk embeddings as centroid
            chunk_embs = np.array([np.mean(embs, axis=0) for embs in chunk_sent_embs_list])
            # Normalize
            norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            chunk_embs = chunk_embs / norms
            
            # Intrinsic metrics
            gpc_res = compute_gpc(chunk_sent_embs_list)
            cma_res = compute_cma(sent_embs, chunk_sent_embs_list)
            cdf_res = compute_cdf(doc_tokens, chunk_token_lists)
            
            # BM25 index
            bm25_index = BM25Okapi(chunk_token_lists)
            
            doc_partitions[(doc.doc_id, pol_name)] = {
                "chunks": chunks,
                "chunk_texts": chunk_texts,
                "chunk_token_lists": chunk_token_lists,
                "chunk_embs": chunk_embs,
                "bm25_index": bm25_index,
                "gpc": gpc_res["document_gpc"],
                "cma": cma_res["document_cma"],
                "cdf": cdf_res["document_cdf"],
                "mean_isd": gpc_res["mean_isd"],
                "mean_ibi": gpc_res["mean_ibi"],
                "num_chunks": len(chunks),
                "total_tokens": sum(len(tl) for tl in chunk_token_lists)
            }
            
    # Query evaluation records
    retrieval_records = []
    
    print("\nEvaluating evidence retrieval under matched token budgets...")
    for qa in tqdm(active_qas, desc="QA Queries"):
        q_id = qa["q_id"]
        doc_id = qa["doc_id"]
        q_text = qa["question"]
        evidence_spans = qa["evidence_spans"]
        doc = doc_map[doc_id]
        
        q_tokens = simple_tokenize(q_text)
        q_emb = encoder.encode(q_text, show_progress_bar=False, normalize_embeddings=True)
        
        for pol_name in policies.keys():
            part = doc_partitions[(doc_id, pol_name)]
            chunk_texts = part["chunk_texts"]
            chunk_token_lists = part["chunk_token_lists"]
            chunk_embs = part["chunk_embs"]
            bm25 = part["bm25_index"]
            
            # 1. Dense retrieval scores
            dense_scores = np.dot(chunk_embs, q_emb)
            dense_rank_indices = np.argsort(-dense_scores)
            
            # 2. Sparse retrieval scores (BM25)
            sparse_scores = np.array(bm25.get_scores(q_tokens))
            sparse_rank_indices = np.argsort(-sparse_scores)
            
            # Evaluate across retrievers and budgets
            for ret_type, ranked_idx in [("Dense", dense_rank_indices), ("BM25", sparse_rank_indices)]:
                for budget in token_budgets:
                    # Greedily take top chunks until token budget reached
                    retrieved_chunks = []
                    retrieved_tokens_count = 0
                    mrr = 0.0
                    ndcg = 0.0
                    hit = False
                    covered_ev_tokens = set()
                    total_ev_tokens = set()
                    for ev in evidence_spans:
                        total_ev_tokens.update(simple_tokenize(ev))
                        
                    for rank_pos, c_idx in enumerate(ranked_idx):
                        c_tok_cnt = len(chunk_token_lists[c_idx])
                        if retrieved_tokens_count + c_tok_cnt > budget and retrieved_chunks:
                            break
                            
                        retrieved_chunks.append(c_idx)
                        retrieved_tokens_count += c_tok_cnt
                        c_txt = chunk_texts[c_idx]
                        
                        overlap, is_hit = check_evidence_overlap(c_txt, evidence_spans)
                        c_tokens_set = set(simple_tokenize(c_txt))
                        covered_ev_tokens.update(total_ev_tokens.intersection(c_tokens_set))
                        
                        if is_hit and not hit:
                            hit = True
                            mrr = 1.0 / (rank_pos + 1)
                            ndcg = 1.0 / np.log2(rank_pos + 2)
                            
                    ev_recall = len(covered_ev_tokens) / len(total_ev_tokens) if total_ev_tokens else 0.0
                    
                    retrieval_records.append({
                        "q_id": q_id,
                        "doc_id": doc_id,
                        "policy": pol_name,
                        "retriever": ret_type,
                        "budget": budget,
                        "retrieved_chunks_count": len(retrieved_chunks),
                        "retrieved_tokens": retrieved_tokens_count,
                        "evidence_recall": float(ev_recall),
                        "evidence_coverage": 1.0 if hit else 0.0,
                        "mrr": float(mrr),
                        "ndcg": float(ndcg),
                        "gpc": part["gpc"],
                        "cma": part["cma"],
                        "cdf": part["cdf"],
                        "mean_isd": part["mean_isd"],
                        "mean_ibi": part["mean_ibi"]
                    })
                    
    df_ret = pd.DataFrame(retrieval_records)
    df_ret.to_csv("experiments/results/predictive_validity_raw.csv", index=False)
    print(f"\nSaved raw retrieval results to experiments/results/predictive_validity_raw.csv")
    
    # Table 3: Summary Table by Policy and Retriever at Budget = 512 tokens
    t3_rows = []
    for pol_name in policies.keys():
        for ret_type in ["Dense", "BM25"]:
            sub = df_ret[(df_ret["policy"] == pol_name) & (df_ret["retriever"] == ret_type) & (df_ret["budget"] == 512)]
            
            rec_vals = sub["evidence_recall"].values
            cov_vals = sub["evidence_coverage"].values
            mrr_vals = sub["mrr"].values
            ndcg_vals = sub["ndcg"].values
            
            rec_ci = bootstrap_ci(rec_vals)
            cov_ci = bootstrap_ci(cov_vals)
            
            t3_rows.append({
                "Policy": pol_name,
                "Retriever": ret_type,
                "Budget": 512,
                "Recall@512": float(np.mean(rec_vals)),
                "Recall_95CI": f"[{rec_ci[0]:.3f}, {rec_ci[1]:.3f}]",
                "Coverage@512": float(np.mean(cov_vals)),
                "Coverage_95CI": f"[{cov_ci[0]:.3f}, {cov_ci[1]:.3f}]",
                "MRR@512": float(np.mean(mrr_vals)),
                "nDCG@512": float(np.mean(ndcg_vals)),
                "Mean_GPC": float(np.mean(sub["gpc"].values)),
                "Mean_CMA": float(np.mean(sub["cma"].values))
            })
            
    t3_df = pd.DataFrame(t3_rows)
    t3_df.to_csv("experiments/results/table3_qa_retrieval_summary.csv", index=False)
    print("\nTable 3: QA Evidence Retrieval Benchmark (Budget = 512 Tokens):")
    print(t3_df[["Policy", "Retriever", "Recall@512", "Coverage@512", "MRR@512", "Mean_GPC", "Mean_CMA"]].to_string(index=False))
    
    # Table 4: Predictive Validity Correlation Analysis (Query & Document-level, Budget = 512)
    t4_rows = []
    for ret_type in ["Dense", "BM25"]:
        sub = df_ret[(df_ret["retriever"] == ret_type) & (df_ret["budget"] == 512)]
        
        # Aggregate at doc-policy level to account for query clustering
        doc_pol = sub.groupby(["doc_id", "policy"]).agg({
            "evidence_recall": "mean",
            "evidence_coverage": "mean",
            "mrr": "mean",
            "gpc": "first",
            "cma": "first",
            "cdf": "first"
        }).reset_index()
        
        for metric_name in ["gpc", "cma", "cdf"]:
            x = doc_pol[metric_name].values
            y_rec = doc_pol["evidence_recall"].values
            y_cov = doc_pol["evidence_coverage"].values
            
            p_rec, p_rec_pval = stats.pearsonr(x, y_rec)
            s_rec, s_rec_pval = stats.spearmanr(x, y_rec)
            
            p_cov, p_cov_pval = stats.pearsonr(x, y_cov)
            s_cov, s_cov_pval = stats.spearmanr(x, y_cov)
            
            # Bootstrap CI for correlation
            rng = np.random.RandomState(42)
            boot_spearmans = []
            for _ in range(1000):
                boot_idx = rng.choice(len(x), size=len(x), replace=True)
                boot_s, _ = stats.spearmanr(x[boot_idx], y_rec[boot_idx])
                if not np.isnan(boot_s):
                    boot_spearmans.append(boot_s)
            spearman_ci = (float(np.percentile(boot_spearmans, 2.5)), float(np.percentile(boot_spearmans, 97.5)))
            
            t4_rows.append({
                "Retriever": ret_type,
                "Intrinsic_Metric": metric_name.upper(),
                "Target_Outcome": "Evidence_Recall",
                "Pearson_r": float(p_rec),
                "Pearson_p": float(p_rec_pval),
                "Spearman_rho": float(s_rec),
                "Spearman_p": float(s_rec_pval),
                "Spearman_95CI": f"[{spearman_ci[0]:.3f}, {spearman_ci[1]:.3f}]"
            })
            
    t4_df = pd.DataFrame(t4_rows)
    t4_df.to_csv("experiments/results/table4_predictive_validity_correlations.csv", index=False)
    print("\nTable 4: Predictive Validity (Correlation with Evidence Recall at Document-Condition Level):")
    print(t4_df[["Retriever", "Intrinsic_Metric", "Pearson_r", "Pearson_p", "Spearman_rho", "Spearman_p", "Spearman_95CI"]].to_string(index=False))
    
    return {
        "df_ret": df_ret,
        "table3": t3_df,
        "table4": t4_df
    }


if __name__ == "__main__":
    run_predictive_validity_experiment()
