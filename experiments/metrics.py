"""
Intrinsic evaluation metrics for document chunk partitions.
Implements:
1. Intra-Chunk Semantic Density (ISD)
2. Inter-Chunk Boundary Isolation (IBI)
3. Composite Geometric Partition Cohesion (GPC)
4. Contextual Manifold Alignment (CMA)
5. Contextual Distributional Fidelity (CDF)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities for unit-normalized vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    normalized = vectors / norms
    return np.clip(np.dot(normalized, normalized.T), -1.0, 1.0)


def compute_isd(sentence_embeddings: np.ndarray, alpha: float = 0.5) -> float:
    """
    Intra-Chunk Semantic Density (ISD).
    
    Args:
        sentence_embeddings: Array of shape (N, D) containing sentence embeddings for a chunk.
        alpha: Variance penalty coefficient (default 0.5).
        
    Returns:
        ISD score in [-1, 1], where 1 represents maximal density and uniform topical focus.
        For N = 1, returns 1.0 (single sentence is maximally self-coherent with zero internal variance).
    """
    N = sentence_embeddings.shape[0]
    if N <= 1:
        return 1.0
    
    sim_matrix = cosine_similarity_matrix(sentence_embeddings)
    # Extract strictly upper triangle pairs (j < k)
    triu_indices = np.triu_indices(N, k=1)
    pairwise_sims = sim_matrix[triu_indices]
    
    mean_sim = float(np.mean(pairwise_sims))
    var_sim = float(np.var(pairwise_sims))
    
    # Penalize internal semantic divergence / variance
    isd = mean_sim - alpha * var_sim
    return float(np.clip(isd, -1.0, 1.0))


def compute_ibi(chunk_centroids: np.ndarray) -> List[float]:
    """
    Inter-Chunk Boundary Isolation (IBI).
    
    Measures the separation between adjacent chunk centroids.
    Defined as normalized angular/cosine distance: d(c_i, c_adj) = (1 - cos(c_i, c_adj)) / 2 in [0, 1].
    
    Args:
        chunk_centroids: Array of shape (K, D) containing centroid embeddings for K chunks.
        
    Returns:
        List of K IBI scores in [0, 1]. High score indicates sharp boundary isolation.
    """
    K = chunk_centroids.shape[0]
    if K <= 1:
        return [1.0]
    
    # Normalize centroids
    norms = np.linalg.norm(chunk_centroids, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    norm_centroids = chunk_centroids / norms
    
    # Compute adjacent cosine similarities
    adj_sims = np.sum(norm_centroids[:-1] * norm_centroids[1:], axis=1) # Length K-1
    adj_sims = np.clip(adj_sims, -1.0, 1.0)
    
    # Distance: (1 - cos) / 2 in [0, 1]
    adj_dists = (1.0 - adj_sims) / 2.0
    
    ibi_scores = []
    for i in range(K):
        if i == 0:
            # First chunk only has right neighbor
            ibi_scores.append(float(adj_dists[0]))
        elif i == K - 1:
            # Last chunk only has left neighbor
            ibi_scores.append(float(adj_dists[-1]))
        else:
            # Interior chunk: minimum isolation from either adjacent neighbor
            ibi_scores.append(float(min(adj_dists[i-1], adj_dists[i])))
            
    return ibi_scores


def compute_gpc(
    chunk_sentence_embeddings: List[np.ndarray],
    alpha: float = 0.5,
    beta: float = 0.6,
    eps: float = 1e-6
) -> Dict[str, any]:
    """
    Composite Geometric Partition Cohesion (GPC).
    
    Combines raw ISD and normalized IBI.
    Uses smoothed harmonic mean across chunks with epsilon regularizer to penalize outlier partitions.
    
    Args:
        chunk_sentence_embeddings: List of (N_i, D) arrays for each chunk C_i.
        alpha: ISD variance penalty.
        beta: Weight balancing internal density vs boundary isolation (default 0.6).
        eps: Epsilon smoothing to prevent division by zero in harmonic mean.
        
    Returns:
        Dict with chunk_isd, chunk_ibi, chunk_gpc, and document_gpc.
    """
    K = len(chunk_sentence_embeddings)
    if K == 0:
        return {"document_gpc": 0.0, "chunk_gpc": [], "chunk_isd": [], "chunk_ibi": []}
    
    # 1. Compute ISD per chunk
    isd_scores = [compute_isd(c, alpha=alpha) for c in chunk_sentence_embeddings]
    # Map ISD from [-1, 1] to [0, 1] for composite formulation: (ISD + 1) / 2
    norm_isd = [(score + 1.0) / 2.0 for score in isd_scores]
    
    # 2. Compute chunk centroids
    centroids = np.array([np.mean(c, axis=0) for c in chunk_sentence_embeddings])
    ibi_scores = compute_ibi(centroids) # Already in [0, 1]
    
    # 3. Chunk-level GPC
    chunk_gpc = [
        float(np.clip(beta * norm_isd[i] + (1.0 - beta) * ibi_scores[i], 0.0, 1.0))
        for i in range(K)
    ]
    
    # 4. Document-level GPC via smoothed harmonic mean
    # H = K / sum(1 / (GPC_i + eps))
    inv_sum = sum(1.0 / (score + eps) for score in chunk_gpc)
    doc_gpc = float(K / inv_sum) if inv_sum > 0 else 0.0
    
    return {
        "document_gpc": float(np.clip(doc_gpc, 0.0, 1.0)),
        "mean_isd": float(np.mean(isd_scores)),
        "mean_ibi": float(np.mean(ibi_scores)),
        "chunk_isd": isd_scores,
        "chunk_ibi": ibi_scores,
        "chunk_gpc": chunk_gpc
    }


def compute_cma(
    all_sentence_embeddings: np.ndarray,
    chunk_sentence_embeddings: List[np.ndarray],
    pca_rank: int = 10
) -> Dict[str, any]:
    """
    Contextual Manifold Alignment (CMA).
    
    Spans the document manifold M via centered PCA on all sentence embeddings in document D.
    Measures the fraction of chunk centroid energy retained under projection onto M.
    
    Args:
        all_sentence_embeddings: Array of shape (Total_N, D) for the entire document.
        chunk_sentence_embeddings: List of (N_i, D) arrays for each chunk.
        pca_rank: Number of principal components retained (default 10).
        
    Returns:
        Dict with chunk_cma and document_cma (mean over chunks).
    """
    N_total, D = all_sentence_embeddings.shape
    K = len(chunk_sentence_embeddings)
    if N_total <= 1 or K == 0:
        return {"document_cma": 1.0, "chunk_cma": [1.0] * K}
    
    # Determine valid PCA rank
    effective_rank = min(pca_rank, N_total - 1, D)
    if effective_rank < 1:
        return {"document_cma": 1.0, "chunk_cma": [1.0] * K}
    
    # Fit centered PCA
    pca = PCA(n_components=effective_rank)
    pca.fit(all_sentence_embeddings)
    components = pca.components_ # Shape (k, D)
    doc_mean = pca.mean_ # Shape (D,)
    
    chunk_cma = []
    for c in chunk_sentence_embeddings:
        centroid = np.mean(c, axis=0)
        centered_centroid = centroid - doc_mean
        norm_orig = np.linalg.norm(centered_centroid)
        
        if norm_orig < 1e-9:
            # Centroid coincides with document center; fully aligned with zero residual
            chunk_cma.append(1.0)
        else:
            # Project onto subspace spanned by components: P_M = V V^T (since V rows are orthonormal)
            coords = np.dot(components, centered_centroid) # Shape (k,)
            proj_norm = np.linalg.norm(coords)
            alignment = float(np.clip(proj_norm / norm_orig, 0.0, 1.0))
            chunk_cma.append(alignment)
            
    doc_cma = float(np.mean(chunk_cma)) if chunk_cma else 0.0
    return {
        "document_cma": doc_cma,
        "chunk_cma": chunk_cma,
        "pca_rank": effective_rank,
        "explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_))
    }


def compute_cdf(
    doc_tokens: List[str],
    chunk_token_lists: List[List[str]],
    vocab: Optional[Dict[str, int]] = None
) -> Dict[str, any]:
    """
    Contextual Distributional Fidelity (CDF).
    
    Measures information preservation via 1 - Jensen-Shannon Divergence (JSD)
    between the token unigram distribution of the document and each chunk.
    
    Args:
        doc_tokens: List of tokens in document D.
        chunk_token_lists: List of token lists for each chunk C_i.
        vocab: Optional precomputed vocabulary.
        
    Returns:
        Dict with chunk_cdf and document_cdf in [0, 1].
    """
    K = len(chunk_token_lists)
    if len(doc_tokens) == 0 or K == 0:
        return {"document_cdf": 1.0, "chunk_cdf": [1.0] * K}
    
    if vocab is None:
        unique_tokens = list(set(doc_tokens))
        vocab = {w: i for i, w in enumerate(unique_tokens)}
    
    V = len(vocab)
    if V <= 1:
        return {"document_cdf": 1.0, "chunk_cdf": [1.0] * K}
        
    # Document distribution with Laplace smoothing
    doc_counts = np.zeros(V, dtype=np.float64)
    for t in doc_tokens:
        if t in vocab:
            doc_counts[vocab[t]] += 1.0
    P = (doc_counts + 1e-4) / np.sum(doc_counts + 1e-4 * V)
    
    chunk_cdf = []
    for c_tokens in chunk_token_lists:
        c_counts = np.zeros(V, dtype=np.float64)
        for t in c_tokens:
            if t in vocab:
                c_counts[vocab[t]] += 1.0
        Q = (c_counts + 1e-4) / np.sum(c_counts + 1e-4 * V)
        
        # JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q)
        M = 0.5 * (P + Q)
        kl_pm = np.sum(P * np.log2(P / M))
        kl_qm = np.sum(Q * np.log2(Q / M))
        jsd = 0.5 * (kl_pm + kl_qm) # in [0, 1] since base 2 log
        jsd = np.clip(jsd, 0.0, 1.0)
        
        # Fidelity: 1 - JSD
        chunk_cdf.append(float(1.0 - jsd))
        
    doc_cdf = float(np.mean(chunk_cdf)) if chunk_cdf else 0.0
    return {
        "document_cdf": doc_cdf,
        "chunk_cdf": chunk_cdf
    }
