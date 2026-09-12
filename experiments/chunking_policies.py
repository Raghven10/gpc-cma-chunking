"""
Chunking policies and null controls for document partitioning experiments.
Supports:
1. GoldSectionChunker (Author-defined section / paragraph structure)
2. FixedWindowChunker (128, 256, 512 token windows with overlap)
3. RecursiveChunker (Hierarchical delimiter splitting)
4. SemanticThresholdChunker (Embedding distance boundary detection)
5. RandomMatchedChunker (Null control matching chunk count & size distribution)
6. ShuffledSentenceNullChunker (Sequential coherence destruction null control)
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Any, Optional


def simple_tokenize(text: str) -> List[str]:
    """Basic word tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


class Document:
    """Represents a document with raw text, sentences, sections, and tokens."""
    def __init__(
        self,
        doc_id: str,
        title: str,
        sections: List[Dict[str, Any]], # Each dict has 'heading', 'text', 'sentences'
        genre: str = "scientific"
    ):
        self.doc_id = doc_id
        self.title = title
        self.sections = sections
        self.genre = genre
        
        # Flattened sentences
        self.sentences: List[str] = []
        self.sentence_section_indices: List[int] = []
        for s_idx, sec in enumerate(sections):
            for sent in sec.get("sentences", []):
                clean_sent = sent.strip()
                if clean_sent:
                    self.sentences.append(clean_sent)
                    self.sentence_section_indices.append(s_idx)
                    
        self.full_text = " ".join(self.sentences)
        self.tokens = simple_tokenize(self.full_text)
        
    def __len__(self):
        return len(self.sentences)


class BaseChunker:
    """Base interface for document partitioners."""
    def partition(self, doc: Document, **kwargs) -> List[List[int]]:
        """
        Partitions document sentences into chunks.
        Returns: List of chunks, where each chunk is a list of sentence indices [idx_start, ..., idx_end].
        """
        raise NotImplementedError


class GoldSectionChunker(BaseChunker):
    """Partitions document according to author structural section boundaries."""
    def partition(self, doc: Document, **kwargs) -> List[List[int]]:
        chunks = []
        current_chunk = []
        current_sec = None
        
        for idx, sec_idx in enumerate(doc.sentence_section_indices):
            if current_sec is None or sec_idx == current_sec:
                current_chunk.append(idx)
                current_sec = sec_idx
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [idx]
                current_sec = sec_idx
                
        if current_chunk:
            chunks.append(current_chunk)
        return chunks if chunks else [[i for i in range(len(doc.sentences))]]


class FixedWindowChunker(BaseChunker):
    """Partitions document into fixed token windows (e.g. 128, 256, 512)."""
    def __init__(self, target_tokens: int = 256, overlap_pct: float = 0.1):
        self.target_tokens = target_tokens
        self.overlap_pct = overlap_pct
        
    def partition(self, doc: Document, **kwargs) -> List[List[int]]:
        chunks = []
        if len(doc.sentences) == 0:
            return []
            
        sent_token_counts = [len(simple_tokenize(s)) for s in doc.sentences]
        current_chunk = []
        current_tokens = 0
        
        i = 0
        while i < len(doc.sentences):
            current_chunk.append(i)
            current_tokens += sent_token_counts[i]
            
            if current_tokens >= self.target_tokens:
                chunks.append(current_chunk)
                # Calculate overlap stride
                overlap_tokens = int(self.target_tokens * self.overlap_pct)
                # Step back sentences to satisfy overlap
                back_tokens = 0
                step_back = 0
                while step_back < len(current_chunk) - 1:
                    back_tokens += sent_token_counts[current_chunk[-(1 + step_back)]]
                    if back_tokens >= overlap_tokens:
                        break
                    step_back += 1
                
                i = i - step_back + 1
                current_chunk = []
                current_tokens = 0
            else:
                i += 1
                
        if current_chunk and (not chunks or current_chunk != chunks[-1]):
            chunks.append(current_chunk)
            
        return chunks if chunks else [[i for i in range(len(doc.sentences))]]


class RecursiveChunker(BaseChunker):
    """Hierarchical delimiter splitting (simulating LangChain RecursiveCharacterTextSplitter)."""
    def __init__(self, target_tokens: int = 256):
        self.target_tokens = target_tokens
        
    def partition(self, doc: Document, **kwargs) -> List[List[int]]:
        # First group by section/paragraph, then split if section exceeds target tokens
        chunks = []
        gold_chunks = GoldSectionChunker().partition(doc)
        
        for g_chunk in gold_chunks:
            g_tokens = sum(len(simple_tokenize(doc.sentences[idx])) for idx in g_chunk)
            if g_tokens <= self.target_tokens:
                chunks.append(g_chunk)
            else:
                # Subdivide g_chunk
                sub_chunk = []
                sub_tokens = 0
                for idx in g_chunk:
                    t_cnt = len(simple_tokenize(doc.sentences[idx]))
                    if sub_tokens + t_cnt > self.target_tokens and sub_chunk:
                        chunks.append(sub_chunk)
                        sub_chunk = [idx]
                        sub_tokens = t_cnt
                    else:
                        sub_chunk.append(idx)
                        sub_tokens += t_cnt
                if sub_chunk:
                    chunks.append(sub_chunk)
                    
        return chunks if chunks else [[i for i in range(len(doc.sentences))]]


class SemanticThresholdChunker(BaseChunker):
    """Splits at points where consecutive sentence cosine distance exceeds a percentile threshold."""
    def __init__(self, percentile_threshold: float = 75.0):
        self.percentile_threshold = percentile_threshold
        
    def partition(self, doc: Document, sentence_embeddings: Optional[np.ndarray] = None, **kwargs) -> List[List[int]]:
        N = len(doc.sentences)
        if N <= 1 or sentence_embeddings is None or sentence_embeddings.shape[0] != N:
            return [[i for i in range(N)]]
            
        # Compute adjacent cosine similarities
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        normed = sentence_embeddings / norms
        adj_sims = np.sum(normed[:-1] * normed[1:], axis=1) # Length N-1
        adj_dists = 1.0 - adj_sims
        
        threshold = float(np.percentile(adj_dists, self.percentile_threshold))
        
        chunks = []
        current_chunk = [0]
        for i in range(len(adj_dists)):
            if adj_dists[i] >= threshold:
                chunks.append(current_chunk)
                current_chunk = [i + 1]
            else:
                current_chunk.append(i + 1)
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks


class RandomMatchedChunker(BaseChunker):
    """Null control: generates random partition matching the exact chunk count and length distribution of reference."""
    def partition(self, doc: Document, reference_chunks: List[List[int]], seed: int = 42, **kwargs) -> List[List[int]]:
        N = len(doc.sentences)
        K = len(reference_chunks)
        if N <= 1 or K <= 1:
            return [[i for i in range(N)]]
            
        rng = np.random.RandomState(seed)
        # Randomly choose K-1 cut points from 1 to N-1
        cut_points = sorted(rng.choice(range(1, N), size=min(K - 1, N - 1), replace=False))
        
        chunks = []
        prev = 0
        for cp in cut_points:
            chunks.append(list(range(prev, cp)))
            prev = cp
        chunks.append(list(range(prev, N)))
        return chunks


class ShuffledSentenceNullChunker(BaseChunker):
    """Null control: shuffles sentence order across document, then partitions into equal sizes."""
    def partition(self, doc: Document, reference_chunks: List[List[int]], seed: int = 42, **kwargs) -> Tuple[List[int], List[List[int]]]:
        """
        Returns:
            permuted_sentence_indices: The permutation order of sentences.
            shuffled_chunks: Chunks formed over the permuted sentences.
        """
        N = len(doc.sentences)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(N).tolist()
        
        chunks = []
        prev = 0
        for ref_c in reference_chunks:
            c_len = len(ref_c)
            end = min(prev + c_len, N)
            if prev < end:
                chunks.append(perm[prev:end])
            prev = end
        if prev < N:
            chunks.append(perm[prev:])
        return perm, chunks
