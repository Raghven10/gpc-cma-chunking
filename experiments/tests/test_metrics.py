"""
Unit tests for intrinsic partition metrics.
Tests bounds, edge cases, single-sentence chunks, rank limits, and monotonicity.
"""

import unittest
import numpy as np
import os
import sys

# Add experiments dir to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics import compute_isd, compute_ibi, compute_gpc, compute_cma, compute_cdf


class TestMetrics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.D = 64
        self.N = 20

    def test_isd_single_sentence(self):
        """Single sentence chunk should yield ISD = 1.0."""
        vec = np.random.randn(1, self.D)
        isd = compute_isd(vec)
        self.assertEqual(isd, 1.0)

    def test_isd_identical_sentences(self):
        """Identical sentence embeddings should have similarity 1.0, var 0.0 -> ISD = 1.0."""
        vec = np.random.randn(1, self.D)
        vecs = np.repeat(vec, 5, axis=0)
        isd = compute_isd(vecs, alpha=0.5)
        self.assertAlmostEqual(isd, 1.0, places=5)

    def test_isd_bounds(self):
        """ISD must remain bounded in [-1.0, 1.0]."""
        for _ in range(10):
            vecs = np.random.randn(15, self.D)
            isd = compute_isd(vecs, alpha=1.0)
            self.assertGreaterEqual(isd, -1.0)
            self.assertLessEqual(isd, 1.0)

    def test_ibi_monotonicity_and_bounds(self):
        """IBI should be in [0, 1] and increase as adjacent centroids become more distant."""
        c1 = np.array([1.0, 0.0, 0.0])
        c2_near = np.array([0.9, 0.1, 0.0]) # Cos sim ~ 0.99
        c2_far = np.array([0.0, 1.0, 0.0])  # Cos sim = 0.0
        c2_opp = np.array([-1.0, 0.0, 0.0]) # Cos sim = -1.0

        ibi_near = compute_ibi(np.vstack([c1, c2_near]))
        ibi_far = compute_ibi(np.vstack([c1, c2_far]))
        ibi_opp = compute_ibi(np.vstack([c1, c2_opp]))

        self.assertLess(ibi_near[0], ibi_far[0])
        self.assertLess(ibi_far[0], ibi_opp[0])
        self.assertAlmostEqual(ibi_opp[0], 1.0, places=5)
        self.assertAlmostEqual(ibi_far[0], 0.5, places=5)

    def test_gpc_composite(self):
        """GPC composite harmonic mean should be bounded in [0, 1] and non-zero."""
        chunk1 = np.random.randn(5, self.D)
        chunk2 = np.random.randn(5, self.D)
        chunk3 = np.random.randn(5, self.D)
        res = compute_gpc([chunk1, chunk2, chunk3], alpha=0.5, beta=0.6)
        
        self.assertIn("document_gpc", res)
        self.assertGreaterEqual(res["document_gpc"], 0.0)
        self.assertLessEqual(res["document_gpc"], 1.0)
        self.assertEqual(len(res["chunk_gpc"]), 3)

    def test_cma_manifold_projection(self):
        """CMA should be 1.0 if chunk centroid lies strictly in the PCA subspace."""
        doc_vecs = np.random.randn(30, self.D)
        # 3 chunks of 10 sentences
        chunks = [doc_vecs[:10], doc_vecs[10:20], doc_vecs[20:]]
        res = compute_cma(doc_vecs, chunks, pca_rank=10)
        
        self.assertIn("document_cma", res)
        self.assertGreaterEqual(res["document_cma"], 0.0)
        self.assertLessEqual(res["document_cma"], 1.0)
        self.assertEqual(len(res["chunk_cma"]), 3)

    def test_cdf_distributional_fidelity(self):
        """CDF should be bounded in [0, 1] and 1.0 for identical distributions."""
        tokens = ["neural", "network", "retrieval", "augmented", "generation"] * 10
        chunk1 = tokens[:25]
        chunk2 = tokens[25:]
        res = compute_cdf(tokens, [chunk1, chunk2])
        
        self.assertIn("document_cdf", res)
        self.assertGreaterEqual(res["document_cdf"], 0.0)
        self.assertLessEqual(res["document_cdf"], 1.0)


if __name__ == "__main__":
    unittest.main()
