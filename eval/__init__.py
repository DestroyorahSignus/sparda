"""SPARDA evaluation harness — DANTE retrieval ablation + VERGIL RAG ablation (§8)."""

from eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from eval.test_queries import TEST_QUERIES

__all__ = ["mrr_at_k", "ndcg_at_k", "recall_at_k", "TEST_QUERIES"]
