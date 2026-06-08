"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả bằng RRF (Reciprocal Rank Fusion)
    3. Rerank bằng Jina cross-encoder
    4. Nếu top result score < threshold -> fallback sang PageIndex
    5. Return top_k results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task5_semantic_search import semantic_search
from task6_lexical_search import lexical_search
from task7_reranking import rerank, rerank_rrf
from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.01  # RRF scores max ~1/60≈0.017; fallback khi khong co ket qua
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "rrf" | "mmr"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoan chinh voi fallback logic.

    Pipeline:
        Query
          |-> Semantic Search (dense)  -|
          |-> Lexical Search  (sparse) -|-> RRF Merge -> Rerank -> Results
          |
          --> If best_score < threshold -> PageIndex fallback

    Args:
        query: Cau truy van
        top_k: So luong ket qua cuoi cung
        score_threshold: Nguong diem toi thieu cho hybrid results
        use_reranking: Co ap dung reranking hay khong

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoac 'pageindex'
        }
    """
    # Step 1: Chay semantic + lexical search
    # Lay nhieu hon top_k de co candidates tot cho reranking
    try:
        dense_results = semantic_search(query, top_k=top_k * 3)
    except Exception:
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=top_k * 3)
    except Exception:
        sparse_results = []

    # Step 2: Merge bang RRF (Reciprocal Rank Fusion)
    # RRF gop rank tu ca 2 nguon - khong can normalize score
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank merged candidates bang cross-encoder
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            if "source" not in item:
                item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    # Step 4: Kiem tra threshold -> fallback PageIndex neu can
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). "
            "Fallback -> PageIndex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback
        # Neu PageIndex cung khong co -> tra ve hybrid du score thap
        return final_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Nghe si nao bi bat vi su dung ma tuy",
        "Luat phong chong ma tuy 2021 quy dinh gi ve cai nghien",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
