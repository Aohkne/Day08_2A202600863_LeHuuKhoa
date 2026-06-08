"""
Task 7 — Reranking Module.

Phương pháp sử dụng: Cross-encoder Jina Reranker v2 (multilingual)

Cơ chế Cross-encoder:
    - Khác với bi-encoder (embed query và doc riêng), cross-encoder nhận
      cặp (query, document) cùng lúc và tính relevance score trực tiếp.
    - Chính xác hơn bi-encoder vì query và doc tương tác với nhau qua
      attention layers — nhưng chậm hơn (không cache được embedding).
    - Jina Reranker v2 là multilingual cross-encoder, hỗ trợ tiếng Việt.
    - Dùng ở cuối pipeline (sau semantic + lexical search) để chọn top_k
      chính xác nhất từ ~20 candidates thô.

Cài đặt:
    pip install requests python-dotenv
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_MODEL = "jina-reranker-v2-base-multilingual"


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates dùng Jina Reranker v2 cross-encoder (multilingual).

    Jina Reranker v2 hoạt động thế nào:
        - Nhận cặp (query, document) và trả về relevance score [0, 1]
        - Dùng transformer cross-attention → chính xác hơn cosine similarity
        - Multilingual: hỗ trợ tiếng Việt tốt

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, sorted by rerank_score descending.
        Mỗi item có thêm key 'rerank_score' (score từ Jina).
    """
    if not candidates:
        return []

    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY không tìm thấy trong .env")

    response = requests.post(
        JINA_RERANK_URL,
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_MODEL,
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()

    reranked = response.json()["results"]

    results = []
    for r in reranked:
        item = candidates[r["index"]].copy()
        item["score"] = round(r["relevance_score"], 4)
        results.append(item)

    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    Cơ chế RRF:
        RRF(d) = Σ_r  1 / (k + rank_r(d))
        - k=60: smoothing constant (paper Cormack et al. 2009)
        - Document xuất hiện ở rank 1 nhiều list → score cao
        - Không cần normalize score giữa các ranker → robust

    Args:
        ranked_lists: Nhiều ranked lists (mỗi list từ 1 ranker: semantic, lexical, ...)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    Cơ chế MMR:
        MMR(d) = λ * sim(query, d) - (1-λ) * max_{s ∈ S} sim(d, s)
        - λ=1.0: chỉ quan tâm relevance (giống cosine search)
        - λ=0.0: chỉ quan tâm diversity (tối đa hóa sự khác biệt)
        - λ=0.7 (default): ưu tiên relevance nhưng tránh duplicate

    Args:
        query_embedding: Vector embedding của query (list[float])
        candidates: List of {'content': str, 'score': float,
                              'embedding': list[float], 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off relevance vs diversity

    Returns:
        List of top_k candidates selected by MMR, sorted by mmr_score.
    """
    import numpy as np

    if not candidates:
        return []

    def cosine_sim(a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    selected_indices: list[int] = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_mmr = float("-inf")

        for idx in remaining_indices:
            emb = candidates[idx].get("embedding")
            if emb is None:
                # Nếu không có embedding, dùng score gốc làm relevance
                relevance = candidates[idx].get("score", 0.0)
            else:
                relevance = cosine_sim(query_embedding, emb)

            max_sim_selected = 0.0
            for sel_idx in selected_indices:
                sel_emb = candidates[sel_idx].get("embedding")
                cur_emb = candidates[idx].get("embedding")
                if sel_emb and cur_emb:
                    sim = cosine_sim(cur_emb, sel_emb)
                    max_sim_selected = max(max_sim_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_selected

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "rrf" | "mmr"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: "cross_encoder" (Jina API) | "rrf" | "mmr"

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        # RRF nhận nhiều ranked lists — wrap candidates thành 1 list
        return rerank_rrf([candidates], top_k)
    elif method == "mmr":
        query_embedding: list[float] = []
        return rerank_mmr(query_embedding, candidates, top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý tại nhà riêng", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ chất cấm", "score": 0.6, "metadata": {}},
        {"content": "Bộ luật Hình sự 2015 quy định mức phạt tội ma tuý", "score": 0.55, "metadata": {}},
        {"content": "Công an bắt giữ nghệ sĩ liên quan đến ma tuý", "score": 0.5, "metadata": {}},
    ]

    print("=== Cross-encoder (Jina Reranker v2) ===")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=3)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
