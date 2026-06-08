"""
Task 6 — Lexical Search Module (BM25).

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Cài đặt:
    pip install rank-bm25
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task4_chunking_indexing import load_documents, chunk_documents

# Build corpus và BM25 index một lần khi import
_corpus: list[dict] = []
_bm25 = None


def _get_bm25():
    """Lazy load BM25 index từ corpus."""
    global _corpus, _bm25
    if _bm25 is not None:
        return _bm25, _corpus

    from rank_bm25 import BM25Okapi

    docs = load_documents()
    _corpus = chunk_documents(docs)

    # Tokenize — split() đơn giản, đủ dùng cho BM25 tiếng Việt
    tokenized_corpus = [doc["content"].lower().split() for doc in _corpus]
    _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25, _corpus


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    import numpy as np

    bm25, corpus = _get_bm25()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k indices, sorted descending theo score
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })

    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] [{r['metadata'].get('type')}] {r['content'][:100]}...")
