"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from task4_chunking_indexing import EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    # Bước 1: Embed query bằng cùng model đã dùng ở Task 4
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query).tolist()

    # Bước 2: Query ChromaDB bằng cosine similarity
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Bước 3: Convert distance → similarity và format output
    # ChromaDB cosine: distance = 1 - similarity
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "content": doc,
            "score": round(1 - dist, 4),
            "metadata": meta,
        })

    # Đã sorted descending theo score (ChromaDB trả về theo distance ascending)
    return output


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] [{r['metadata'].get('type')}] {r['content'][:100]}...")
