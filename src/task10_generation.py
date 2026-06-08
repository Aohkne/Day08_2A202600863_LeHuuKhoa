"""
Task 10 — Generation Co Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): xác suất tích lũy cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random cho RAG
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = "meta/llama-3.1-70b-instruct"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Ví dụ với 5 chunks [1, 2, 3, 4, 5] (sorted by score desc):
        Output: [1, 3, 5, 4, 2]
        - Chunk 1 (quan trọng nhất) -> ĐẦU
        - Chunk 2 (quan trọng thứ 2) -> CUỐI
        - Chunk 3, 4, 5 -> GIỮA (ít được LLM chú ý)

    Reference: Liu et al. (2023) "Lost in the Middle"

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # Split: odd indices -> front half, even indices -> back half (reversed)
    front = [chunks[i] for i in range(0, len(chunks), 2)]   # [0, 2, 4, ...]
    back = [chunks[i] for i in range(1, len(chunks), 2)]    # [1, 3, 5, ...]

    # front: most important first; back: most important last (reversed)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation co citation.

    Pipeline:
        1. Retrieve relevant chunks (Task 9)
        2. Reorder de tranh lost in the middle
        3. Format context voi source labels
        4. Build prompt (system + context + query)
        5. Call LLM (NVIDIA API)
        6. Return answer + sources

    Args:
        query: Cau hoi cua user
        top_k: So chunks dua vao context

    Returns:
        {
            'answer': str,           # Cau tra loi co citation
            'sources': list[dict],   # Cac chunks da dung
            'retrieval_source': str  # 'hybrid' hoac 'pageindex'
        }
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Toi khong the xac minh thong tin nay tu nguon hien co.",
            "sources": [],
            "retrieval_source": "none",
        }

    retrieval_source = chunks[0].get("source", "hybrid")

    # Step 2: Reorder chunks de tranh lost in the middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context voi source labels cho citation
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Please answer based on the context above with citations."
    )

    # Step 5: Call LLM
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"Loi khi goi LLM: {e}"

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    query = "Hinh phat cho toi tang tru trai phep chat ma tuy la bao nhieu nam tu?"
    print(f"Query: {query}\n")

    result = generate_with_citation(query, top_k=5)
    print("=== ANSWER ===")
    print(result["answer"])
    print(f"\n=== SOURCES ({len(result['sources'])} chunks, via {result['retrieval_source']}) ===")
    for i, s in enumerate(result["sources"], 1):
        print(f"  {i}. [{s.get('metadata', {}).get('source', 'unknown')}] {s['content'][:60]}...")
