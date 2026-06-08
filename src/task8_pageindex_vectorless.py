"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex fpdf2
"""

import json
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
# Cache doc_ids de tranh upload lai
DOC_IDS_CACHE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

# Helpers

def _md_to_temp_pdf(md_path: Path) -> Path:
    """
    Convert markdown file -> temp PDF (ASCII-safe cho FPDF2).
    PageIndex chi nhan PDF, nen can buoc nay.

    Tra ve path cua file PDF tam thoi. Caller co trach nhiem xoa sau khi dung.
    """
    from fpdf import FPDF

    text = md_path.read_text(encoding="utf-8")
    # Normalize: bo ky tu non-latin (FPDF2 core fonts khong ho tro Unicode)
    text_safe = text.encode("ascii", errors="replace").decode("ascii")

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)

    for line in text_safe.split("\n"):
        line = line[:300]
        try:
            pdf.multi_cell(0, 4, line)
        except Exception:
            pass

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp.name)
    return Path(tmp.name)


def _load_doc_ids() -> dict:
    """Load cache doc_ids tu file JSON."""
    if DOC_IDS_CACHE.exists():
        return json.loads(DOC_IDS_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_doc_ids(doc_ids: dict) -> None:
    """Luu cache doc_ids ra file JSON."""
    DOC_IDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DOC_IDS_CACHE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def upload_documents(wait_for_completion: bool = False) -> dict:
    """
    Convert markdown files -> PDF va upload len PageIndex.

    PageIndex build tree index tu PDF -> san sang cho chat/retrieval.

    Args:
        wait_for_completion: Neu True, cho den khi tat ca docs duoc xu ly xong.

    Returns:
        dict mapping filename -> doc_id
    """
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY chua duoc set trong .env")

    pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    existing = _load_doc_ids()

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print("  Khong tim thay file markdown trong data/standardized/")
        return existing

    for md_file in md_files:
        if md_file.name in existing:
            print(f"  Skip (da upload): {md_file.name} -> {existing[md_file.name]}")
            continue

        print(f"  Converting {md_file.name} -> PDF...")
        tmp_pdf = _md_to_temp_pdf(md_file)
        try:
            result = pi.submit_document(str(tmp_pdf))
            doc_id = result["doc_id"]
            existing[md_file.name] = doc_id
            print(f"  Uploaded: {md_file.name} -> {doc_id}")
        except Exception as e:
            print(f"  Loi upload {md_file.name}: {e}")
        finally:
            tmp_pdf.unlink(missing_ok=True)

    _save_doc_ids(existing)

    if wait_for_completion:
        print("  Cho PageIndex xu ly...")
        for fname, doc_id in existing.items():
            for _ in range(30):  # max ~5 phut
                try:
                    doc = pi.get_document(doc_id)
                    if doc.get("status") == "completed":
                        print(f"  Completed: {fname}")
                        break
                    elif doc.get("status") == "failed":
                        print(f"  Failed: {fname}")
                        break
                except Exception:
                    pass
                time.sleep(10)

    return existing


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Cau tra loi tu PageIndex
            'score': float,      # 1.0 (PageIndex khong tra ve numeric score)
            'metadata': dict,    # doc_ids duoc dung
            'source': 'pageindex'
        }
        Tra ve [] neu khong co documents hoac khong co API key.
    """
    if not PAGEINDEX_API_KEY:
        print("  PAGEINDEX_API_KEY chua set -- bo qua PageIndex fallback")
        return []

    from pageindex import PageIndexClient

    pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    # Lay doc_ids tu cache hoac list_documents()
    cached = _load_doc_ids()
    if cached:
        doc_ids = list(cached.values())[:top_k]
    else:
        try:
            result = pi.list_documents(limit=top_k)
            doc_ids = [d["id"] for d in result.get("documents", [])]
        except Exception as e:
            print(f"  Loi list_documents: {e}")
            return []

    if not doc_ids:
        print("  Khong co documents nao tren PageIndex -- hay chay upload_documents() truoc")
        return []

    # Prompt yeu cau trich dan doan van lien quan
    prompt = (
        f"Tim va trich dan cac doan van ban lien quan den: {query}\n"
        "Tra loi bang tieng Viet, bao gom cac doan trich dan co nguon goc ro rang."
    )

    try:
        response = pi.chat_completions(
            messages=[{"role": "user", "content": prompt}],
            doc_id=doc_ids,
            enable_citations=True,
        )
        content = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  Loi PageIndex chat: {e}")
        return []

    return [
        {
            "content": content,
            "score": 1.0,
            "metadata": {
                "source": "pageindex",
                "doc_ids": doc_ids,
                "query": query,
            },
            "source": "pageindex",
        }
    ]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY chua set trong .env")
        print("Dang ky tai: https://pageindex.ai/")
    else:
        print("=== Upload documents len PageIndex ===")
        doc_ids = upload_documents(wait_for_completion=True)
        print(f"\nDa upload {len(doc_ids)} documents")

        print("\n=== Test query ===")
        results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] [{r['source']}] {r['content'][:200]}...")
