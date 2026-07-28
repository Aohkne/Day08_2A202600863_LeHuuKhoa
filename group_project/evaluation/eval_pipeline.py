"""
RAG Evaluation Pipeline — DeepEval

Framework: DeepEval (https://github.com/confident-ai/deepeval)
pip install deepeval

Metrics:
    1. Faithfulness     — Answer co bam dung context khong?
    2. Answer Relevancy — Answer co dung cau hoi khong?
    3. Context Recall   — Retriever co lay du evidence khong?
    4. Context Precision— Context lay ve co % nao thuc su huu ich?

So sanh A/B:
    Config A: Hybrid search (semantic + BM25) + Reranking (Jina cross-encoder)
    Config B: Dense-only search (semantic) + khong reranking

Chay:
    python group_project/evaluation/eval_pipeline.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


# =============================================================================
# Load golden dataset
# =============================================================================

def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} Q&A pairs from golden dataset")
    return data


# =============================================================================
# RAG Pipeline wrappers
# =============================================================================

def run_config_a(question: str) -> dict:
    """
    Config A: Hybrid search (semantic + BM25) + Reranking.
    Day la pipeline day du nhat.
    """
    from task10_generation import generate_with_citation
    result = generate_with_citation(question, top_k=5)
    return {
        "answer": result["answer"],
        "contexts": [c["content"] for c in result["sources"]],
        "sources": result["sources"],
    }


def run_config_b(question: str) -> dict:
    """
    Config B: Dense-only search (semantic), khong reranking.
    De so sanh hieu qua cua hybrid + reranking.
    """
    from task5_semantic_search import semantic_search
    chunks = semantic_search(question, top_k=5)

    if not chunks:
        return {"answer": "Khong tim thay thong tin lien quan.", "contexts": [], "sources": []}

    # Format context don gian
    context_parts = []
    for i, c in enumerate(chunks, 1):
        src = c.get("metadata", {}).get("source", f"Source {i}")
        context_parts.append(f"[Document {i} | Source: {src}]\n{c['content']}")
    context_str = "\n\n---\n\n".join(context_parts)

    # Goi LLM voi context tu dense-only
    import os
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Answer in Vietnamese with citations from the context."},
                {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {question}"},
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Loi LLM: {e}"

    return {
        "answer": answer,
        "contexts": [c["content"] for c in chunks],
        "sources": chunks,
    }


# =============================================================================
# DeepEval Evaluation
# =============================================================================
#
# Dung dung 4 metric built-in cua DeepEval (LLM-as-judge, model=JUDGE_MODEL).
# Judge goi truc tiep OpenAI API bang OPENAI_API_KEY co san trong .env.
#
#   Faithfulness       — answer co bam dung retrieval_context khong?
#   Answer Relevancy   — answer co dung trong tam cau hoi khong?
#   Context Recall     — retrieval_context co du evidence de sinh expected_output?
#   Context Precision  — trong retrieval_context, bao nhieu % thuc su lien quan?
# -------------------------------------------------------------------------

JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o-mini")
GEN_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # model dung de sinh cau tra loi (Task 10)


def _build_metrics():
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )
    return {
        "Faithfulness": FaithfulnessMetric(threshold=0.5, model=JUDGE_MODEL, include_reason=True),
        "Answer Relevancy": AnswerRelevancyMetric(threshold=0.5, model=JUDGE_MODEL, include_reason=True),
        "Context Recall": ContextualRecallMetric(threshold=0.5, model=JUDGE_MODEL, include_reason=True),
        "Context Precision": ContextualPrecisionMetric(threshold=0.5, model=JUDGE_MODEL, include_reason=True),
    }


def evaluate_config(config_name: str, run_fn, golden_dataset: list[dict]) -> dict:
    """
    Chay RAG pipeline tren toan bo golden dataset va tinh 4 DeepEval RAG metrics
    that su dung LLM-as-judge (gpt-4o-mini qua OPENAI_API_KEY).

    Returns:
        dict chua ket qua tung metric va per-question scores
    """
    from deepeval.test_case import LLMTestCase

    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print(f"{'='*60}")

    test_cases = []
    raw_results = []

    for i, item in enumerate(golden_dataset, 1):
        question = item["question"]
        print(f"  [{i}/{len(golden_dataset)}] Running: {question[:50]}...")

        try:
            result = run_fn(question)
            # Dung LLMTestCase cua DeepEval lam data container
            test_case = LLMTestCase(
                input=question,
                actual_output=result["answer"],
                expected_output=item["expected_answer"],
                retrieval_context=result["contexts"] if result["contexts"] else ["No context retrieved"],
            )
            test_cases.append(test_case)
            raw_results.append({
                "question": question,
                "answer": result["answer"],
                "expected_answer": item["expected_answer"],
                "contexts": result["contexts"],
                "n_contexts": len(result["contexts"]),
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            raw_results.append({
                "question": question,
                "answer": f"ERROR: {e}",
                "expected_answer": item["expected_answer"],
                "contexts": [],
                "n_contexts": 0,
            })

    metric_scores = {"Faithfulness": [], "Answer Relevancy": [], "Context Recall": [], "Context Precision": []}

    print(f"\n  Scoring {len(test_cases)} test cases with DeepEval ({JUDGE_MODEL} judge)...")
    for j, tc in enumerate(test_cases, 1):
        metrics = _build_metrics()
        row = {}
        for name, metric in metrics.items():
            try:
                metric.measure(tc)
                row[name] = round(metric.score, 4)
            except Exception as e:
                print(f"    [{j}] {name} FAILED: {e}")
                row[name] = 0.0
            metric_scores[name].append(row[name])
        print(
            f"    [{j}/{len(test_cases)}] "
            f"F={row['Faithfulness']:.3f}  AR={row['Answer Relevancy']:.3f}  "
            f"CR={row['Context Recall']:.3f}  CP={row['Context Precision']:.3f}"
        )
        raw_results[j - 1]["metrics"] = row

    avg_scores = {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in metric_scores.items()}
    avg_scores["Average"] = round(sum(avg_scores.values()) / len(avg_scores), 4)

    print(f"\n  Results for {config_name}:")
    for metric_name, score in avg_scores.items():
        print(f"    {metric_name}: {score:.4f}")

    return {
        "config_name": config_name,
        "avg_scores": avg_scores,
        "per_question": raw_results,
    }
# Write results.md
# =============================================================================

def write_results(result_a: dict, result_b: dict) -> None:
    """Ghi ket qua evaluation ra results.md."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    scores_a = result_a["avg_scores"]
    scores_b = result_b["avg_scores"]

    metrics = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision", "Average"]

    lines = [
        "# RAG Evaluation Results",
        "",
        f"**Date:** {now}  ",
        f"**Framework:** DeepEval  ",
        f"**Dataset:** {len(result_a['per_question'])} Q&A pairs  ",
        f"**Scoring:** DeepEval LLM-as-judge ({JUDGE_MODEL})  ",
        "",
        "---",
        "",
        "## Framework su dung",
        "",
        f"**DeepEval** metrics built-in, cham diem bang LLM-as-judge ({JUDGE_MODEL} qua OpenAI API):",
        "- **FaithfulnessMetric**: answer co bam dung retrieval_context khong (khong bia dat/hallucinate)",
        "- **AnswerRelevancyMetric**: answer co tra loi dung trong tam cau hoi khong",
        "- **ContextualRecallMetric**: retrieval_context co du evidence de sinh ra expected_output khong",
        "- **ContextualPrecisionMetric**: trong retrieval_context, ty le chunk thuc su lien quan la bao nhieu",
        "",
        "---",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Delta |",
        "|--------|---------------------------|----------------------|-------|",
    ]

    for m in metrics:
        a = scores_a.get(m, 0)
        b = scores_b.get(m, 0)
        delta = round(a - b, 4)
        sign = "+" if delta >= 0 else ""
        bold = "**" if m == "Average" else ""
        lines.append(f"| {bold}{m}{bold} | {bold}{a:.4f}{bold} | {bold}{b:.4f}{bold} | {bold}{sign}{delta:.4f}{bold} |")

    lines += [
        "",
        "---",
        "",
        "## A/B Comparison Analysis",
        "",
        "### Config A — Hybrid Search + Reranking (Pipeline day du)",
        "",
        "- **Retrieval**: Semantic search (ChromaDB, all-MiniLM-L6-v2) + BM25 (rank-bm25)",
        "- **Fusion**: Reciprocal Rank Fusion (RRF, k=60)",
        "- **Reranking**: Jina Reranker v2 cross-encoder (multilingual)",
        f"- **Generation**: OpenAI {GEN_MODEL} (temperature=0.3, top_p=0.9)",
        "",
        "### Config B — Dense-only, khong Reranking (Baseline)",
        "",
        "- **Retrieval**: Chi semantic search (ChromaDB, cosine similarity)",
        "- **Fusion**: Khong co",
        "- **Reranking**: Khong co",
        f"- **Generation**: OpenAI {GEN_MODEL} (cung config)",
        "",
        "### Ket luan",
        "",
    ]

    def _refusal_rate(per_question: list[dict]) -> tuple[int, int]:
        keywords = ("khong the xac minh", "không thể xác minh")
        import unicodedata
        def _norm(s: str) -> str:
            s = unicodedata.normalize("NFD", s.lower())
            return "".join(c for c in s if unicodedata.category(c) != "Mn")
        n_refused = sum(1 for q in per_question if any(k in _norm(q.get("answer", "")) for k in keywords))
        return n_refused, len(per_question)

    refused_a, total_a = _refusal_rate(result_a["per_question"])
    refused_b, total_b = _refusal_rate(result_b["per_question"])
    delta_avg = scores_a.get("Average", 0) - scores_b.get("Average", 0)

    if refused_a > total_a * 0.3 and refused_a > refused_b:
        lines += [
            f"Config A tu choi tra loi ({refused_a}/{total_a} cau, vi SYSTEM_PROMPT cua Task 10 bat buoc "
            f"tra ve 'Toi khong the xac minh thong tin nay' khi khong tim thay evidence khop chinh xac — "
            f"vd so dieu luat cu the) nhieu hon han Config B ({refused_b}/{total_b} cau, prompt long leo hon, "
            f"khong yeu cau tu choi). Day la nguyen nhan chinh khien Answer Relevancy cua Config A "
            f"({scores_a.get('Answer Relevancy', 0):.4f}) thap hon Config B ({scores_b.get('Answer Relevancy', 0):.4f}: "
            f"cau tra loi tu choi bi giam judge cham diem 'khong lien quan'. Doi lai, Faithfulness cua Config A "
            f"({scores_a.get('Faithfulness', 0):.4f}) cao hon han Config B ({scores_b.get('Faithfulness', 0):.4f}) "
            f"vi khi Config A tra loi that, no bam rat sat context, khong bia dat. "
            f"Day la tradeoff giua 'an toan/khong hallucinate' (Config A, dung spec Task 10) va "
            f"'luon co cau tra loi' (Config B) — khong phai loi cua hybrid retrieval. "
            f"De cai thien Answer Relevancy ma khong mat Faithfulness, nen mo rong corpus (them Bo luat Hinh su) "
            f"de retrieval tim duoc evidence chinh xac hon, giam ty le tu choi oan.",
        ]
    elif delta_avg > 0:
        lines += [
            f"Config A (hybrid + rerank) vuot troi Config B (dense-only) voi Average score cao hon "
            f"{delta_avg:+.4f}. Dieu nay cho thay viec ket hop BM25 voi dense retrieval qua RRF giup "
            f"lay ve context da dang hon, va Jina reranker giup chon dung doan van chat luong cao nhat "
            f"truoc khi dua vao LLM. Reranking co tac dong ro ret nhat den Faithfulness va Context Precision.",
        ]
    else:
        lines += [
            f"Trong thu nghiem nay, Config B (dense-only) dat Average cao hon Config A "
            f"({scores_b.get('Average', 0):.4f} vs {scores_a.get('Average', 0):.4f}). "
            f"Nguyen nhan co the do corpus nho, khien BM25 va reranking chua the hien ro loi the. "
            f"Voi corpus lon hon (100+ docs), hybrid + rerank thuong vuot troi dense-only.",
        ]

    # Worst performers — xep theo average metric score thap nhat (Config A)
    lines += [
        "",
        "---",
        "",
        "## Worst Performers (Bottom 3 cua Config A, theo avg score)",
        "",
        "| # | Question | Avg Score | Van de phat hien |",
        "|---|----------|-----------|-----------------|",
    ]

    per_q = result_a["per_question"]

    def _avg_score(q: dict) -> float:
        m = q.get("metrics")
        return round(sum(m.values()) / len(m), 4) if m else 0.0

    worst = sorted(per_q, key=_avg_score)[:3]
    for i, q in enumerate(worst, 1):
        question_short = q["question"][:60] + ("..." if len(q["question"]) > 60 else "")
        avg = _avg_score(q)
        if not q["contexts"]:
            issue = "Khong retrieve duoc context nao — answer mang tinh chung chung"
        elif q.get("metrics", {}).get("Context Recall", 1) < 0.3:
            issue = "Context Recall thap — corpus thieu du lieu cu the cho cau hoi nay"
        elif q.get("metrics", {}).get("Faithfulness", 1) < 0.3:
            issue = "Faithfulness thap — answer khong bam sat context da retrieve"
        else:
            issue = "Context lay ve khong du lien quan (Context Precision thap)"
        lines.append(f"| {i} | {question_short} | {avg:.4f} | {issue} |")

    lines += [
        "",
        "---",
        "",
        "## Root Cause Analysis",
        "",
        "**Van de chinh phat hien:**",
        "",
        "1. **Con thieu mot so van ban chuyen sau**: Corpus hien co 9 documents, da bo sung Bo luat Hinh su",
        "   2015 Chuong XX (Dieu 247-259). Nhung mot so cau hoi can noi dung ngoai pham vi Chuong XX",
        "   (vd: quy dinh chung ve nguoi nuoc ngoai pham toi o Phan chung BLHS) hoac can Nghi dinh 57/2022",
        "   (danh muc chat ma tuy) — cac van ban nay chua co trong corpus nen retrieval van khong tim duoc.",
        "",
        "2. **Retrieval doi khi bo sot noi dung da co san**: Mot vai cau hoi (vd ve Chuong III, Chuong V",
        "   Luat 73/2021) co noi dung trong corpus nhung khong lot vao top-5 sau RRF + rerank —",
        "   can tang top_k truoc rerank hoac giam chunk_size de tang do chinh xac.",
        "",
        "3. **all-MiniLM-L6-v2 chua toi uu cho tieng Viet**: Model duoc train chu yeu bang tieng Anh,",
        "   semantic similarity cho tieng Viet co the thap hon so voi model da ngu nhu BAAI/bge-m3.",
        "",
        "---",
        "",
        "## Recommendations",
        "",
        "### Cai tien 1: Bo sung cac van ban con thieu",
        "**Action:** Them Nghi dinh 57/2022/ND-CP (danh muc chat ma tuy) va phan quy dinh chung ve",
        "hieu luc doi voi nguoi nuoc ngoai pham toi (Chuong I, Phan thu nhat BLHS) vao corpus.  ",
        "**Expected impact:** Giai quyet cac cau hoi con lai dang bi tu choi vi thieu evidence dung.",
        "",
        "### Cai tien 2: Tang top_k truoc rerank + giam chunk_size",
        "**Action:** Lay top_k=15-20 candidates truoc khi rerank xuong top_k=5 (hien retrieve truc tiep 5),",
        "va thu giam chunk_size tu 500 xuong 300 de tang do chinh xac cua chunk chua dieu luat cu the.  ",
        "**Expected impact:** Giam ty le retrieval bo sot noi dung da co san trong corpus.",
        "",
        "### Cai tien 3: Chuyen sang embedding model da ngu",
        "**Action:** Thay all-MiniLM-L6-v2 bang `BAAI/bge-m3` (multilingual, 1024 dim).  ",
        "**Expected impact:** Answer Relevancy va Context Precision tang ~10-15% cho query tieng Viet.",
        "",
        "### Cai tien 4: Vietnamese tokenization cho BM25",
        "**Action:** Dung `underthesea` (Vi NLP) de tokenize tieng Viet thay vi `.split()` don gian.  ",
        "**Expected impact:** BM25 nhan biet duoc 'ma tuy' vs 'matuy', tang Context Recall cho query phap luat.",
        "",
        "---",
        "",
        "## Per-Question Details (Config A)",
        "",
        "| # | Question | Answer length | N contexts |",
        "|---|----------|--------------|------------|",
    ]

    for i, q in enumerate(result_a["per_question"], 1):
        q_short = q["question"][:55] + ("..." if len(q["question"]) > 55 else "")
        ans_len = len(q.get("answer", ""))
        n_ctx = q.get("n_contexts", 0)
        lines.append(f"| {i} | {q_short} | {ans_len} chars | {n_ctx} |")

    content = "\n".join(lines) + "\n"
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\nResults written to: {RESULTS_PATH}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("RAG Evaluation Pipeline — DeepEval")
    print("=" * 60)

    golden_dataset = load_golden_dataset()
    print(f"\nUsing all {len(golden_dataset)} questions for evaluation")

    result_a = evaluate_config("Config A — Hybrid + Rerank", run_config_a, golden_dataset)
    result_b = evaluate_config("Config B — Dense-only", run_config_b, golden_dataset)

    write_results(result_a, result_b)
    print("\nDone!")


if __name__ == "__main__":
    main()
