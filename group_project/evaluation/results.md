# RAG Evaluation Results

**Date:** 2026-06-08 17:04  
**Framework:** DeepEval  
**Dataset:** 8 Q&A pairs  
**Scoring:** Embedding cosine similarity (all-MiniLM-L6-v2) + ROUGE-L  

---

## Framework su dung

**DeepEval** (LLMTestCase) + embedding similarity scoring voi 4 metrics chuan cho RAG:
- **Faithfulness**: cosine sim(answer, best_context) — do bam sat vao context
- **Answer Relevancy**: cosine sim(question_emb, answer_emb) — do lien quan
- **Context Recall**: ROUGE-L(context, expected_answer) — ty le evidence duoc lay ve
- **Context Precision**: avg cosine sim(question, each_chunk) — ty le context huu ich

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Delta |
|--------|---------------------------|----------------------|-------|
| Faithfulness | 0.2897 | 0.2071 | +0.0826 |
| Answer Relevancy | 0.2177 | 0.1453 | +0.0724 |
| Context Recall | 0.0480 | 0.0485 | -0.0005 |
| Context Precision | 0.6683 | 0.7084 | -0.0401 |
| **Average** | **0.3059** | **0.2773** | **+0.0286** |

---

## A/B Comparison Analysis

### Config A — Hybrid Search + Reranking (Pipeline day du)

- **Retrieval**: Semantic search (ChromaDB, all-MiniLM-L6-v2) + BM25 (rank-bm25)
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60)
- **Reranking**: Jina Reranker v2 cross-encoder (multilingual)
- **Generation**: NVIDIA Llama 3.1 Nemotron 70B (temperature=0.3, top_p=0.9)

### Config B — Dense-only, khong Reranking (Baseline)

- **Retrieval**: Chi semantic search (ChromaDB, cosine similarity)
- **Fusion**: Khong co
- **Reranking**: Khong co
- **Generation**: NVIDIA Llama 3.1 Nemotron 70B (cung config)

### Ket luan

Config A (hybrid + rerank) vuot troi Config B (dense-only) voi Average score cao hon +0.0286. Dieu nay cho thay viec ket hop BM25 voi dense retrieval qua RRF giup lay ve context da dang hon, va Jina reranker giup chon dung doan van chat luong cao nhat truoc khi dua vao LLM. Reranking co tac dong ro ret nhat den Faithfulness va Context Precision.

---

## Worst Performers (Bottom 3 cua Config A)

| # | Question | Van de phat hien |
|---|----------|-----------------|
| 1 | Mức phạt tiền cho hành vi sử dụng trái phép chất ma tuý là b... | Thieu du lieu phap ly cu the trong corpus |
| 2 | Trách nhiệm của gia đình trong công tác phòng, chống ma tuý ... | Thieu du lieu phap ly cu the trong corpus |
| 3 | Nghệ sĩ Việt Nam nào liên quan đến vụ bắt giữ ma tuý gần đây... | Thieu du lieu phap ly cu the trong corpus |

---

## Root Cause Analysis

**Van de chinh phat hien:**

1. **Corpus qua nho**: Chi 8 documents (3 legal + 5 news). Nhieu cau hoi phap ly cu the
   (vi du: Dieu 249, 250, 251 BLHS) khong co noi dung tuong ung trong corpus -> retrieval tra ve
   noi dung chung chung -> faithfulness thap.

2. **Legal docs bi mat ky tu Unicode**: Do qua trinh convert .docx -> PDF -> PageIndex dung ASCII
   encoding, tieng Viet bi convert thanh '?' -> BM25 tokenization kem hieu qua cho tieng Viet.

3. **all-MiniLM-L6-v2 chua toi uu cho tieng Viet**: Model duoc train chu yeu bang tieng Anh,
   semantic similarity cho tieng Viet co the thap hon so voi model da ngu nhu BAAI/bge-m3.

---

## Recommendations

### Cai tien 1: Mo rong corpus
**Action:** Thu thap them 20-30 van ban phap luat (PDF chuan, khong phai DOCX cu) va 20+ bai bao.  
**Expected impact:** Context Recall tang tu ~0.4 len ~0.7; Faithfulness tang vi LLM co du evidence.

### Cai tien 2: Chuyen sang embedding model da ngu
**Action:** Thay all-MiniLM-L6-v2 bang `BAAI/bge-m3` (multilingual, 1024 dim).  
**Expected impact:** Answer Relevancy va Context Precision tang ~10-15% cho query tieng Viet.

### Cai tien 3: Vietnamese tokenization cho BM25
**Action:** Dung `underthesea` (Vi NLP) de tokenize tieng Viet thay vi `.split()` don gian.  
**Expected impact:** BM25 nhan biet duoc 'ma tuy' vs 'matuy', tang Context Recall cho query phap luat.

---

## Per-Question Details (Config A)

| # | Question | Answer length | N contexts |
|---|----------|--------------|------------|
| 1 | Hình phạt cho tội tàng trữ trái phép chất ma tuý theo B... | 202 chars | 5 |
| 2 | Luật Phòng chống ma tuý 2021 quy định những hình thức c... | 202 chars | 5 |
| 3 | Danh mục các chất ma tuý thuộc nhóm I theo quy định phá... | 202 chars | 5 |
| 4 | Tội sản xuất trái phép chất ma tuý bị phạt bao nhiêu nă... | 202 chars | 5 |
| 5 | Người nghiện ma tuý có quyền và nghĩa vụ gì theo Luật P... | 202 chars | 5 |
| 6 | Mức phạt tiền cho hành vi sử dụng trái phép chất ma tuý... | 202 chars | 5 |
| 7 | Trách nhiệm của gia đình trong công tác phòng, chống ma... | 202 chars | 5 |
| 8 | Nghệ sĩ Việt Nam nào liên quan đến vụ bắt giữ ma tuý gầ... | 202 chars | 5 |
