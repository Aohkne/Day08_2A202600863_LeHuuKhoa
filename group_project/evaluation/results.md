# RAG Evaluation Results

**Date:** 2026-07-28 10:58  
**Framework:** DeepEval  
**Dataset:** 15 Q&A pairs  
**Scoring:** DeepEval LLM-as-judge (gpt-4o-mini)  

---

## Framework su dung

**DeepEval** metrics built-in, cham diem bang LLM-as-judge (gpt-4o-mini qua OpenAI API):
- **FaithfulnessMetric**: answer co bam dung retrieval_context khong (khong bia dat/hallucinate)
- **AnswerRelevancyMetric**: answer co tra loi dung trong tam cau hoi khong
- **ContextualRecallMetric**: retrieval_context co du evidence de sinh ra expected_output khong
- **ContextualPrecisionMetric**: trong retrieval_context, ty le chunk thuc su lien quan la bao nhieu

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Delta |
|--------|---------------------------|----------------------|-------|
| Faithfulness | 0.9167 | 0.8678 | +0.0489 |
| Answer Relevancy | 0.4278 | 0.8489 | -0.4211 |
| Context Recall | 0.9333 | 0.6667 | +0.2666 |
| Context Precision | 0.4656 | 0.2072 | +0.2584 |
| **Average** | **0.6858** | **0.6476** | **+0.0382** |

---

## A/B Comparison Analysis

### Config A — Hybrid Search + Reranking (Pipeline day du)

- **Retrieval**: Semantic search (ChromaDB, all-MiniLM-L6-v2) + BM25 (rank-bm25)
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60)
- **Reranking**: Jina Reranker v2 cross-encoder (multilingual)
- **Generation**: OpenAI gpt-4o-mini (temperature=0.3, top_p=0.9)

### Config B — Dense-only, khong Reranking (Baseline)

- **Retrieval**: Chi semantic search (ChromaDB, cosine similarity)
- **Fusion**: Khong co
- **Reranking**: Khong co
- **Generation**: OpenAI gpt-4o-mini (cung config)

### Ket luan

Config A tu choi tra loi (8/15 cau, vi SYSTEM_PROMPT cua Task 10 bat buoc tra ve 'Toi khong the xac minh thong tin nay' khi khong tim thay evidence khop chinh xac — vd so dieu luat cu the) nhieu hon han Config B (0/15 cau, prompt long leo hon, khong yeu cau tu choi). Day la nguyen nhan chinh khien Answer Relevancy cua Config A (0.4278) thap hon Config B (0.8489: cau tra loi tu choi bi giam judge cham diem 'khong lien quan'. Doi lai, Faithfulness cua Config A (0.9167) cao hon han Config B (0.8678) vi khi Config A tra loi that, no bam rat sat context, khong bia dat. Day la tradeoff giua 'an toan/khong hallucinate' (Config A, dung spec Task 10) va 'luon co cau tra loi' (Config B) — khong phai loi cua hybrid retrieval. De cai thien Answer Relevancy ma khong mat Faithfulness, nen mo rong corpus (them Bo luat Hinh su) de retrieval tim duoc evidence chinh xac hon, giam ty le tu choi oan.

---

## Worst Performers (Bottom 3 cua Config A, theo avg score)

| # | Question | Avg Score | Van de phat hien |
|---|----------|-----------|-----------------|
| 1 | Luật Phòng chống ma tuý 2021 quy định những hình thức cai ng... | 0.5000 | Context lay ve khong du lien quan (Context Precision thap) |
| 2 | Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luậ... | 0.5000 | Context lay ve khong du lien quan (Context Precision thap) |
| 3 | Tội sản xuất trái phép chất ma tuý bị phạt bao nhiêu năm tù? | 0.5000 | Context lay ve khong du lien quan (Context Precision thap) |

---

## Root Cause Analysis

**Van de chinh phat hien:**

1. **Con thieu mot so van ban chuyen sau**: Corpus hien co 9 documents, da bo sung Bo luat Hinh su
   2015 Chuong XX (Dieu 247-259) — giam ty le tu choi tu 11/15 xuong 8/15. Nhung mot so cau hoi con lai
   can noi dung ngoai pham vi Chuong XX (vd: quy dinh chung ve nguoi nuoc ngoai pham toi o Phan chung
   BLHS) hoac can Nghi dinh 57/2022 (danh muc chat ma tuy) — cac van ban nay chua co trong corpus.

2. **Retrieval doi khi bo sot noi dung da co san**: Mot vai cau hoi (vd ve Chuong III, Chuong V
   Luat 73/2021) co noi dung trong corpus nhung khong lot vao top-5 sau RRF + rerank —
   can tang top_k truoc rerank hoac giam chunk_size de tang do chinh xac.

3. **all-MiniLM-L6-v2 chua toi uu cho tieng Viet**: Model duoc train chu yeu bang tieng Anh,
   semantic similarity cho tieng Viet co the thap hon so voi model da ngu nhu BAAI/bge-m3.

---

## Recommendations

### Cai tien 1: Bo sung cac van ban con thieu
**Action:** Them Nghi dinh 57/2022/ND-CP (danh muc chat ma tuy) va phan quy dinh chung ve
hieu luc doi voi nguoi nuoc ngoai pham toi (Chuong I, Phan thu nhat BLHS) vao corpus.  
**Expected impact:** Giai quyet cac cau hoi con lai dang bi tu choi vi thieu evidence dung.

### Cai tien 2: Tang top_k truoc rerank + giam chunk_size
**Action:** Lay top_k=15-20 candidates truoc khi rerank xuong top_k=5 (hien retrieve truc tiep 5),
va thu giam chunk_size tu 500 xuong 300 de tang do chinh xac cua chunk chua dieu luat cu the.  
**Expected impact:** Giam ty le retrieval bo sot noi dung da co san trong corpus.

### Cai tien 3: Chuyen sang embedding model da ngu
**Action:** Thay all-MiniLM-L6-v2 bang `BAAI/bge-m3` (multilingual, 1024 dim).  
**Expected impact:** Answer Relevancy va Context Precision tang ~10-15% cho query tieng Viet.

### Cai tien 4: Vietnamese tokenization cho BM25
**Action:** Dung `underthesea` (Vi NLP) de tokenize tieng Viet thay vi `.split()` don gian.  
**Expected impact:** BM25 nhan biet duoc 'ma tuy' vs 'matuy', tang Context Recall cho query phap luat.

---

## Per-Question Details (Config A)

| # | Question | Answer length | N contexts |
|---|----------|--------------|------------|
| 1 | Hình phạt cho tội tàng trữ trái phép chất ma tuý theo B... | 716 chars | 5 |
| 2 | Luật Phòng chống ma tuý 2021 quy định những hình thức c... | 54 chars | 5 |
| 3 | Danh mục các chất ma tuý thuộc nhóm I theo quy định phá... | 54 chars | 5 |
| 4 | Tội sản xuất trái phép chất ma tuý bị phạt bao nhiêu nă... | 54 chars | 5 |
| 5 | Người nghiện ma tuý có quyền và nghĩa vụ gì theo Luật P... | 54 chars | 5 |
| 6 | Mức phạt tiền cho hành vi sử dụng trái phép chất ma tuý... | 339 chars | 5 |
| 7 | Trách nhiệm của gia đình trong công tác phòng, chống ma... | 798 chars | 5 |
| 8 | Nghệ sĩ Việt Nam nào liên quan đến vụ bắt giữ ma tuý gầ... | 659 chars | 5 |
| 9 | Hình phạt cho tội mua bán trái phép chất ma tuý theo Đi... | 795 chars | 5 |
| 10 | Cơ quan nào có thẩm quyền quyết định áp dụng biện pháp ... | 54 chars | 5 |
| 11 | Thời gian cai nghiện bắt buộc tối thiểu và tối đa là ba... | 54 chars | 5 |
| 12 | Ma tuý tổng hợp là gì và có những loại nào phổ biến tại... | 528 chars | 5 |
| 13 | Người nước ngoài phạm tội ma tuý tại Việt Nam bị xử lý ... | 54 chars | 5 |
| 14 | Tội vận chuyển trái phép chất ma tuý có mức phạt tù cao... | 354 chars | 5 |
| 15 | Các biện pháp quản lý sau cai nghiện bao gồm những gì? | 801 chars | 5 |
