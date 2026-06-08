"""
RAG Chatbot — Pháp luật ma tuý & Tin tức nghệ sĩ Việt Nam
Yêu cầu 1: Giao diện chat Streamlit với:
    - Trả lời có citation (Task 10: generate_with_citation)
    - Follow-up questions (conversation memory)
    - Hiển thị source documents đã dùng
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st

# Page config
st.set_page_config(
    page_title="DrugLaw RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
/* ─── Global ─── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2d3347;
}
[data-testid="stSidebar"] * { color: #e0e6f0 !important; }

/* ─── Headings ─── */
h1, h2, h3 { color: #e0e6f0 !important; }
p, li, label { color: #b0bcd4 !important; }

/* ─── Chat messages ─── */
[data-testid="stChatMessage"] {
    background: #1a2035 !important;
    border: 1px solid #2d3347;
    border-radius: 12px;
    margin-bottom: 10px;
    padding: 4px 0;
}

/* ─── Source card ─── */
.source-card {
    background: #1e2538;
    border-left: 3px solid #4f8ef7;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 0.84em;
    color: #c8d4e8;
}
.source-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.source-name {
    font-weight: 700;
    color: #4f8ef7;
    font-size: 0.95em;
}
.tag {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: 600;
}
.tag-news  { background: #1a3a2a; color: #4ade80; border: 1px solid #22543d; }
.tag-legal { background: #2a1f3a; color: #c084fc; border: 1px solid #553c7b; }
.tag-score { background: #1f2d3a; color: #60a5fa; border: 1px solid #1e3a5f; }
.source-content { color: #94a3b8; line-height: 1.5; }

/* ─── Retrieval badge ─── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: 600;
    margin-top: 6px;
}
.badge-hybrid    { background: #0d2618; color: #4ade80; border: 1px solid #166534; }
.badge-pageindex { background: #2d1f00; color: #fbbf24; border: 1px solid #78350f; }
.badge-error     { background: #2d0f0f; color: #f87171; border: 1px solid #7f1d1d; }

/* ─── Sidebar sample question buttons ─── */
[data-testid="stSidebar"] button {
    background: #1e2d45 !important;
    color: #93c5fd !important;
    border: 1px solid #2d4a6e !important;
    border-radius: 8px !important;
    font-size: 0.82em !important;
    text-align: left !important;
    padding: 6px 10px !important;
}
[data-testid="stSidebar"] button:hover {
    background: #243557 !important;
    border-color: #4f8ef7 !important;
}

/* ─── Expander ─── */
details { background: #161b27 !important; border: 1px solid #2d3347 !important; border-radius: 8px; }
summary { color: #93c5fd !important; font-weight: 600; }

/* ─── Chat input ─── */
[data-testid="stChatInput"] textarea {
    background: #1a2035 !important;
    color: #e0e6f0 !important;
    border: 1px solid #2d3347 !important;
    border-radius: 10px !important;
}

/* ─── Divider ─── */
hr { border-color: #2d3347 !important; }
</style>
""", unsafe_allow_html=True)

# Lazy-load pipeline (cached)
@st.cache_resource(show_spinner="Đang tải RAG pipeline...")
def load_pipeline():
    from task10_generation import generate_with_citation
    return generate_with_citation


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, sources, retrieval_source}
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False


# Sidebar
with st.sidebar:
    st.title("⚖️ DrugLaw RAG")
    st.markdown("Chatbot hỏi đáp về **pháp luật ma tuý Việt Nam** và **tin tức nghệ sĩ liên quan**.")

    st.divider()

    st.markdown("### ⚙️ Cấu hình")
    top_k = st.slider("Số chunks tối đa", min_value=3, max_value=10, value=5, step=1,
                      help="Số context chunks truyền vào LLM")
    show_sources = st.toggle("Hiển thị nguồn tài liệu", value=True)
    show_retrieval_info = st.toggle("Hiển thị thông tin retrieval", value=False)

    st.divider()

    st.markdown("### 💡 Câu hỏi gợi ý")
    sample_questions = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý là bao nhiêu?",
        "Luật Phòng chống ma tuý 2021 quy định những gì về cai nghiện?",
        "Nghệ sĩ Việt Nam nào liên quan đến vụ bắt giữ ma tuý gần đây?",
        "Tội sản xuất trái phép chất ma tuý bị phạt bao nhiêu năm tù?",
        "Mức phạt tiền cho hành vi sử dụng trái phép chất ma tuý là bao nhiêu?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state.pending_question = q

    st.divider()

    if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Stack: Streamlit · ChromaDB · BM25 · Jina Reranker · NVIDIA Llama 3.1 Nemotron 70B")


# Main area
st.title("⚖️ DrugLaw RAG Chatbot")
st.markdown(
    "<span style='color:#94a3b8'>Hỏi đáp về <b style='color:#93c5fd'>Luật Phòng chống ma tuý 2021</b>, "
    "<b style='color:#93c5fd'>Bộ luật Hình sự</b> và <b style='color:#93c5fd'>tin tức nghệ sĩ liên quan ma tuý</b>. "
    "Câu trả lời có trích dẫn nguồn.</span>",
    unsafe_allow_html=True
)

# Load pipeline
generate_fn = load_pipeline()


# ─── Helper: render source cards ────────────────────────────────────────────
def render_sources(sources: list[dict], expanded: bool = False):
    with st.expander(f"📄 Nguồn tài liệu — {len(sources)} chunks", expanded=expanded):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            source_name = meta.get("source", meta.get("file", f"Source {i}"))
            doc_type = meta.get("type", "unknown")
            score = src.get("score", 0.0)
            content_preview = src.get("content", "")[:280]
            ellipsis = "..." if len(src.get("content", "")) > 280 else ""

            tag_class = "tag-news" if doc_type == "news" else "tag-legal"
            type_icon = "📰" if doc_type == "news" else "📜"

            st.markdown(f"""
<div class="source-card">
  <div class="source-header">
    <span class="source-name">[{i}] {source_name}</span>
    <span class="tag {tag_class}">{type_icon} {doc_type}</span>
    <span class="tag tag-score">score: {score:.3f}</span>
  </div>
  <div class="source-content">{content_preview}{ellipsis}</div>
</div>
""", unsafe_allow_html=True)


def render_retrieval_badge(retrieval_source: str):
    badge_class = {
        "hybrid": "badge-hybrid",
        "pageindex": "badge-pageindex",
    }.get(retrieval_source, "badge-error")
    icons = {"hybrid": "🔀", "pageindex": "📑", "error": "⚠️"}
    icon = icons.get(retrieval_source, "⚠️")
    st.markdown(
        f'<span class="badge {badge_class}">{icon} retrieval: {retrieval_source}</span>',
        unsafe_allow_html=True
    )


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if show_sources and msg.get("sources"):
                render_sources(msg["sources"], expanded=False)
            if show_retrieval_info and msg.get("retrieval_source"):
                render_retrieval_badge(msg["retrieval_source"])


# Handle input (từ chat box hoặc sample button)
# Lấy pending question từ sample buttons nếu có
pending = st.session_state.pop("pending_question", None)

user_input = st.chat_input("Nhập câu hỏi về pháp luật ma tuý hoặc tin tức nghệ sĩ...") or pending

if user_input:
    # Build conversation context cho follow-up questions
    # Gắn thêm lịch sử (tối đa 3 lượt gần nhất) vào query
    history_context = ""
    recent = st.session_state.messages[-6:]  # 3 lượt = 6 messages
    if recent:
        history_parts = []
        for m in recent:
            role_label = "Người dùng" if m["role"] == "user" else "Trợ lý"
            history_parts.append(f"{role_label}: {m['content'][:200]}")
        history_context = "\n".join(history_parts)

    # Nếu có lịch sử, bổ sung context vào query để hỗ trợ follow-up
    if history_context:
        augmented_query = f"[Lịch sử hội thoại gần đây:\n{history_context}\n]\n\nCâu hỏi hiện tại: {user_input}"
    else:
        augmented_query = user_input

    # Hiển thị user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tổng hợp câu trả lời..."):
            try:
                result = generate_fn(augmented_query, top_k=top_k)
                answer = result.get("answer", "Không thể tạo câu trả lời.")
                sources = result.get("sources", [])
                retrieval_source = result.get("retrieval_source", "hybrid")
            except Exception as e:
                answer = f"❌ Lỗi pipeline: {e}"
                sources = []
                retrieval_source = "error"

        st.markdown(answer)

        if show_sources and sources:
            render_sources(sources, expanded=True)
        if show_retrieval_info:
            render_retrieval_badge(retrieval_source)

    # Lưu vào history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })
