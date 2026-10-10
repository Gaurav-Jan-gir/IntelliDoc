"""IntelliDoc Streamlit UI - AI-Powered Document Q&A Platform."""

import os
import requests
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="IntelliDoc - AI Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean, professional look
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-online {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .status-offline {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .meta-tag {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Backend API Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def check_api_health():
    """Verify backend API connectivity and retrieve server stats."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def fetch_documents():
    """Fetch all indexed documents from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception as e:
        st.sidebar.error(f"Error fetching documents: {e}")
    return []


def upload_pdf(uploaded_file):
    """Upload a PDF file to the backend."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=60)
        return response
    except Exception as e:
        st.error(f"Network error during upload: {e}")
        return None


def delete_pdf(doc_id: int):
    """Delete a document by ID."""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}", timeout=10)
        return response
    except Exception as e:
        st.error(f"Network error during deletion: {e}")
        return None


def query_backend(question: str):
    """Send user query to backend RAG pipeline."""
    try:
        payload = {"question": question}
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
        return response
    except Exception as e:
        st.error(f"Network error during query: {e}")
        return None


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= SIDEBAR =================
with st.sidebar:
    st.title("📄 IntelliDoc")
    st.caption("AI-Powered Document Intelligence")

    # API Status Indicator
    health_data = check_api_health()
    if health_data:
        st.markdown(
            f'<span class="status-badge status-online">● Backend Online</span> '
            f'<small>({health_data.get("total_documents", 0)} docs, '
            f'{health_data.get("total_vectors", 0)} vectors)</small>',
            unsafe_allow_html=True,
        )
        if not health_data.get("gemini_configured", False):
            st.warning("⚠️ `GOOGLE_API_KEY` not set. Responses will show retrieved passages.")
    else:
        st.markdown(
            '<span class="status-badge status-offline">● Backend Offline</span> '
            f'<br><small>Ensure FastAPI is running at {API_BASE_URL}</small>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Document Upload Section
    st.subheader("📤 Upload Document")
    pdf_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="Upload a PDF file to chunk, embed, and query.",
    )

    if pdf_file is not None:
        if st.button("Process & Index PDF", use_container_width=True, type="primary"):
            with st.spinner("Extracting text, generating embeddings, and building index..."):
                resp = upload_pdf(pdf_file)
                if resp is not None:
                    if resp.status_code == 201:
                        data = resp.json()
                        st.success(data.get("message", "Document uploaded successfully!"))
                        st.rerun()
                    else:
                        error_detail = (
                            resp.json().get("detail", "Failed to upload document")
                            if resp.headers.get("content-type") == "application/json"
                            else resp.text
                        )
                        st.error(f"Upload failed: {error_detail}")

    st.markdown("---")

    # Uploaded Documents Management Section
    st.subheader("📚 Indexed Documents")
    docs = fetch_documents()

    if not docs:
        st.info("No documents indexed yet. Upload a PDF above to get started!")
    else:
        for doc in docs:
            with st.expander(f"📄 {doc['filename']}", expanded=False):
                st.write(f"**Chunks:** {doc['num_chunks']}")
                st.write(f"**Uploaded:** {doc['upload_date'][:19]}")
                st.write(f"**Hash:** `{doc['file_hash'][:8]}...`")
                if st.button("🗑️ Delete Document", key=f"del_{doc['id']}", use_container_width=True):
                    with st.spinner("Deleting document and vector records..."):
                        del_resp = delete_pdf(doc["id"])
                        if del_resp and del_resp.status_code == 200:
                            st.success("Document deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete document.")

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ================= MAIN AREA =================
st.markdown('<div class="main-header">IntelliDoc: Document Q&A Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions across your uploaded PDF documents with vector similarity retrieval and citation-grounded AI answers.</div>',
    unsafe_allow_html=True,
)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 View Retrieved Sources & Citations", expanded=False):
                for i, src in enumerate(msg["sources"], start=1):
                    page_info = f" | Page: {src.get('page')}" if src.get("page") else ""
                    score_info = f" | Distance: {src.get('score'):.4f}" if src.get("score") is not None else ""
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <div class="meta-tag">Citation #{i} — <strong>{src.get('filename')}</strong>{page_info}{score_info}</div>
                            <p style="margin-top: 0.3rem;">{src.get('content')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        if msg.get("processing_time") is not None:
            st.caption(f"⏱️ Response generated in {msg['processing_time']}s")

# User Question Input
if user_question := st.chat_input("Ask a question about your uploaded documents..."):
    # Append and render user query
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Process query through RAG backend
    with st.chat_message("assistant"):
        with st.spinner("Searching document index and synthesizing answer..."):
            query_response = query_backend(user_question)

            if query_response and query_response.status_code == 200:
                data = query_response.json()
                answer_text = data.get("answer", "No answer received.")
                sources_list = data.get("sources", [])
                proc_time = data.get("processing_time", 0.0)

                st.markdown(answer_text)

                if sources_list:
                    with st.expander("📚 View Retrieved Sources & Citations", expanded=False):
                        for i, src in enumerate(sources_list, start=1):
                            page_info = f" | Page: {src.get('page')}" if src.get("page") else ""
                            score_info = f" | Distance: {src.get('score'):.4f}" if src.get("score") is not None else ""
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <div class="meta-tag">Citation #{i} — <strong>{src.get('filename')}</strong>{page_info}{score_info}</div>
                                    <p style="margin-top: 0.3rem;">{src.get('content')}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.caption(f"⏱️ Response generated in {proc_time}s")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_list,
                        "processing_time": proc_time,
                    }
                )
            else:
                err_msg = "Could not communicate with the backend API. Please make sure the backend service is running."
                if query_response:
                    try:
                        err_msg = query_response.json().get("detail", query_response.text)
                    except Exception:
                        err_msg = query_response.text
                st.error(f"Error: {err_msg}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"⚠️ Error processing query: {err_msg}",
                    }
                )

