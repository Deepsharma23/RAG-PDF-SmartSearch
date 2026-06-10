import html
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, field

import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from utils import groq_llm, huggingface_instruct_embedding

warnings.filterwarnings("ignore")

MAX_FILE_SIZE_MB = 20

PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful document question-answering assistant.

Answer the question using ONLY the information from the provided context.

If the answer is not in the context, say: "I could not find this information in the uploaded documents."

Do not use any outside knowledge.

<context>
{context}
</context>

Question: {input}

Answer:
""",
)


@dataclass
class ProcessingResult:
    success: bool
    error: str | None = None
    chunk_count: int = 0
    loaded_files: list[str] = field(default_factory=list)


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]
    response_time: float


class StoredUploadedFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.size = len(data)
        self._data = data

    def getbuffer(self):
        return memoryview(self._data)


def inject_styles() -> None:
    page_bg = "#0B0F19"
    sidebar_bg = "#111827"
    card_bg = "#151C2C"
    input_bg = "#1B2232"
    border = "#273244"
    text_primary = "#F8FAFC"
    text_secondary = "#94A3B8"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        #MainMenu, footer, header {{
            visibility: hidden;
        }}

        .stApp {{
            background-color: {page_bg} !important;
            color: {text_primary} !important;
        }}
        
        /* Force all text elements to use our colors */
        .stApp * {{
            color: {text_primary} !important;
        }}
        
        .stApp .dq-status-badge,
        .stApp .dq-message-user .dq-bubble {{
            color: white !important;
        }}
        
        .stApp .dq-status-badge.dq-status-ready {{
            color: #22C55E !important;
        }}
        
        .stApp .dq-status-badge.dq-status-pending {{
            color: #F59E0B !important;
        }}
        
        .stApp .dq-status-badge.dq-status-processing {{
            color: #6366F1 !important;
        }}
        
        .stApp .dq-status-badge.dq-status-error {{
            color: #EF4444 !important;
        }}
        
        .stApp .dq-status-badge.dq-status-empty {{
            color: {text_secondary} !important;
        }}
        
        .stApp .dq-subtitle,
        .stApp .dq-upload-hint,
        .stApp .dq-file-meta,
        .stApp .dq-source-label,
        .stApp .dq-empty-desc {{
            color: {text_secondary} !important;
        }}
        
        .block-container {{
            max-width: 1500px;
            padding-top: 24px;
            padding-bottom: 32px;
            padding-left: 32px;
            padding-right: 32px;
        }}

        .dq-navbar {{
            height: 64px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            border-bottom: 1px solid {border};
            background: {sidebar_bg} !important;
            margin-bottom: 0;
        }}

        .dq-brand {{
            min-height: 46px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .dq-logo {{
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: #6366F1;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white !important;
            font-weight: 700;
            font-size: 14px;
        }}

        .dq-brand-name {{
            font-size: 16px;
            font-weight: 600;
            color: {text_primary} !important;
        }}


        .dq-section-title {{
            font-size: 18px;
            font-weight: 600;
            color: {text_primary} !important;
            margin: 0 0 8px 0;
        }}

        .dq-subtitle {{
            font-size: 13px;
            color: {text_secondary} !important;
            margin: 0;
        }}

        .dq-upload-zone {{
            border: 1.5px dashed {border};
            border-radius: 16px;
            background: {card_bg} !important;
            padding: 24px 16px;
            text-align: center;
            transition: border-color 0.2s, background 0.2s;
        }}

        .dq-upload-zone:hover {{
            border-color: #6366F1;
            background: #1A2030 !important;
        }}

        .dq-upload-title {{
            font-size: 14px;
            font-weight: 500;
            color: {text_primary} !important;
            margin-bottom: 4px;
        }}

        .dq-upload-hint {{
            font-size: 13px;
            color: {text_secondary} !important;
            margin-top: 8px;
        }}

        .dq-file-card {{
            background: {card_bg} !important;
            border: 1px solid {border};
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}

        .dq-file-content {{
            flex-grow: 1;
            margin-right: 12px;
        }}

        .dq-file-name {{
            font-size: 14px;
            font-weight: 500;
            color: {text_primary} !important;
            margin: 0;
        }}

        .dq-file-meta {{
            font-size: 13px;
            color: {text_secondary} !important;
            margin: 4px 0 0 0;
        }}

        .dq-remove-btn {{
            background: transparent !important;
            border: none !important;
            color: {text_secondary} !important;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            padding: 0 4px;
            line-height: 1;
            border-radius: 4px;
        }}

        .dq-remove-btn:hover {{
            color: #EF4444 !important;
            background: rgba(239, 68, 68, 0.1) !important;
        }}

        .dq-status-badge {{
            display: inline-block;
            font-size: 12px;
            font-weight: 500;
            padding: 4px 8px;
            border-radius: 999px;
            margin-top: 8px;
        }}

        .dq-status-ready {{
            background: rgba(34, 197, 94, 0.15);
            color: #22C55E !important;
        }}

        .dq-status-pending {{
            background: rgba(245, 158, 11, 0.15);
            color: #F59E0B !important;
        }}

        .dq-status-processing {{
            background: rgba(99, 102, 241, 0.15);
            color: #6366F1 !important;
        }}

        .dq-status-error {{
            background: rgba(239, 68, 68, 0.15);
            color: #EF4444 !important;
        }}

        .dq-status-empty {{
            background: rgba(148, 163, 184, 0.15);
            color: {text_secondary} !important;
        }}

        .dq-chat-header {{
            margin-bottom: 24px;
        }}

        .dq-chat-title {{
            font-size: 30px;
            font-weight: 700;
            color: {text_primary} !important;
            margin: 0 0 8px 0;
        }}

        .dq-empty-state {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 48px 24px;
        }}

        .dq-empty-title {{
            font-size: 24px;
            font-weight: 600;
            color: {text_primary} !important;
            margin-bottom: 8px;
        }}

        .dq-empty-desc {{
            font-size: 15px;
            color: {text_secondary} !important;
            max-width: 480px;
            margin-bottom: 24px;
        }}

        .dq-prompt-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            width: 100%;
            max-width: 720px;
        }}

        .dq-message-user {{
            display: flex;
            justify-content: flex-end;
            margin-bottom: 16px;
        }}

        .dq-message-user .dq-bubble {{
            background: #6366F1 !important;
            color: white !important;
            border-radius: 16px 16px 4px 16px;
            padding: 12px 16px;
            max-width: 75%;
            font-size: 15px;
            line-height: 1.5;
        }}

        .dq-message-assistant {{
            display: flex;
            justify-content: flex-start;
            margin-bottom: 16px;
        }}

        .dq-message-assistant .dq-bubble {{
            background: {card_bg} !important;
            border: 1px solid {border};
            color: {text_primary} !important;
            border-radius: 16px 16px 16px 4px;
            padding: 12px 16px;
            max-width: 85%;
            font-size: 15px;
            line-height: 1.6;
        }}

        .dq-source-label {{
            font-size: 13px;
            color: {text_secondary} !important;
            margin-top: 12px;
        }}

        /* Hide default Streamlit file uploader styling */
        div[data-testid="stFileUploader"] {{
            background: transparent;
            padding: 0;
        }}

        div[data-testid="stFileUploader"] section {{
            border: none;
            padding: 0;
        }}

        div[data-testid="stFileUploader"] section > div {{
            border: none;
            padding: 0;
        }}

        div[data-testid="stFileUploader"] small {{
            display: none;
        }}

        /* Hide the file uploader's file list area */
        div[data-testid="stFileUploader"] > div:first-child {{
            display: none;
        }}

        div[data-testid="stFileUploader"] button {{
            background: {input_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {border};
            border-radius: 10px;
            font-size: 14px;
            font-weight: 500;
        }}

        div[data-testid="stFileUploader"] button:hover {{
            border-color: #6366F1;
            color: #6366F1 !important;
        }}

        /* Default button style */
        .stButton > button {{
            border-radius: 10px;
            font-size: 14px;
            font-weight: 500;
            border: 1px solid {border} !important;
            background: {input_bg} !important;
            color: {text_primary} !important;
        }}

        /* Small remove button styling */
        [data-testid="column"] [data-testid="stButton"] > button {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0;
            min-height: 36px;
            min-width: 36px;
            font-size: 24px;
            font-weight: 700;
            line-height: 1;
        }}

        .stButton > button:hover {{
            border-color: #6366F1 !important;
            color: #6366F1 !important;
        }}

        div[data-testid="stButton"][data-dq-primary="true"] > button {{
            background: #6366F1 !important;
            color: white !important;
            border: none;
        }}

        div[data-testid="stButton"][data-dq-primary="true"] > button:hover {{
            background: #4F46E5 !important;
            color: white !important;
        }}

        div[data-testid="stButton"][data-dq-primary="true"] > button:disabled {{
            background: #3F3F5C !important;
            color: #94A3B8 !important;
        }}

        .stTextInput > div > div > input,
        .stChatInput textarea {{
            background: {input_bg} !important;
            border: 1px solid {border} !important;
            border-radius: 12px;
            color: {text_primary} !important;
            font-size: 15px;
        }}
        
        .stChatInput button {{
            background: #6366F1 !important;
            color: white !important;
        }}

        .stExpander {{
            background: {card_bg} !important;
            border: 1px solid {border} !important;
            border-radius: 10px;
        }}

        @media (max-width: 900px) {{
            .dq-layout {{
                flex-direction: column;
            }}

            .dq-sidebar {{
                width: 100%;
                min-width: 0;
                border-right: none;
                border-bottom: 1px solid {border};
            }}

            .dq-prompt-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "chat_messages": [],
        "document_files": {},
        "processing_status": "not_uploaded",
        "process_error": None,
        "processed_file_names": [],
        "removed_files": [],
        "uploader_key": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_document_state() -> None:
    keys_to_clear = [
        "embeddings",
        "docs",
        "final_documents",
        "vectors",
        "processed_file_names",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    st.session_state.processing_status = "not_uploaded"
    st.session_state.process_error = None


def clear_chat() -> None:
    st.session_state.chat_messages = []
    st.session_state.removed_files = []


def format_file_size(size_bytes: int) -> str:
    size_mb = size_bytes / (1024 * 1024)

    if size_mb < 1:
        return f"{size_bytes / 1024:.1f} KB"

    return f"{size_mb:.2f} MB"


def get_stored_files() -> list[StoredUploadedFile]:
    return [
        StoredUploadedFile(name=data["name"], data=data["bytes"])
        for data in st.session_state.document_files.values()
    ]


def sync_uploaded_files(uploaded_files) -> None:
    if not uploaded_files:
        if not st.session_state.document_files:
            st.session_state.processing_status = "not_uploaded"
        return

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.document_files:
            continue
        
        # If user is re-uploading a previously removed file, remove it from removed_files
        if uploaded_file.name in st.session_state.removed_files:
            st.session_state.removed_files.remove(uploaded_file.name)

        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.session_state.process_error = (
                f"{uploaded_file.name} exceeds the "
                f"{MAX_FILE_SIZE_MB} MB limit."
            )
            continue

        st.session_state.document_files[uploaded_file.name] = {
            "name": uploaded_file.name,
            "bytes": uploaded_file.getvalue(),
            "size": uploaded_file.size,
        }

    if st.session_state.document_files:
        current_names = set(st.session_state.document_files.keys())
        processed_names = set(
            st.session_state.get("processed_file_names", [])
        )

        if (
            st.session_state.processing_status == "ready"
            and current_names != processed_names
        ):
            clear_document_state()
            st.session_state.processing_status = "uploaded"
        elif st.session_state.processing_status == "not_uploaded":
            st.session_state.processing_status = "uploaded"
        elif (
            "vectors" not in st.session_state
            and st.session_state.processing_status != "processing"
        ):
            st.session_state.processing_status = "uploaded"


def remove_document(file_name: str) -> None:
    st.session_state.document_files.pop(file_name, None)
    st.session_state.removed_files.append(file_name)
    st.session_state.uploader_key += 1
    clear_document_state()
    st.session_state.chat_messages = []

    if st.session_state.document_files:
        st.session_state.processing_status = "uploaded"
    else:
        st.session_state.processing_status = "not_uploaded"


def vector_embedding(uploaded_files: list[StoredUploadedFile]) -> ProcessingResult:
    if not uploaded_files:
        return ProcessingResult(
            success=False,
            error="Please upload at least one PDF.",
        )

    docs = []

    try:
        st.session_state.embeddings = huggingface_instruct_embedding()
    except Exception as error:
        return ProcessingResult(
            success=False,
            error=f"Could not load the embedding model: {error}",
        )

    loaded_files = []

    for uploaded_file in uploaded_files:
        temporary_file_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temporary_file:
                temporary_file.write(uploaded_file.getbuffer())
                temporary_file_path = temporary_file.name

            loader = PyPDFLoader(temporary_file_path)
            loaded_documents = loader.load()

            for document in loaded_documents:
                document.metadata["source"] = uploaded_file.name

            docs.extend(loaded_documents)
            loaded_files.append(uploaded_file.name)

        except Exception as error:
            return ProcessingResult(
                success=False,
                error=f"Could not load {uploaded_file.name}: {error}",
            )

        finally:
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)

    if not docs:
        return ProcessingResult(
            success=False,
            error="No valid PDF documents were loaded.",
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    final_documents = text_splitter.split_documents(docs)

    if not final_documents:
        return ProcessingResult(
            success=False,
            error="The uploaded documents did not contain readable text.",
        )

    st.session_state.docs = docs
    st.session_state.final_documents = final_documents
    st.session_state.processed_file_names = loaded_files

    try:
        st.session_state.vectors = FAISS.from_documents(
            final_documents,
            st.session_state.embeddings,
        )
    except Exception as error:
        return ProcessingResult(
            success=False,
            error=f"Could not prepare your documents: {error}",
        )

    return ProcessingResult(
        success=True,
        chunk_count=len(final_documents),
        loaded_files=loaded_files,
    )


def answer_question(user_input: str) -> AnswerResult:
    llm = groq_llm()
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 5}
    )
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start_time = time.perf_counter()
    response = retrieval_chain.invoke({"input": user_input})
    response_time = time.perf_counter() - start_time

    relevant_documents = response.get("context", [])
    sources = []

    for document in relevant_documents:
        page_number = document.metadata.get("page")
        sources.append(
            {
                "source": document.metadata.get(
                    "source",
                    "Unknown document",
                ),
                "page": page_number + 1 if page_number is not None else None,
                "excerpt": document.page_content,
            }
        )

    return AnswerResult(
        answer=response.get("answer", "No answer was generated."),
        sources=sources,
        response_time=response_time,
    )


def get_file_status_label(file_name: str) -> tuple[str, str]:
    status = st.session_state.processing_status

    if status == "processing":
        return "Preparing document...", "dq-status-processing"

    if status == "error":
        return "Document processing failed", "dq-status-error"

    if (
        status == "ready"
        and file_name in st.session_state.get("processed_file_names", [])
    ):
        return "Ready for questions", "dq-status-ready"

    if file_name in st.session_state.document_files:
        return "Ready to prepare", "dq-status-pending"

    return "No documents uploaded", "dq-status-empty"


def render_navbar() -> None:
    brand_col, spacer_col, new_chat_col = st.columns(
        [5, 4.2, 1.2]
    )

    with brand_col:
        st.markdown(
            """
            <div class="dq-brand">
                <div class="dq-logo">DQ</div>
                <span class="dq-brand-name">DocuQuery</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with new_chat_col:
        if st.button(
            "New Chat",
            key="nav_new_chat",
            use_container_width=True,
        ):
            clear_chat()
            st.rerun()

    st.divider()




def render_sidebar() -> None:
    st.markdown(
        '<p class="dq-section-title">Your documents</p>',
        unsafe_allow_html=True,
    )

    

    uploaded_files = st.file_uploader(
        "Browse Files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    st.markdown(
        f'<p class="dq-upload-hint">PDF files up to {MAX_FILE_SIZE_MB} MB</p>',
        unsafe_allow_html=True,
    )

    sync_uploaded_files(uploaded_files)

    if st.session_state.document_files:
        for file_name, file_data in st.session_state.document_files.items():
            status_text, status_class = get_file_status_label(file_name)

            # Create two columns for file content and remove button
            col1, col2 = st.columns([8, 1])
            with col1:
                st.markdown(
                    f"""
                    <div class="dq-file-content">
                        <p class="dq-file-name">📄 {file_name}</p>
                        <p class="dq-file-meta">{format_file_size(file_data["size"])}</p>
                        <span class="dq-status-badge {status_class}">{status_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                # Align vertically with the file content
                st.markdown(
                    """
                    <div style="display: flex; justify-content: flex-end; margin-top: 6px;">
                    """,
                    unsafe_allow_html=True
                )
                # Remove button with custom styling
                remove_key = f"remove_{file_name}"
                if st.button(
                    "×", 
                    key=remove_key, 
                    help=f"Remove {file_name}"
                ):
                    remove_document(file_name)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="dq-file-card">
                <p class="dq-file-name">No documents uploaded</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Add spacing before the Prepare Documents button
    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    is_ready = st.session_state.processing_status == "ready"
    is_processing = st.session_state.processing_status == "processing"
    has_files = bool(st.session_state.document_files)

    prepare_clicked = st.button(
        "Prepare Documents",
        type="primary",
        disabled=not has_files or is_processing,
        use_container_width=True,
        key="prepare_documents",
    )

    if prepare_clicked:
        clear_document_state()
        st.session_state.processing_status = "processing"
        st.session_state.process_error = None

        with st.spinner("Preparing document..."):
            result = vector_embedding(get_stored_files())

        if result.success:
            st.session_state.processing_status = "ready"
            st.session_state.process_error = None
        else:
            st.session_state.processing_status = "error"
            st.session_state.process_error = result.error

        st.rerun()

    if st.session_state.processing_status == "error":
        st.error(
            st.session_state.process_error
            or "Document processing failed"
        )

        if st.button("Retry", key="retry_processing"):
            st.session_state.processing_status = "uploaded"
            st.session_state.process_error = None
            st.rerun()

    elif is_ready:
        st.markdown(
            '<span class="dq-status-badge dq-status-ready">Ready for questions</span>',
            unsafe_allow_html=True,
        )


def render_chat_messages() -> None:
    for message in st.session_state.chat_messages:
        safe_content = html.escape(message["content"]).replace("\n", "<br>")

        if message["role"] == "user":
            st.markdown(
                f"""
                <div class="dq-message-user">
                    <div class="dq-bubble">{safe_content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="dq-message-assistant">
                    <div class="dq-bubble">{safe_content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for index, source in enumerate(message.get("sources", [])):
                page_label = (
                    f", Page {source['page']}"
                    if source.get("page") is not None
                    else ""
                )
                expander_label = f"Source: {source['source']}{page_label}"

                with st.expander(expander_label):
                    st.write(source.get("excerpt", ""))

            st.caption(
                f"Answered in {message.get('response_time', 0):.2f} seconds"
            )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="dq-empty-state">
            <p class="dq-empty-title">Ask questions about your documents</p>
            <p class="dq-empty-desc">
                Upload and prepare a PDF to start asking questions and receive
                answers grounded in its content.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def handle_question_submission(user_input: str) -> None:
    if st.session_state.processing_status != "ready":
        return

    st.session_state.chat_messages.append(
        {"role": "user", "content": user_input}
    )

    try:
        with st.spinner("Searching your documents..."):
            result = answer_question(user_input)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
                "response_time": result.response_time,
            }
        )
    except Exception as error:
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": (
                    "Something went wrong while answering your question. "
                    f"Please try again. ({error})"
                ),
                "sources": [],
                "response_time": 0,
            }
        )


def render_main_chat() -> None:
    st.markdown(
        """
        <div class="dq-chat-header">
            <p class="dq-chat-title">Chat with your documents</p>
            <p class="dq-subtitle">
                Upload PDFs and get answers grounded in your document content.
            </p>
            <p class="dq-section-title" style="margin-top: 24px;">Document assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.processing_status == "ready":
        processed_names = st.session_state.get("processed_file_names", [])
        st.caption(
            f"{len(processed_names)} document(s) ready for questions."
        )
    elif st.session_state.processing_status == "processing":
        st.caption("Preparing document...")
    elif st.session_state.document_files:
        st.caption("Documents uploaded. Click Prepare Documents to continue.")
    else:
        st.caption("No documents uploaded yet.")

    if st.session_state.chat_messages:
        render_chat_messages()
    else:
        render_empty_state()

    documents_ready = st.session_state.processing_status == "ready"

    user_input = st.chat_input(
        "Ask a question about your documents...",
        disabled=not documents_ready,
    )

    if user_input:
        handle_question_submission(user_input)
        st.rerun()

    if not documents_ready:
        st.caption("Prepare your documents before asking questions.")


def main() -> None:
    st.set_page_config(
        page_title="DocuQuery",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_session_state()
    inject_styles()

    render_navbar()

    sidebar_col, main_col = st.columns(
        [1, 3],
        gap="large",
        vertical_alignment="top",
    )

    with sidebar_col:
        render_sidebar()

    with main_col:
        render_main_chat()


if __name__ == "__main__":
    main()
