"""
PDF RAG 챗봇 (Streamlit) — Session 04 실습
============================================
사용법:
    1) cp .env.example .env  → OPENAI_API_KEY 입력
    2) pip install -r requirements.txt
    3) streamlit run app.py

기능:
    - PDF 여러 개 업로드 → 자동 인덱싱 (Chroma)
    - 대화 이력 유지 (멀티턴)
    - 스트리밍 응답 + 출처 표시
    - 모델·청크 크기 사이드바에서 조절
    - 대화 초기화 / DB 리셋 버튼
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# === LangChain 임포트 ============================================
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


# === 0. 환경 설정 ================================================
load_dotenv(override=True)  # 셸에 남은 낡은 OPENAI_API_KEY 보다 .env 를 우선한다

st.set_page_config(
    page_title="📚 PDF RAG 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# === 1. 헬퍼 함수 =================================================
def get_api_key() -> str | None:
    """API 키를 환경변수 또는 Streamlit secrets에서 가져옴."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            key = None
    if key and key.startswith("sk-your"):
        return None  # placeholder
    return key


@st.cache_resource(show_spinner="📚 PDF 인덱싱 중... (최초 1회만)")
def build_vectorstore(
    file_bytes_list: list[tuple[str, bytes]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Chroma | None:
    """업로드된 PDF 바이트들을 인덱싱하여 Chroma 반환.

    Streamlit @st.cache_resource로 같은 입력엔 1회만 실행.
    file_bytes_list: [(filename, bytes), ...] — 캐시 키로 쓰임
    """
    if not file_bytes_list:
        return None

    # 1) PDF 로드 (임시 파일 경유 — PyPDFLoader는 경로 필요)
    all_docs: list[Document] = []
    for filename, file_bytes in file_bytes_list:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
            # 메타데이터에 원본 파일명 주입
            for p in pages:
                p.metadata["source"] = filename
            all_docs.extend(pages)
        finally:
            os.unlink(tmp_path)

    # 2) 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    # 3) Chroma — in-memory (앱 재시작 시 휘발)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="pdf_rag_session",
    )
    return vectorstore


def build_rag_chain(vectorstore: Chroma, model_name: str, top_k: int):
    """RAG LCEL 체인 — retriever | prompt | llm | parser."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    llm = ChatOpenAI(model=model_name, temperature=0, streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 PDF 문서를 참고해 정확하게 답하는 어시스턴트입니다.\n"
         "주어진 문서에서 근거를 찾아 답하세요.\n"
         "문서에 없는 내용은 \"제공된 문서에서 찾을 수 없습니다\" 라고 답하세요.\n"
         "한국어로 답하세요."),
        ("human",
         "참고 문서:\n{context}\n\n질문: {question}"),
    ])

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[문서 {i+1}] {d.page_content}"
            for i, d in enumerate(docs)
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def render_sources_expander(retrieved_docs: list[Document]) -> list[str]:
    """검색된 문서를 expander로 표시하고 인용 문자열 반환."""
    citations: list[str] = []
    with st.expander(f"📚 출처 보기 (Top-{len(retrieved_docs)})"):
        for i, doc in enumerate(retrieved_docs, 1):
            src = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:250].replace("\n", " ")
            st.markdown(f"**[{i}]** `{src}` · p.{page}")
            st.caption(preview + ("..." if len(doc.page_content) > 250 else ""))
            citations.append(f"{src} p.{page}")
    return citations


# === 2. UI ========================================================
st.title("📚 PDF RAG 챗봇")
st.caption("PDF를 업로드하고 자연어로 질문하세요. LangChain + OpenAI + Chroma 기반.")

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")

    # API 키 확인
    api_key = get_api_key()
    if not api_key:
        st.error("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        st.markdown(
            ".env 파일에 `OPENAI_API_KEY=sk-...` 추가 또는\n"
            "Streamlit Cloud의 Secrets에 등록해주세요."
        )
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    st.success(f"✅ API 키 OK (`{api_key[:8]}...`)")

    st.divider()

    # PDF 업로드
    st.subheader("📄 PDF 업로드")
    uploaded_files = st.file_uploader(
        "PDF 파일을 선택하세요",
        type=["pdf"],
        accept_multiple_files=True,
        help="여러 PDF를 동시에 업로드할 수 있습니다",
    )

    st.divider()

    # 모델 / 검색 설정
    st.subheader("🛠️ 모델 / 검색")
    model_name = st.selectbox(
        "LLM 모델",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
        help="gpt-4o-mini가 가장 저렴 (Top 추천)",
    )
    chunk_size = st.slider("청크 크기 (자)", 200, 2000, 500, 100)
    chunk_overlap = st.slider("청크 오버랩 (자)", 0, 200, 50, 10)
    top_k = st.slider("검색 Top-K", 1, 10, 3)

    st.divider()

    # 컨트롤
    st.subheader("🔧 컨트롤")
    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.rerun()
    if st.button("♻️ 인덱스 재구축"):
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("💡 Tip: 청크 크기/Top-K를 바꾸면 답변 품질이 달라집니다.")


# --- 메인: 인덱싱 + 채팅 ---

# 인덱싱 — 업로드된 파일 바이트로 vectorstore 만들기
vectorstore: Chroma | None = None
if uploaded_files:
    # bytes 추출 (캐시 키용 — 파일명과 바이트가 같으면 캐시 히트)
    file_bytes = [(f.name, f.getvalue()) for f in uploaded_files]
    vectorstore = build_vectorstore(file_bytes, chunk_size, chunk_overlap)
    if vectorstore:
        n_docs = vectorstore._collection.count()
        st.sidebar.success(f"✅ 인덱스 준비 완료 — 청크 {n_docs}개")

# 채팅 영역
if not vectorstore:
    st.info("👈 왼쪽 사이드바에서 PDF를 먼저 업로드해주세요.")
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 이력 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 출처"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# 사용자 입력 + 응답
if user_q := st.chat_input("PDF에 대해 물어보세요…"):
    # 1) 사용자 메시지 저장 + 표시
    st.session_state.messages.append({
        "role": "user",
        "content": user_q,
    })
    with st.chat_message("user"):
        st.markdown(user_q)

    # 2) 어시스턴트 답변 — 스트리밍
    with st.chat_message("assistant"):
        # 체인 + retriever 생성
        chain, retriever = build_rag_chain(
            vectorstore, model_name, top_k
        )

        # 먼저 검색해서 출처 확보 (스트리밍 전에)
        retrieved = retriever.invoke(user_q)

        # 스트리밍 응답 — st.write_stream이 generator를 받아 토큰 단위 렌더링
        try:
            answer = st.write_stream(chain.stream(user_q))
        except Exception as e:
            st.error(f"❌ 오류: {e}")
            answer = ""

        # 출처 표시
        citations = render_sources_expander(retrieved)

        # 메시지 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": citations,
        })
