import os
import re
import time
import json
import csv
from pathlib import Path
import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from paths import uploads_dir
from rag import RAG
from llm_client import chat_stream

st.set_page_config(
    page_title="LocalHelper AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "rag" not in st.session_state:
    rag = RAG()
    try:
        rag.load()
    except Exception:
        pass
    st.session_state.rag = rag

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {"topic": "", "qnum": 1, "pending_key": None, "pending_question": ""}

ANSWER_KEY_RE = re.compile(r"(?im)^\s*ANSWER_KEY\s*:\s*([ABCD])\s*$")

def extract_flashcards(text: str):
    cards = []
    for part in re.split(r"(?i)\**FRONT\**\s*:\s*", str(text))[1:]:
        sub = re.split(r"(?i)\**BACK\**\s*:\s*", part, maxsplit=1)
        if len(sub) == 2:
            f = sub[0].strip().strip("*# \n")
            b = re.sub(r"\n\s*---+.*", "", sub[1], flags=re.DOTALL).strip().strip("*# \n")
            if f and b:
                cards.append({"front": f, "back": b})
    return cards

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 LocalHelper AI")
    st.caption("Cloud Study Assistant · Hybrid RAG + Llama 3.2 3B")
    st.markdown("---")

    st.subheader("⚙️ Study Modes")
    study_mode = st.selectbox("Mode", ["Explain", "Summarize", "Make MCQs", "Flashcards", "Quiz me"], index=0)
    level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"], index=1)

    st.markdown("---")
    st.subheader("🎛️ Parameters")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.4, 0.05)
    max_tokens = st.slider("Max Tokens", 128, 2048, 800, 64)
    top_k = st.slider("Top-K Chunks", 1, 8, 3, 1)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.quiz_state = {"topic": "", "qnum": 1, "pending_key": None, "pending_question": ""}
        st.rerun()

# --- MAIN TABS ---
tab_chat, tab_lib, tab_cards, tab_about = st.tabs(["💬 Chat", "📚 Library", "🃏 Flashcards", "ℹ️ About"])

# --- TAB 1: CHAT ---
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your study documents...")
    if user_input:
        if not st.session_state.rag.chunks:
            st.warning("⚠️ No documents indexed yet. Upload documents in the **📚 Library** tab first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_resp = ""

                # Quiz Mode
                if study_mode == "Quiz me":
                    qs = st.session_state.quiz_state
                    if qs.get("pending_key"):
                        ans = user_input.strip().upper()
                        if ans not in ("A", "B", "C", "D"):
                            full_resp = "Please answer with **A**, **B**, **C**, or **D**."
                            placeholder.markdown(full_resp)
                        else:
                            verdict = "✅ **Correct!**" if ans == qs["pending_key"] else f"❌ **Incorrect.** Correct answer: **{qs['pending_key']}**"
                            full_resp = verdict + "\n\n_Generating next question..._\n\n"
                            placeholder.markdown(full_resp)

                            ret = st.session_state.rag.retrieve(qs["topic"], top_k=int(top_k))
                            ctx = "\n\n".join([f"[S{i+1}] {c.text}" for i, c in enumerate(ret)])
                            sys_p = "Strict quiz evaluator. Feedback + Next MCQ A/B/C/D. End with hidden: ANSWER_KEY: X"
                            usr_p = f"CONTEXT:\n{ctx}\n\nTOPIC: {qs['topic']}\n\nNext Q{qs['qnum']+1} with A/B/C/D.\n\nANSWER_KEY: X"

                            for chunk in chat_stream([{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}], temperature=0.3, max_tokens=int(max_tokens)):
                                full_resp += chunk
                                placeholder.markdown(full_resp)

                            m = ANSWER_KEY_RE.search(full_resp)
                            st.session_state.quiz_state = {
                                "topic": qs["topic"],
                                "qnum": qs["qnum"] + 1,
                                "pending_key": m.group(1) if m else None,
                                "pending_question": ANSWER_KEY_RE.sub("", full_resp).rstrip(),
                            }
                            full_resp = ANSWER_KEY_RE.sub("", full_resp).rstrip()
                            placeholder.markdown(full_resp)
                    else:
                        st.session_state.quiz_state = {"topic": user_input, "qnum": 1, "pending_key": None, "pending_question": ""}
                        ret = st.session_state.rag.retrieve(user_input, top_k=int(top_k))
                        ctx = "\n\n".join([f"[S{i+1}] {c.text}" for i, c in enumerate(ret)])
                        sys_p = "Strict quiz generator. Create ONE MCQ A/B/C/D based on CONTEXT. End with hidden line: ANSWER_KEY: X"
                        usr_p = f"CONTEXT:\n{ctx}\n\nTOPIC: {user_input}\n\nGenerate Q1.\n\nANSWER_KEY: X"

                        for chunk in chat_stream([{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}], temperature=0.3, max_tokens=int(max_tokens)):
                            full_resp += chunk
                            placeholder.markdown(full_resp)

                        m = ANSWER_KEY_RE.search(full_resp)
                        st.session_state.quiz_state["pending_key"] = m.group(1) if m else None
                        full_resp = ANSWER_KEY_RE.sub("", full_resp).rstrip()
                        placeholder.markdown(full_resp)

                # Standard Modes
                else:
                    t0 = time.perf_counter()
                    retrieved = st.session_state.rag.retrieve(user_input, top_k=int(top_k))
                    ret_ms = (time.perf_counter() - t0) * 1000

                    context = "\n\n".join([f"[S{i+1}] {c.source_name} ({c.loc_kind} {c.loc})\n{c.text}" for i, c in enumerate(retrieved)]) or "(no context)"
                    modes = {
                        "Summarize": "Provide a title, 6-12 bullets, 3 definitions, 3 exam questions.",
                        "Make MCQs": "Generate 10 MCQs with A/B/C/D options and citations [S#].",
                        "Flashcards": "Generate 5+ flashcards in EXACT format:\nFRONT: [concept]\nBACK: [answer] [S#]",
                        "Explain": "Explain clearly using structured headings and bullets."
                    }
                    system = f"You are a study assistant. Use ONLY context. Cite as [S#]. Difficulty: {level}. Task: {modes.get(study_mode, modes['Explain'])}"
                    msgs = [{"role": "system", "content": system}, {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {user_input}"}]

                    for chunk in chat_stream(msgs, temperature=float(temperature), max_tokens=int(max_tokens)):
                        full_resp += chunk
                        placeholder.markdown(full_resp)

                    if retrieved:
                        refs = " · ".join([f"**[S{i+1}]** {c.source_name} ({c.loc_kind} {c.loc})" for i, c in enumerate(retrieved)])
                        full_resp += f"\n\n---\n**References:** {refs}\n\n*⏱ Retrieval: {ret_ms:.0f} ms*"
                        placeholder.markdown(full_resp)

                    if study_mode == "Flashcards":
                        new_cards = extract_flashcards(full_resp)
                        if new_cards:
                            st.session_state.flashcards.extend(new_cards)
                            st.toast(f"Saved {len(new_cards)} flashcards!", icon="🃏")

                st.session_state.messages.append({"role": "assistant", "content": full_resp})

# --- TAB 2: LIBRARY ---
with tab_lib:
    st.subheader("📁 Upload Study Materials")
    st.write("Supported formats: **PDF, DOCX, PPTX, TXT, MD**")
    uploaded = st.file_uploader("Select files", type=["pdf", "docx", "pptx", "txt", "md"], accept_multiple_files=True)

    col1, col2 = st.columns(2)
    with col1:
        chunk_sz = st.slider("Chunk Size", 500, 3000, 1500, 100)
    with col2:
        overlap_sz = st.slider("Overlap", 0, 500, 100, 50)

    if st.button("🔨 Build / Rebuild Vector Index", type="primary", use_container_width=True):
        if not uploaded:
            st.error("Please select at least one document.")
        else:
            with st.spinner("Extracting text and computing ONNX embeddings..."):
                paths = []
                dest = uploads_dir()
                for file in uploaded:
                    p = dest / file.name
                    with open(p, "wb") as f:
                        f.write(file.getbuffer())
                    paths.append(str(p))
                    if file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(file.name)

                n_f, n_c = st.session_state.rag.build(paths, int(chunk_sz), int(overlap_sz))
                st.success(f"✅ Indexed **{n_f}** file(s) into **{n_c}** searchable chunks.")

    st.markdown("---")
    st.subheader("📑 Indexed Documents")
    if st.session_state.uploaded_files:
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- 📄 `{fname}`")
    else:
        st.info("No documents uploaded yet.")

# --- TAB 3: FLASHCARDS ---
with tab_cards:
    st.subheader("🃏 Study Flashcards Deck")
    cards = st.session_state.flashcards

    if not cards:
        st.info("No flashcards yet. Switch Study Mode to **Flashcards** in the Chat tab and ask about any topic!")
    else:
        st.write(f"**Total cards in deck:** {len(cards)}")
        if st.button("🗑️ Clear Deck"):
            st.session_state.flashcards = []
            st.rerun()

        st.markdown("---")
        for i, c in enumerate(cards, 1):
            with st.expander(f"🎴 Card {i}: {c['front'][:60]}..."):
                st.markdown(f"**FRONT (Question):**\n\n{c['front']}")
                st.markdown("---")
                st.markdown(f"**BACK (Answer):**\n\n{c['back']}")

        st.markdown("---")
        st.subheader("💾 Export Deck")
        col_md, col_csv, col_json = st.columns(3)

        with col_md:
            md_t = f"# 📚 Flashcards\n\n" + "".join([f"### Card {i}\n**Front:** {c['front']}\n**Back:** {c['back']}\n\n---\n" for i, c in enumerate(cards, 1)])
            st.download_button("📥 Markdown (.md)", data=md_t, file_name="flashcards.md", mime="text/markdown")

        with col_csv:
            import io
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["Front", "Back"])
            for c in cards:
                w.writerow([c["front"], c["back"]])
            st.download_button("📥 CSV for Anki (.csv)", data=buf.getvalue(), file_name="flashcards.csv", mime="text/csv")

        with col_json:
            st.download_button("📥 JSON (.json)", data=json.dumps(cards, indent=2), file_name="flashcards.json", mime="application/json")

# --- TAB 4: ABOUT ---
with tab_about:
    st.subheader("ℹ️ System Architecture & Specifications")
    st.markdown("""
    **LocalHelper AI** is a cloud-hosted academic study assistant.

    | Layer | Technology |
    |---|---|
    | **Frontend UI** | Streamlit 1.40 (Streamlit Cloud) |
    | **Retrieval Engine** | Hybrid HNSW Vector Search + BM25 Keyword Fusion |
    | **Vector Embeddings** | BAAI/bge-small-en-v1.5 (ONNX Runtime) |
    | **Language Model** | Meta Llama 3.2 3B Instruct via Hugging Face Serverless API |
    | **Document Ingestion** | PyPDF, python-docx, python-pptx |

    ---
    *Developed as a Computer Engineering College Project.*
    """)