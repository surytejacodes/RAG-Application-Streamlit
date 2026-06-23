import tempfile, streamlit as st
from services.document_loader import load_pdf, load_txt, chunk_text
from services.bedrock import embed, chat
from services.vector_store import VectorStore

DB_PATH = "data/faiss"

st.set_page_config(page_title="AWS Bedrock RAG Chatbot")
st.title("AWS Bedrock RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

files = st.file_uploader("Upload PDF/TXT", type=["pdf","txt"], accept_multiple_files=True)

if st.button("Build Knowledge Base") and files:
    store = VectorStore()
    embeddings = []
    metadata = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        text = load_pdf(path) if file.name.endswith(".pdf") else load_txt(path)

        for chunk in chunk_text(text):
            embeddings.append(embed(chunk))
            metadata.append({"source": file.name, "text": chunk})

    store.add(embeddings, metadata)
    store.save(DB_PATH)
    st.success("Knowledge base created")

question = st.chat_input("Ask a question")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question:
    st.session_state.messages.append({"role":"user","content":question})

    store = VectorStore()
    store.load(DB_PATH)

    docs = store.search(embed(question), k=5)
    context = "\n\n".join(d["text"] for d in docs)

    answer = chat(question, context)

    sources = "\n".join({d["source"] for d in docs})
    final = f"{answer}\n\n**Sources**\n{sources}"

    st.session_state.messages.append({"role":"assistant","content":final})
    st.rerun()
