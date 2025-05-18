import os
import dotenv
import streamlit as st

from rag import load_pdf, embed_documents, retrieval, generate_response

dotenv.load_dotenv()
os.environ['LANGSMITH_TRACING'] = os.getenv("LANGSMITH_TRACING")
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['USER_AGENT'] = os.getenv("USER_AGENT")

# ---------------------------------------

st.set_page_config(page_title="PDF Loader", page_icon="📄")
st.write("## Chat RAG 📄📎")
st.write(
    """
    This is a chatbot based on Llama 3.2 that uses a PDF file as a knowledge base.
    It can answer questions based on the content of the PDF file.
    """
)
st.markdown("---")


st.sidebar.header("📄 Upload a PDF file")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])


st.sidebar.header("🧠 Model Parameters")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
top_k = st.sidebar.slider("Top-k", 1, 100, 40, 1)


if uploaded_file is not None:
    
    if "vector_store" not in st.session_state:
  
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Procesando PDF..."):
            splits = load_pdf("temp_uploaded.pdf")
            st.session_state.vector_store = embed_documents(splits)

        os.remove("temp_uploaded.pdf")

else:
    st.warning("🔺 Please, load a PDF file from the sidebar")
    st.stop()


# sesssion state para historial de mensajes -------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#-------------------------------------

if user_input := st.chat_input("Enter your question here..."):

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.status("🔍 Looking up relevant documents...", expanded=True) as status:
        retrieved_docs, n = retrieval(st.session_state.vector_store, user_input)
        status.write("✅ Documents retrieved.")
        status.write(f"{n} fragments found.")

    with st.chat_message("assistant"):
        stream, metadata = generate_response(retrieved_docs, user_input,top_p=top_p, temperature=temperature, top_k=top_k)
        response = st.write_stream(stream)

        with st.expander("📊 Response parameters"):
            st.markdown(f"""
            - Temperature: `{metadata['temperature']}`
            - Top-p: `{metadata['top_p']}`
            - Top-k: `{metadata['top_k']}`
            """)

    st.session_state.messages.append({"role": "assistant", "content": response})
