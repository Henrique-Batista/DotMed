from langchain.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
import streamlit as st
import pandas as pd
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
context_path = "contexto"

st.set_page_config(page_title="DotMed - Assistente Virtual", page_icon="🏥")

st.title("🏥 Assistente DotMed")
st.markdown("Bem-vindo ao canal de suporte da DotMed. Como posso ajudar com seus exames hoje?")

memory = {}
session = "dotmed_chat"

def historico_por_sessao(sessao: str):
    if session not in memory:
        memory[session] = InMemoryChatMessageHistory()
        return memory[session]

@st.cache_resource
def load_vectorstore():
    local_store = "faiss_docs_embeddings"
    docs = []
    
    # PDF Loading
    pdf_files = ["manual-de-preparo-para-exames.pdf", "Manual-de-preparo-para-realizacao-dos-exames.pdf"]
    for pdf in pdf_files:
        path = f"{context_path}/{pdf}"
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        
    # CSV Loading
    csv_path = f"{context_path}/planilha-de-procedimentos.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        texts = df.apply(lambda row: ', '.join(row.astype(str)), axis=1).tolist()
        docs.extend([Document(page_content=text, metadata={"source": "csv"}) for text in texts])
    
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

    if os.path.exists(local_store):
        return FAISS.load_local(local_store, embeddings, allow_dangerous_deserialization=True)

    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

    vector_store.save_local(local_store)

    return vector_store

def get_rag_chain():
    vectorstore = load_vectorstore()
    if not vectorstore:
        st.error("Erro: Nenhum documento de contexto encontrado na pasta 'contexto'.")
        return None

    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        temperature=0.3,
        google_api_key=api_key
    )

    system_prompt = """
            Você é o assistente virtual da clínica DotMed. 
            Seu objetivo é auxiliar pacientes com dúvidas sobre preparo de exames. 
            
            DIRETRIZES DE RESPOSTA:
            1. Use APENAS o contexto fornecido para informações técnicas de exames.
            2. Se a informação não estiver no contexto, diga gentilmente que não possui essa informação específica e sugira contato direto com a central.
            3. Para dúvidas sobre 'andamento de resultados', se não houver dados específicos no contexto, informe que os resultados podem ser consultados no portal do paciente ou retirados na unidade, respeitando o prazo dado no dia da coleta.
            4. Mantenha um tom profissional, acolhedor e empático.
            5. Não responda sobre assuntos que não sejam médicos ou da clínica DotMed.

            Verifique o historico de mensagens em "placeholder" para mensagens anteriores.
            """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{history}"),
        ("user", "{input}\n\nCONTEXTO:\n{context}"),
    ])

    chain = prompt | llm

    chain_with_memory = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=historico_por_sessao,
        input_messages_key="input",
        history_messages_key="history"
    )

    return chain_with_memory, retriever


# Inicializar histórico de chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Sou o assistente da DotMed. Posso tirar suas dúvidas sobre preparo de exames. Como posso ajudar?"}
    ]

# Exibir mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if user_input := st.chat_input("Ex: Como preciso me preparar para fazer o exame de urina ?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consultando manuais da DotMed..."):
            try:
                chain_results = get_rag_chain()
                if chain_results:
                    chain, retriever = chain_results
                    context_docs = retriever.invoke(user_input)
                    context_text = "\n\n".join([doc.page_content for doc in context_docs])
                    
                    answer = chain.invoke(
                        {
                            "input": user_input, 
                            "context": context_text
                        }, 
                        config = {
                            "session_id":session
                        }
                    )

                    content = answer.content

                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})
            except Exception as e:
                st.error(f"Erro: {e}")
                error_msg = "Desculpe, ocorreu um erro ao processar sua solicitação."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
