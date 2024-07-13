import streamlit as st
import os
from utils import *
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


FILE_LIST = "archivos.txt"
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_ENV = st.secrets['PINECONE_ENV']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
INDEX_NAME = st.secrets['PINECONE_INDEX_NAME']

st.set_page_config('Chokobot Admin')
st.header("Pregunta a VectorDb")

with st.sidebar:
    archivos = load_name_files(FILE_LIST)
    files_uploaded = st.file_uploader(
        "Carga tu archivo",
        type="pdf",
        accept_multiple_files=True
        )
    
    if st.button('Procesar'):
        for pdf in files_uploaded:
            if pdf is not None and pdf.name not in archivos:
                archivos.append(pdf.name)
                text_to_pinecone(pdf)

        archivos = save_name_files(FILE_LIST, archivos)

    if len(archivos)>0:
        st.write('Archivos Cargados:')
        lista_documentos = st.empty()
        with lista_documentos.container():
            for arch in archivos:
                st.write(arch)
            if st.button('Borrar Documentos'):
                archivos = []
                clean_files(FILE_LIST)
                lista_documentos.empty()



user_question = st.text_input("Pregunta: ")
if user_question:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    embeddings = OpenAIEmbeddings()
    vstore = Pinecone.from_existing_index(INDEX_NAME, embeddings)

    docs = vstore.similarity_search(user_question, 10)
    llm = ChatOpenAI(model='gpt-4', temperature=1)
    chain = load_qa_chain(llm, chain_type="stuff")
    respuesta = chain.run(input_documents=docs, question=user_question)

    st.write(respuesta)
