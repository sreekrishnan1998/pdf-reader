import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()

        return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index_path = os.path.join("faiss_index", "index.faiss")
    vector_store = FAISS.load_local(faiss_index_path, embeddings, create_if_missing=True)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""
    Answer the questions as detailed as possible from provided context, make sure to provide all details. If the answer is not in the provided context
    just say "Answer is not available in context". Dont provide wrong answer\n
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
"""    
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, create_if_missing=True)
    docs=new_db.similarity_search(user_question)

    chain=get_conversational_chain()


    response=chain (
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ",response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question=st.text("Ask Question from PDF files")    

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your PDF")   
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text=get_pdf_text(pdf_docs)
                get_vector_store(text_chunks)
                st.sucsess("Done")

if __name__ =="__main__":
    main()