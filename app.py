#Importing libraries
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import tempfile
import os
import os

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disables problematic inspection
load_dotenv()
st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:") #page title

@st.cache_resource 
def initiate_project():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10, length_function=len)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
                )
    
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                                    verbose = True,
                                                    temperature = 0.5,
                                                    google_api_key=os.getenv("GEMINI_API_KEY"))
    
    return text_splitter, embeddings, gemini_llm

def store_documents(file_paths,text_splitter,embeddings):
    try:
        if not os.path.exists('processed_files.txt'):
            open('processed_files.txt','x')
            
        with open('processed_files.txt','r') as f:
            processed_files_list=f.readlines()   #processed_files_list=list of "filenames".
            processed_files_list=[i.rstrip() for i in processed_files_list]
            f.close()

        all_pages=[]
        for path in file_paths:
            if path in processed_files_list:
                continue
            loader = PyMuPDFLoader(path)
            all_pages.extend(loader.load())
            chunks = text_splitter.split_documents(all_pages)

            #turns text into numerical vectors using transformer models.
            #create a vector store from the list of documents uploaded
            vectorstore  = Chroma.from_documents(documents=chunks,
                                                        embedding=embeddings,
                                                        persist_directory="./chroma_db")
            with open('processed_files.txt','a') as f:
                f.write(path+"\n")
        return True
    except Exception as e:
        print(e)
        return False

def main(): 
    st.header("Chat with multiple PDFs:books:")
    text_splitter,embeddings , gemini_llm = initiate_project()

    with st.sidebar:
        st.subheader("Upload Documents")
        files=st.file_uploader("Upload documents here",accept_multiple_files=True)#streamlit receives uploaded files as file object
        button=st.button("Process",key='process')
        if button and files:
            with st.spinner("Processing documents..."):
                file_paths=[]
                
                for file in files:
                    os.makedirs("temp",exist_ok=True)
                    with open(os.path.join("temp",file.name),"wb") as f:    #(temp/file.name.pdf)   (f is an empty pdf)
                        f.write(file.getvalue()) #bytes of uploaded files
                    file_paths.append(os.path.join("temp",file.name))
                status=store_documents(file_paths,text_splitter,embeddings)
                if status:
                    st.write("Document added to vectorstore")

                else:
                    st.write("Document already exist")
                                
    query=st.text_input("Ask a question related to uploaded pdfs")
    button=st.button("Submit",key='query')        

    if button and query:

        vectorstore=Chroma(persist_directory="./chroma_db",embedding_function=embeddings)
        retriever = vectorstore.similarity_search(query, k=5)
        retrieved_string=""

        for doc in retriever:
            text = doc.page_content
            retrieved_string+=text+"\n\n"

        combined_string=f"""You are an expert assistant. Use the context below to answer the user's question as clearly,directly and accurately as possible.

            Context:
            {retrieved_string}

            Question:
            {query}

            Answer:"""
        #Predicting the answer of the query using geminillm
        response=gemini_llm.predict(combined_string)
        st.write(response)

if __name__ == '__main__':
    main()
