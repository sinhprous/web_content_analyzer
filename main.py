import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title('Web Analyzer')
def analyze(url, fields):
    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    all_splits = text_splitter.split_documents(documents)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = OpenAIEmbeddings() # HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    db = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    retriever = db.as_retriever()


    llm = OpenAI(temperature=0.3, model="gpt-3.5-turbo-instruct", max_tokens=1024)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    return {k: qa.run(v) for k, v in fields.items()}


def main():
    url = st.text_input("URL:")
    open_ai_key = st.text_input("Your OpenAI key:")

    fields = {
        "Fees/Cost (Boolean)" : "Response with TRUE or FALSE to indicate if the this website content contains any information about the fees or cost or financial obligations which users might incur by using the service or product of that website. No additional explanation is required",
        "Explanation (Text)": "Please to explain the fees and costs if it’s stated in the content.",
        "Section Location (Text)": "Please return the locations where the found fees or cost information are located in the given URL content.",
        "Original Raw Text (Text)": "Please response with chunks of original text containing information about fees/cost.",
    }
    
    if url and open_ai_key:
        os.environ['OPENAI_API_KEY'] = open_ai_key
        for k, v in analyze(url, fields).items():
            st.header(k)
            st.write(v)

if __name__ == '__main__':
    main()