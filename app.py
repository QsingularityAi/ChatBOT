from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingfaceembedding
from dotenv import load_dotenv
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.prompt import *
import os
from src.helper import load_data, text_split, download_huggingfaceembedding
from langchain.vectorstores import Chroma
app = Flask(__name__)

load_dotenv()

extracted_pdf = load_data("Data/")
text_chunks = text_split(extracted_pdf)
embedding = download_huggingfaceembedding()

def embeddings_on_local_vectordb(text_chunks):
    vectordb = Chroma.from_documents(text_chunks, embedding=embedding,
                                     persist_directory="./chroma_db")
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever


# query = "What did the president say about Ketanji Brown Jackson"
# db2 = Chroma.from_documents(text_chunks, embedding, persist_directory="./chroma_db")
# docs = db2.similarity_search(query)

# # load from disk
# db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
# docs = db3.similarity_search(query)
# #print(docs[0].page_content)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
chain_type_kwargs = {"prompt":PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={"max_new_tokens":512,
                            'temperature':0.8,
                            'context_length':2048})

retriever = embeddings_on_local_vectordb(text_chunks)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents = True,
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods = ["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query":input})
    print("Response: ", result["result"])
    return str(result["result"])


if __name__ == "__main__":
    app.run(debug=True)