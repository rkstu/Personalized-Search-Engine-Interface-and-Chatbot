from flask import Flask, request, render_template
import numpy as np
import pandas as pd

import openai
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import os
# from dotenv import load_dotenv
# load_dotenv()
import time
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

from helper_functions import (read_doc,
                              chunk_data,
                              retrieve_answer,
                              retrieve_query)

application = Flask(__name__)

app = application

## Route for a home page

# @app.route('/')
# def index():
#   return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    # print("Check out : http://127.0.0.1:5000/predictdata", )
    
    if request.method=='GET':
        return render_template('index.html')
    else:
        question=str(request.form.get('question'))

        print("WOrking 1")

        embeddings=OpenAIEmbeddings(api_key=${{ secrets.OPENAI_API_KEY }})
        # print("Embeddings: ", embeddings)

        ## Vector search DB in  Pinecone
        pinecone.init(
          api_key=${{ secrets.PINECONE_API_KEY }},
          environment=${{ secrets.PINECONE_ENVIRONMENT }}
        )
        index_name=${{ secrets.PINECONE_INDEX_NAME }}
        # time.sleep(20)

        llm = OpenAI(model_name="davinci-002", temperature=0.5)
        chain = load_qa_chain(llm, chain_type="stuff")

        # query = "Tell me something about AI?"
        index = Pinecone.from_documents("", embeddings, index_name=index_name)
        doc_search = index.similarity_search(question, k=2)
        answer=chain.run(input_documents=doc_search, question=question)
        # print(str(response))

        # out_query = "How much the agriculture target will be increased by how many crores"
        # out_query = "Why agriculture is important?"
        # time.sleep(20)
        answer = str(answer).replace('\n', ' ')


        return render_template('home.html',results=answer)
    



if __name__=="__main__":
    app.run(host="0.0.0.0")        


