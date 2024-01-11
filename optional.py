import os
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

import openai
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()
import time
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
SAMPLE_TEXT = os.environ["SAMPLE_TEXT"]

print(f"PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}  and type: {type(PINECONE_INDEX_NAME)}")
print(f"PINECONE_ENVIRONMENT: {PINECONE_INDEX_NAME}  and type: {type(PINECONE_INDEX_NAME)}")
print(f'PINECONE_API_KEY: {PINECONE_API_KEY} and type: {type(PINECONE_API_KEY)}')
print(f'OPENAI_API_KEY: {OPENAI_API_KEY} and type: {type(OPENAI_API_KEY)}') 
print(f'SAMPLE_TEXT: {SAMPLE_TEXT} and type: {type(SAMPLE_TEXT)}, and check if {SAMPLE_TEXT == "sample-1276182"}')


question = "Tell me something about machine learning clustering methods?"
embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# print("Embeddings: ", embeddings)

## Vector search DB in  Pinecone
pinecone.init(
  api_key=PINECONE_API_KEY,
  environment=PINECONE_ENVIRONMENT
)
index_name=PINECONE_INDEX_NAME
time.sleep(2)

llm = OpenAI(model_name="davinci-002", temperature=0.5)
chain = load_qa_chain(llm, chain_type="stuff")

# query = "Tell me something about AI?"
index = Pinecone.from_documents("", embeddings, index_name=index_name)
doc_search = index.similarity_search(question, k=2)
answer=chain.run(input_documents=doc_search, question=question)
answer = str(answer).replace('\n', ' ')

print(answer)