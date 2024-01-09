

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


## Let's read the document
def read_doc(directory):
  file_loader = PyPDFDirectoryLoader(directory)
  documents = file_loader.load()
  return documents

## Divide the docs into chuncks 

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  doc = text_splitter.split_documents(docs)
  return docs

## Cosine Similarity Retreive Results from vectorDB
def retrieve_query(query, index, k=2):
  matching_results=index.similarity_search(query, k=k)
  return matching_results

## Search answers from VectorDB
def retrieve_answer(query, chain, index):
  doc_search = retrieve_query(query=query, index=index)
  print(doc_search)
  response=chain.run(input_documents=doc_search, question=query)
  return response

