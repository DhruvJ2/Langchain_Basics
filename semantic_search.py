import os
from unittest import loader
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

print('======================================')
print("PDF book summarizer")
print('======================================')
pdf_path = input("Enter book file path: ")
user_input = input("Ask the question?\n")


load_dotenv()
apikey = os.getenv("OPENAI_API_KEY")

file_path = pdf_path
loader = PyPDFLoader(file_path)
docs = loader.load()

# print(f"{docs[0].page_content[:200]}\n\n")
# print(docs[0].metadata)

## Text Splitters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
# print(len(all_splits))

"""Vector search is a common way to store and search over unstructured data"""
#Embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=apikey)
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

# assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])

## Vector Store
vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search_with_score(user_input)

doc, score = results[0]

## Retrieval process
# retriever = vector_store.as_retriever(
    # search_type="similarity",
    # search_kwargs={"k": 1},
# )

# page_info = retriever.batch(user_input)

print('======================================')
print(f"Author: {doc.metadata["author"]}")
print('======================================')
print(f"Score: {score}")
print(f"Page number: {doc.metadata["page"]}\nPage Label: {doc.metadata["page_label"]}\nSource: {doc.metadata["source"]}")
print(f"data: {doc.page_content}")

"""
Retrievers can easily be incorporated into more complex applications, 
such as retrieval-augmented generation (RAG) applications 
that combine a given question with retrieved context into a prompt for a LLM
"""