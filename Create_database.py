import os
# os is a Python module to interact with operating system (manage paths and load environment variables)
from langchain_community.document_loaders import DirectoryLoader
# DirectoryLoader is a func from langchain_community that loads files from a directory into a list of Document objects.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# Chroma is a vector database, optimized for storing and retrieving large amounts of data using embeddings
# Help store and retrieve quickly the most relevant documents or chunks when a query is made
import shutil # module support file copying and removal
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
# Convert text into tokens

# Step 1: LOAD DOCUMENTS AND SPLIT IN SMALL CHUNKS
Data_path = 'data'
Chroma_path = 'chroma'

loader = DirectoryLoader(Data_path, glob="*.md")
documents = loader.load()
#for doc in documents:
    #print(doc)

# Step 2. SPLIT DOCUMENTS INTO SMALL CHUNKS

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, #Maximum of 300 characters in a chunk
    chunk_overlap = 150,
    add_start_index = True,
)

chunks = text_splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(chunks)} chunks.")



chunk_10 = chunks[10]
#print(document)
#print(document.page_content)
#print(document.metadata)
print(chunk_10)

# Step 3. CREATE EMBEDDINGS DATABASE
# => This involves transforming the text into numerical vectors that can be used for similarity searches in RAG system.
# Convert text into vectors (embeddings), then store in Chroma, then vector save in disk that can be reused

# Clean database first => Prevent duplicate data
if os.path.exists(Chroma_path):
    shutil.rmtree(Chroma_path)

# Create an instance of the embedding class
embeddings_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a list of the text contents from your chunks
#text_contents = [chunk.page_content for chunk in chunks]
#Generate embeddings for your cleaned texts
#document_embeddings = embeddings.embed_documents(text_contents[10])

# Create the Chroma database
db = Chroma.from_documents(
    chunks,
    embeddings_func,
    persist_directory= Chroma_path
    # parameters: documents (to convert into embeddings),
    # embedding_function (function used to convert)
    # persist_directory (directory where embeddings will be saved). If directory DNE, will create automatically
)
print(f"Saved {len(chunks)} chunks to {Chroma_path}.")