# # ''' In this script we are creating a RAG system using Pinecone database instead of FAISS
# # so for this we need to import pinecone,PineconeEmbeddings
# # PineconeEmbeddings= For converting text into numeric representation
# # '''
# #
# # import os
# # import configparser
# # from langchain.document_loaders import TextLoader
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# # from langchain.vectorstores import pinecone
# # from langchain.chains import RetrievalQA
# # from langchain_pinecone import PineconeEmbeddings,PineconeVectorStore
# # from openai import api_key
# # from pinecone import Pinecone, ServerlessSpec
# # from langchain.vectorstores import Pinecone
# # from datetime import  time
# # from langchain import hub  # For chatbot
# #
# #
# # # Loading API key from config file
# # config = configparser.ConfigParser()
# # config.read("config - Copy.ini")
# #
# # OPENAI_API_KEY = config.get("API_KEYS", "OpenAI", fallback=None)
# # Pinecone_API_KEY=config.get("API_KEYS", "Pinecone_API_KEY", fallback=None)
# # if not OPENAI_API_KEY:
# #     raise ValueError("OpenAI API Key not found in config file.")
# # if not Pinecone_API_KEY:
# #     raise ValueError("Pinecone API KEY  not found in config file.")
# #
# # #Setting API key as an enviornment variable
# # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# # os.environ["Pinecone_API_KEY"] = Pinecone_API_KEY
# # print(f"API Key Loaded: {OPENAI_API_KEY[:5]}********")
# # print(f"API Key Loaded: {Pinecone_API_KEY[:5]}********")
# #
# # # Here we are loading the data
# # loader = TextLoader("source.txt")
# # documents = loader.load()
# # print("Data is loaded successfully")
# #
# #
# # # Splitting the loaded data into smaller parts
# # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# # docs = text_splitter.split_documents(documents)
# # print("Splitted the document into smaller parts")
# #
# #
# # # Converting the text data into numeric form and storing it to the vector database
# # model_name = 'multilingual-e5-large'
# # embeddings = PineconeEmbeddings(
# #     model=model_name,
# #     pinecone_api_key=Pinecone_API_KEY
# # ) # creating the object of embeddings
# #
# # print("Created the object of embeddings"
# #       "")
# #
# # index_name="rag-index"
# # pine = Pinecone()
# # cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
# # region = os.environ.get('PINECONE_REGION') or 'us-east-1'
# # spec = ServerlessSpec(cloud=cloud, region=region)
# #
# # index_name = "rag-index"
# #
# # if index_name not in pine.list_indexes().names():
# #     pine.create_index(
# #         name=index_name,
# #         dimension=embeddings.dimension,
# #         metric="cosine",
# #         spec=spec
# #     )
# # print("Index is created ! create_index is working properly ")
# # print(pine.Index(index_name).describe_index_stats())
# # print("\n")
# #
# #
# # namespace = "Fidel"
# # print("Till namespace is working proprly")
# # vector_db = PineconeVectorStore.from_documents(
# #     api_key=os.environ.get("PINECONE_API_KEY"),
# #     documents=docs,
# #     index_name=index_name,
# #     embedding=embeddings
# #     # namespace=namespace
# #
# # )
# # print("Vector db is working .......")
# # time.sleep(5)
# #
# # # See how many vectors have been upserted
# # print("Index after upsert:")
# # print(pine.Index(index_name).describe_index_stats())
# # print("\n")
# # time.sleep(2) # Passing the data to our embedding s
# # print("Data is stored in the database along with embeddings")
# #
# # # here we are searching for relevant parts as per user's query
# # retriever = vector_db.as_retriever()
# # # print("Retriever is working properly")
# #
# # # Defining the llm model (gpt 3.5 turbo)
# # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
# # ''''''
# # rag_chain = RetrievalQA.from_chain_type(
# #     llm=llm,
# #     chain_type="stuff",
# #     retriever=retriever
# # )
# #
# # # query = "What is the main topic in source.txt?"
# # while True:
# #     query = input('''Do you have any question?
# #                 Please ask me:-''')
# #     response = rag_chain.invoke({"query": query})
# #
# #     print("\n **Answer:**")
# #     print(response["result"])
# #
#
#
# ''' In this script we are creating a RAG system using Pinecone database instead of FAISS
# so for this we need to import pinecone,PineconeEmbeddings
# PineconeEmbeddings= For converting text into numeric representation
# '''
#
# import os
# import configparser
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.vectorstores import pinecone
# from langchain.chains import RetrievalQA
# from langchain_pinecone import PineconeEmbeddings,PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain.vectorstores import Pinecone
# from datetime import  time
# from langchain import hub  # For chatbot
# from langchain_pinecone import PineconeVectorStore  # Use this for vector store
# from langchain_openai import OpenAIEmbeddings  # Use OpenAI embeddings
#
# # Loading API key from config file
# config = configparser.ConfigParser()
# config.read("config - Copy.ini")
#
# OPENAI_API_KEY = config.get("API_KEYS", "OpenAI", fallback=None)
# Pinecone_API_KEY=config.get("API_KEYS", "Pinecone_API_KEY", fallback=None)
# if not OPENAI_API_KEY:
#     raise ValueError("OpenAI API Key not found in config file.")
# if not Pinecone_API_KEY:
#     raise ValueError("Pinecone API KEY  not found in config file.")
#
# #Setting API key as an enviornment variable
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["PINECONE_API_KEY"] = Pinecone_API_KEY
# print(f"API Key Loaded: {OPENAI_API_KEY[:5]}********")
# print(f"API Key Loaded: {Pinecone_API_KEY[:5]}********")
#
# # Here we are loading the data
# loader = TextLoader("source.txt")
# documents = loader.load()
# print("Data is loaded successfully")
#
#
# # Splitting the loaded data into smaller parts
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)
# print("Splitted the document into smaller parts")
#
#
# # Converting the text data into numeric form and storing it to the vector database
# model_name = 'multilingual-e5-large'
# embeddings = PineconeEmbeddings(
#     model=model_name,
#     pinecone_api_key=Pinecone_API_KEY
# ) # creating the object of embeddings
#
# print("Created the object of embeddings"
#       "")
#
# index_name="rag-index"
#
# #pine = Pinecone(index=index_name,embedding=embeddings,text_key=docs)
# pine = Pinecone(Pinecone_API_KEY)
# cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
# region = os.environ.get('PINECONE_REGION') or 'us-east-1'
# spec = ServerlessSpec(cloud=cloud, region=region)
#
# index_name = "rag-index"
#
# if index_name not in [index["name"] for index in pine.list_indexes()]:
#     pine.create_index(
#         name=index_name,
#         dimension=embeddings.dimension,
#         metric="cosine",
#         spec=spec
#     )
# print("Index is created ! create_index is working properly ")
# print(pine.Index(index_name).describe_index_stats())
# print("\n")
#
#
# namespace = "Fidel"
# print("Till namespace is working proprly")
#
# vector_db = PineconeVectorStore.from_documents(
#     documents=docs,
#     index_name=index_name,
#     embedding=embeddings,
#     namespace=namespace)
#
# print("Vector db is working .......")
# time.sleep(5)
#
# # See how many vectors have been upserted
# print("Index after upsert:")
# print(pine.Index(index_name).describe_index_stats())
# print("\n")
# time.sleep(2) # Passing the data to our embedding s
# print("Data is stored in the database along with embeddings")
#
# # here we are searching for relevant parts as per user's query
# retriever = vector_db.as_retriever()
# # print("Retriever is working properly")
#
# # Defining the llm model (gpt 3.5 turbo)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
# ''''''
# rag_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever
# )
#
# # query = "What is the main topic in source.txt?"
# while True:
#     query = input('''Do you have any question?
#                 Please ask me:-''')
#     response = rag_chain.invoke({"query": query})
#
#     print("\n **Answer:**")
#     print(response["result"])
#



import os
import configparser
import logging
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API keys from config file
config = configparser.ConfigParser()
config.read("config - Copy.ini")

OPENAI_API_KEY = config.get("API_KEYS", "OPENAI_API_KEY", fallback=None)
PINECONE_API_KEY = config.get("API_KEYS", "PINECONE_API_KEY", fallback=None)

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found in config file.")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API Key not found in config file.")

# Set API keys as environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
logging.info("API Keys loaded successfully.")

# Load the data
SOURCE_FILE = "source.txt"
if not os.path.exists(SOURCE_FILE):
    raise FileNotFoundError(f"Source file '{SOURCE_FILE}' not found.")

loader = TextLoader(SOURCE_FILE)
documents = loader.load()
logging.info("Data loaded successfully.")

# Split the loaded data into smaller parts
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
logging.info("Document split into smaller parts.")

# Use OpenAI's `text-embedding-3-small` model (1024 dimensions)
embedding_model = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embedding_model)
logging.info("Embedding model initialized successfully.")

# Initialize Pinecone
pine = Pinecone(api_key=PINECONE_API_KEY)
cloud = os.environ.get("PINECONE_CLOUD", "aws")
region = os.environ.get("PINECONE_REGION", "us-east-1")
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "rag-index-fsl"
namespace = "Fidel"

# Ensure the Pinecone index exists
existing_indexes = [index["name"] for index in pine.list_indexes()]
if index_name not in existing_indexes:
    logging.warning(f"Pinecone index '{index_name}' not found. Creating a new one...")
    pine.create_index(name=index_name, dimension=1536, metric="cosine", spec=spec)
    logging.info(f"Created new Pinecone index '{index_name}' with 1536 dimensions.")
else:
    logging.info("Connected to existing Pinecone index.")

# Create a vector database using Pinecone
vector_db = PineconeVectorStore.from_documents(
    documents=docs,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace,
    pinecone_api_key=PINECONE_API_KEY
)
logging.info("Vector database initialized successfully.")

# Setup the retriever
retriever = vector_db.as_retriever()

# Define the LLM model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Define the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Query loop
while True:
    try:
        query = input("Do you have any questions?\nPlease ask me: ")
        if query.lower() in ["exit", "quit", "q"]:
            logging.info("Exiting chatbot.")
            break
        response = rag_chain.invoke({"query": query})
        print("\n **Answer:**")
        print(response["result"])
    except Exception as e:
        logging.error(f"An error occurred: {e}")
