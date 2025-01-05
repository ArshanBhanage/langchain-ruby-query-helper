import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings


load_dotenv()

embeddings = VertexAIEmbeddings(model="text-embedding-005")

for root, dirs, files in os.walk("migrations"):
    for file in files:
        loader = TextLoader(f"migrations/{file}")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # print(documents)

        vector_store = PineconeVectorStore.from_documents(documents=docs, index_name=os.getenv("INDEX_NAME"),
                                                          embedding=embeddings)
