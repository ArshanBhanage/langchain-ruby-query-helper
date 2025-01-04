import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == '__main__':
    print("Hello")