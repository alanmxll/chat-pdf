"""Vectors file"""

import os

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, CSVLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.redis import Redis


load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
MODEL_PATH = os.getenv("MODEL_PATH")
DOCS_PATH = os.getenv("DOCS_PATH")

CHUNK_SIZE = 400

loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.csv",
    loader_cls=CSVLoader,
)


def load() -> None:
    """
    Load files to Redis Database
    """
    docs = loader.load()

    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0,
    )

    splitted_documents = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    Redis.from_documents(
        documents=splitted_documents,
        embedding=embeddings,
        index_name="mms3m-embeddings",
        redis_url=REDIS_URL,
    )


if __name__ == "__main__":
    load()
