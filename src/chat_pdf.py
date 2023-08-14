"""Chat PDF file"""

import os
import gradio as gr
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv


load_dotenv()


model_path = os.getenv("MODEL_PATH")
file_path = os.getenv("FILE_PATH")
vectordb_path = os.getenv("VECTORDB_PATH")


class ChatPDF:
    def __init__(self, files: list, vectordb_path: str) -> None:
        self.files = files
        self.pages = []
        self.documents = []
        self.vectordb_path = vectordb_path

    def load(self) -> tuple[int, int]:
        pages = []

        for file in self.files:
            print(f"FILE {file}")
            loader = PyPDFLoader(file)
            pages = loader.load()
            self.pages.extend(pages)
            print(f"Loading file {file}")

        return len(self.files), len(self.pages)

    def split(self, chunk_size: int = 1500, chunk_overlap: int = 150) -> int:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.documents = text_splitter.split_documents(self.pages)

        return len(self.documents)

    def get_embeddings(self) -> None:
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def store(self):
        vectordb = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
        )

        vectordb.persist()

        self.vectordb = vectordb

    def create_llm(self, temperature: float = 0.7) -> None:
        self.llm = LlamaCpp(
            # model_path="/home/alanmxll/www/personal/chat-pdf/models/llama-2-7b-chat.ggmlv3.q4_0.bin",
            model_path=model_path,
            verbose=True,
            n_ctx=2048,
            temperature=temperature,
        )

    def create_memory(self) -> None:
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

    def create_retriever(self) -> None:
        self.retriever = self.vectordb.as_retriever()

    def create_chat_session(self):
        PROMPT_TEMPLATE = """
        Use the following pieces of context to answer the question at the end.
        If you don't the answer, just say that you don't know,
        don't try to male up the answer.
        Use three sentences maximum.
        Keep answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.
        {context}
        Question: {question}
        Helpful Answer:
        """

        QA_CHAIN_PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
        )


files = [file_path]

chat = ChatPDF(files, vectordb_path)

chat.load()
chat.split()
chat.get_embeddings()
chat.store()
chat.create_llm()
chat.create_memory()
chat.create_retriever()

chat.create_chat_session()

chat_history = []

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    chat_history = []

    def user(user_message, chat_history):
        result = chat.qa(
            {"question": user_message, "chat_history": chat_history})

        chat_history.append((user_message, result["answer"]))

        return gr.update(value=""), chat_history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True)
