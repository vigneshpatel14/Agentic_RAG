import os
import uuid
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
load_dotenv()

from langchain.schema import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages


from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient


os.environ["LANGSMITH_PROJECT"] = "HR_RAG_PROJECT"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

def process_all_pdf(directory_path):
    all_docs = []
    pdf_files = list(Path(directory_path).rglob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in:", directory_path)
        return []

    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({"source_file": file_path.name, "file_type": "pdf"})
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} documents from {directory_path}")
    return all_docs


def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.embedding_model.encode(texts)


class VectorStore:
    def __init__(self, persistent_directory="./chroma_db"):
        self.client = PersistentClient(path=persistent_directory)
        self.collection = self.client.get_or_create_collection(name="PDF_DOCUMENTS")

    def add_documents(self, documents, embeddings):
        ids = [str(uuid.uuid4()) for _ in documents]
        metadatas = [doc.metadata for doc in documents]
        documents_text = [doc.page_content for doc in documents]
        embeddings_list = embeddings.tolist()

        self.collection.add(
            ids=ids,
            documents=documents_text,
            embeddings=embeddings_list,
            metadatas=metadatas
        )

    def search(self, query_embedding, top_k=5):
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)


class RAGRetriever:
    def __init__(self, vectorstore, embedding_manager):
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=5, min_score=0.0):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vectorstore.search(query_embedding, top_k=top_k)

        docs, metas, dists = results["documents"][0], results["metadatas"][0], results["distances"][0]
        retrieved = []

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            similarity_score = 1 - dist
            if similarity_score >= min_score:
                retrieved.append({
                    "id": str(uuid.uuid4()),
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": similarity_score,
                    "rank": i + 1
                })
        return retrieved


class GroqLLM:
    def __init__(self, api_key):
        self.llm = ChatGroq(temperature=0, model="gemma2-9b-it", api_key=api_key)

    def generate_response(self, question, context):
        system_prompt = f"""
        You are a helpful AI. Use the provided context to answer.
        If context is insufficient, say you don't know.

        Context:
        {context}
        """
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        return self.llm.invoke(messages).content


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm_graph = ChatGroq(model="gemma2-9b-it", temperature=0.6, api_key=os.getenv("GROQ_API_KEY"))
tool = TavilySearch(max_results=3)
tools = [tool]

llm_with_tools = llm_graph.bind_tools(tools)


def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


memory = MemorySaver()
builder = StateGraph(State)

builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
    {"tools": "tools", END: END}
)
builder.add_edge("tools", "tool_calling_llm")

tool_agent = builder.compile()



docs = process_all_pdf("./data")
if docs:
    chunks = split_docs(docs)
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.generate_embeddings([c.page_content for c in chunks])
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, embeddings)

    retriever = RAGRetriever(vectorstore, embedding_manager)
    llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))

    print("RAG pipeline initialized with PDFs in ./data")
else:
    retriever, llm = None, None
    print("No PDFs found in ./data - RAG pipeline not initialized")