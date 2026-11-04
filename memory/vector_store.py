from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMB = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
INDEX_DIR = "faiss_index"


def _load():
    if Path(INDEX_DIR).exists():
        return FAISS.load_local(INDEX_DIR, EMB, allow_dangerous_deserialization=True)
    return FAISS.from_texts(["init"], EMB)   # FAISS建库必须有至少一条文本，用占位符


def save_conclusion(session_id: str, csv_name: str, conclusion: str):
    """每次分析结束，把结论写入长期记忆"""
    vs = _load()
    vs.add_texts([conclusion], metadatas=[{"session": session_id, "file": csv_name}])
    vs.save_local(INDEX_DIR)


def recall(query: str, k: int = 3) -> list[str]:
    """按语义召回相关的历史结论"""
    if not Path(INDEX_DIR).exists():
        return []
    docs = _load().similarity_search(query, k=k)
    return [d.page_content for d in docs if d.page_content != "init"]  # 过滤占位符
