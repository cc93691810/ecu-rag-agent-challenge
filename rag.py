import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils import load_docs_from_markdown

# 全局缓存，避免重复加载
_vectorstores = {}

def get_vectorstore(series: str):
    if series in _vectorstores:
        return _vectorstores[series]
    # 定义本地持久化路径（与脚本同目录下的 chroma_db/ 子文件夹）
    persist_dir = os.path.join("chroma_db", f"ecu_{series}")
    os.makedirs(persist_dir, exist_ok=True)

    # 加载文档
    file_map = {
        "700": "data/ECU-700_Series_Manual.md",
        "800B": "data/ECU-800_Series_Base.md",
        "800P": "data/ECU-800_Series_Plus.md"
    }

    docs = load_docs_from_markdown(file_map[series])
    # 创建嵌入模型
    local_model_path = "./models/bge-small-en-v1.5"
    print(local_model_path)
    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_path,  # 指向本地目录
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    #embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    # 🔍 调试：打印第一条文档的 embedding 向量（前10个值）
    if docs:
        sample_text = docs[0].page_content[:100]  # 取前100字符作为样本
        # print(f"\n[DEBUG] Embedding sample text (series={series}):")
        # print(f"Text: {repr(sample_text)}")

        # 手动生成 embedding
        embedding_vector = embeddings.embed_query(sample_text)
        # print(f"Embedding shape: {len(embedding_vector)}")
        # print(f"First 10 values: {embedding_vector[:10]}\n")

    # 检查是否已有持久化数据
    if os.listdir(persist_dir):  # 非空目录 → 已存在
        print(f"📂 加载已存在的 ChromaDB: {persist_dir}")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"ecu_{series}_collection"
        )
    else:
        # 首次创建：从文档构建并持久化
        print(f"🆕 首次构建 ChromaDB 并保存到: {persist_dir}")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=f"ecu_{series}_collection"
        )
        # Chroma 会自动持久化，无需显式调用 persist()
    _vectorstores[series] = vectorstore
    return vectorstore
