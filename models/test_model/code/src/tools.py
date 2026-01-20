from pathlib import Path
import os
from langchain_core.tools import tool
from rag import get_vectorstore

current_dir = Path(__file__).parent  # tools.py所在的目录
base_dir = current_dir / "data"      # 与tools.py同级的data文件夹

def read_file(name: str) -> str:
    """Return file content. If not exist, return error message.
    """
    print(f"(read_file {name})")
    try:
        with open(base_dir / name, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"An error occurred: {e}"

def list_files() -> list[str]:
    print("list_files() of path: "+ str(base_dir))
    file_list = []
    for item in base_dir.rglob("*"):
        if item.is_file():
            file_list.append(str(item.relative_to(base_dir)))
    return file_list

def rename_file(name: str, new_name: str) -> str:
    print(f"(rename_file {name} -> {new_name})")
    try:
        new_path = base_dir / new_name
        if not str(new_path).startswith(str(base_dir)):
            return "Error: new_name is outside base_dir."

        os.makedirs(new_path.parent, exist_ok=True)
        os.rename(base_dir / name, new_path)
        return f"File '{name}' successfully renamed to '{new_name}'."
    except Exception as e:
        return f"An error occurred: {e}"
    
@tool
def query_ecu_700_series(query: str) -> str:
    """Use this to answer questions about ECU-700 series models (e.g., ECU-750)."""
    vs = get_vectorstore("700")
    results = vs.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in results])

@tool
def query_ecu_800_series(query: str) -> str:
    """Use this to answer questions about ECU-800 series models (e.g., ECU-800b, ECU-800P)."""
    vs = get_vectorstore("800")
    results = vs.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in results])