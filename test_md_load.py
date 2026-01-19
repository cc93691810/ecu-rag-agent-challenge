import os
from pathlib import Path
from utils import load_docs_from_markdown
import tools

tools.list_files()
# 获取脚本所在目录
script_dir = Path(__file__).parent

# 构建完整路径
md_file = script_dir / "data" / "ECU-800_Series_Base.md"


docs = load_docs_from_markdown("data/ECU-800_Series_Base.md")
for d in docs:
    if "850" in d.page_content:
        print(d.page_content)

from rag import get_vectorstore
_ = get_vectorstore("700")  # Base
_ = get_vectorstore("800B")  # Base
_ = get_vectorstore("800P")  # Plus