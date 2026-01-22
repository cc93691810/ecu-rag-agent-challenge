import re
from langchain_core.documents import Document

def load_docs_from_markdown(file_path: str) -> list[Document]:
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    # 提取 Series 名称（从文件名推断）
    if "700" in file_path:
        series_name = "ECU-700"
    elif "800" and "base" in file_path.lower():
        series_name = "ECU-800B"
    elif "800" and "plus" in file_path.lower():
        series_name = "ECU-800P"
    else:
        series_name = "Unknown"
    
    docs = []
    sections = re.split(r'\n##\s+(.+)', content)    

    for i in range(1, len(sections), 2):
        title = sections[i].strip()
        body = sections[i + 1].strip() if i + 1 < len(sections) else ""
        if not title or "series" in title.lower():
            continue
        ## 加入表格逻辑细分
        if "|" in body and "**" in body:
            # 尝试提取表格部分
            table_lines = []
            non_table_lines = []
            in_table = False
            for line in body.split("\n"):
                stripped = line.strip()
                if stripped.startswith("|") and stripped.endswith("|"):
                    table_lines.append(stripped)
                    in_table = True
                else:
                    if in_table and (stripped == "" or all(c in "-| " for c in stripped)):
                        # 跳过表格分隔线（如 |---|---|)
                        continue                    
                    non_table_lines.append(line)
                    in_table = False  # 非表格内容重置状态
            # === 1. 处理表格行：每行一个 chunk ===
            if table_lines:
                # 过滤掉表头分隔行（通常含 ---）
                data_rows = [
                    row for row in table_lines
                    if not re.match(r'\|\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|', row)
                ]
                # 跳过表头（第一行）
                for row in data_rows[1:]:
                    cells = [cell.strip().replace("**", "").strip() for cell in row.split("|")[1:-1]]
                    if len(cells) >= 2:
                        param_name = cells[0]
                        param_value = cells[1]
                        if param_name and param_value:
                            text = f"Series:{series_name}\nModel:{title}\nParameter:{param_name}\nValue:{param_value}"
                            docs.append(Document(
                                page_content=text,
                                metadata={
                                    "source": file_path,
                                    "model": title,
                                    "parameter": param_name
                                }
                            ))
            # === 2. 保留非表格描述（如软件配置）作为单独 chunk ===
            non_table_body = "\n".join(non_table_lines).strip()
            if non_table_body:
                text = f"Series: {series_name}\nModel: {title}\n\n{non_table_body}"
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "model": title}
                ))
        else:
            # 无表格：保留原始整块 chunk（兼容 ECU-700）
            text = f"Series: {series_name}\nModel: {title}\n\n{body}"
            docs.append(Document(
                page_content=text,
                metadata={"source": file_path, "model": title}
            ))        
    return docs

## test
# from pathlib import Path
# import os
# current_file = Path(__file__).resolve()  # rag.py 的绝对路径
# project_root = current_file.parent.parent  # 项目根目录

# # 加载文档
# data_dir = project_root / "data"
# file_map = {
#     "700": data_dir / "ECU-700_Series_Manual.md",
#     "800B": data_dir / "ECU-800_Series_Base.md",
#     "800P": data_dir / "ECU-800_Series_Plus.md"
# }

# test_series = "800P"  # 可以改为 "700", "800B", "800P"
# if test_series in file_map:
#     file_path = str(file_map[test_series])
#     if os.path.exists(file_path):
#         print(f"\n🔍 开始测试 {test_series} 系列:")
#         print(f"文件路径: {file_path}")
#         docs = load_docs_from_markdown(file_path)
        
#         # 可选：查看特定文档的完整内容
#         if docs:
#             print(f"\n{'='*80}")
#             print(f"🔍 查看文档的完整内容:")
#             print(f"{'='*80}")
#             for i in range(len(docs)):
#                 print(f"\n📄 文档 {i+1}/{len(docs)}:")
#                 print(f"元数据: {docs[i].metadata}")
#                 print(f"内容:")
#                 print(f"{'-'*60}")
#                 print(docs[i].page_content)
#                 print(f"{'-'*60}")
#                 print(f"长度: {len(docs[i].page_content)} 字符\n")
#     else:
#         print(f"❌ 文件不存在: {file_path}")
# else:
#     print(f"❌ 无效的系列标识: {test_series}")
#     print(f"可用的系列: {list(file_map.keys())}")
