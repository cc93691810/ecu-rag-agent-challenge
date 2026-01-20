import json
from pathlib import Path
import langchain_core
from utils import load_docs_from_markdown
import os
# 获取脚本所在目录
script_dir = Path(__file__).parent
def test_utils_parsing():
    test_files = [
            "data/ECU-700_Series_Manual.md",
            "data/ECU-800_Series_Plus.md",   # 确保你已创建此文件
            "data/ECU-800_Series_Base.md"  # 如有 Base 系列也可加入
        ]

    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping {file_path} (not found)")
            continue

        try:
            docs = load_docs_from_markdown(file_path)
            print(f"✅ Loaded {len(docs)} document chunks\n")

            for i, doc in enumerate(docs, 1):
                print(f"--- Chunk {i} ---")
                print("Metadata:", doc.metadata)
                print("Content preview:")
                # 打印前 300 字符，避免太长
                print(doc.page_content[:300])
                if len(doc.page_content) > 300:
                    print("... (truncated)")
                print()

        except Exception as e:
            print(f"test.py: ❌ Error parsing {file_path}: {e}")
#test_utils_parsing()

# #测试agent
# from agent import build_ecu_agent
# app=build_ecu_agent()

# result = app.invoke({
#     "user_question": "What is the RAM size of ECU-850?",
#     "series_to_query": "unknown",
#     "retrieved_docs": [],
#     "final_answer": ""
# })

# print(result["final_answer"])

# #测试便捷回复
# from agent import query_ecu_agent
# questions = [
#         "What is the maximum operating temperature for the ECU-750?",
#         "How much RAM does the ECU-850 have?",
#         "What are the AI capabilities of the ECU-850b?",
#         "What are the differences between ECU-850 and ECU-850b?",
#         "Compare the CAN bus capabilities of ECU-750 and ECU-850.",
#         "What is the power consumption of the ECU-850b under load?",
#         "Which ECU models support Over-the-Air (OTA) updates?",
#         "How does the storage capacity compare across all ECU models?",
#         "Which ECU can operate in the harshest temperature conditions?",
#         "How do you enable the NPU on the ECU-850b?"
#     ]
    
# print("🚀 Starting ECU Agent Test Suite...\n")

# for i, q in enumerate(questions, 1):
#         print(f"--- Question {i} ---")
#         print(f"❓ {q}")
#         try:
#             answer = query_ecu_agent(q)
#             print(f"✅ {answer}\n")
#         except Exception as e:
#             print(f"❌ Error: {e}\n")