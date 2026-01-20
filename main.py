"""
ECU Agent 主程序入口模块

该模块负责启动 ECU 技术文档问答代理。
"""

#测试便捷回复
from agent import query_ecu_agent
questions = [
        "What is the maximum operating temperature for the ECU-750?",
        "How much RAM does the ECU-850 have?",
        "What are the AI capabilities of the ECU-850b?",
        "What are the differences between ECU-850 and ECU-850b?",
        "Compare the CAN bus capabilities of ECU-750 and ECU-850.",
        "What is the power consumption of the ECU-850b under load?",
        "Which ECU models support Over-the-Air (OTA) updates?",
        "How does the storage capacity compare across all ECU models?",
        "Which ECU can operate in the harshest temperature conditions?",
        "How do you enable the NPU on the ECU-850b?"
    ]
    
print("🚀 Starting ECU Agent Test Suite...\n")

for i, q in enumerate(questions, 1):
        print(f"--- Question {i} ---")
        print(f"❓ {q}")
        try:
            answer = query_ecu_agent(q)
            print(f"✅ {answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
