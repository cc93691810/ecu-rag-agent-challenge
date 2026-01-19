"""
ECU Agent 主程序入口模块

该模块负责启动 ECU 技术文档问答代理。
"""
from agent import build_ecu_agent
app=build_ecu_agent()
result = app.invoke({
    "user_question": "What is the RAM size of ECU-850?",
    "series_to_query": "unknown",
    "retrieved_docs": [],
    "final_answer": ""
})

print(result["final_answer"])
