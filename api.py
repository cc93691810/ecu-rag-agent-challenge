# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import logging

# 添加项目根目录到 Python 路径（确保能导入 agent）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入你的 agent 函数
from agent import query_ecu_agent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecu-agent-api")

app = FastAPI(
    title="ECU Technical Q&A Agent API",
    description="基于 LangGraph + RAG 的 ECU 技术问答服务",
    version="1.0"
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    向 ECU Agent 提问技术问题
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        logger.info(f"📥 Received question: {request.question}")
        answer = query_ecu_agent(request.question)
        logger.info("✅ Answer generated successfully")
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent failed to generate answer: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "agent": "ready"}