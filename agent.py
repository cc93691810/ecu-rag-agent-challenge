from typing import TypedDict, List, Literal
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
import mlflow
import time
from rag import get_vectorstore

# 设置本地 MLflow 跟踪 URI（默认就是 ./mlruns，可省略）
mlflow.set_tracking_uri("./mlruns")  # 可选，显式指定
mlflow.set_experiment("ECU-mlflow-test")

# ======================
# 1. 定义 Agent 状态
# ======================
class ECUAgentState(TypedDict):
    """ECU Agent 的全局状态"""
    user_question: str                          # 用户输入的问题
    series_to_query: Literal["700", "800b", "800p", "unknown"]  # 路由决策
    retrieved_docs: List[Document]              # 检索到的文档
    final_answer: str                           # 最终回答

# ======================
# 2. 初始化 LLM（全局复用）
# ======================
llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.0,
    num_predict=256,
    timeout=120
)

# ======================
# 3. 定义节点函数
# ======================
def route_question(state: ECUAgentState) -> dict:
    """根据问题内容决定查询哪个 ECU 系列"""
    q = state["user_question"].lower()
    user_question = state["user_question"]  # 保持原始大小写用于显示
    print(f"🔍 Analyzing question: '{user_question}'")

    # 检测问题中涉及的型号
    has_700 = any(kw in q for kw in ["700", "750", "legacy"])
    has_800P = any(kw in q for kw in ["800b", "800 p", "plus", "800 plus", "850b", "ECU-850b"])
    has_800B = any(kw in q for kw in ["800 ", "base", "800 base", "850", "ECU-850"])
    # 统计匹配的系列数
    series_count = sum([has_700, has_800B, has_800P])

    # 检测比较意图的关键字
    comparison_keywords = [
        "compare", "comparison", "difference", "vs", "versus", "between", 
        "vs.", "comparing", "contrast", "differences", "vs ", " versus ", "and"
    ]
    is_comparison = any(keyword in q for keyword in comparison_keywords)
    # 检测通用查询（涉及多个型号）
    general_keywords = [
        "which ", "all ", "models", "How many", "est "
    ]
    is_general = any(keyword in q for keyword in general_keywords)

    # 处理比较问题
    if is_comparison and series_count >= 2:
        # 确定涉及哪些系列
        multi_series = []
        if has_700:
            multi_series.append("700")
        if has_800B:
            multi_series.append("800B")
        if has_800P:
            multi_series.append("800P")
        result = f"multi:{','.join(multi_series)}"
        print(f"🎯 Route: '{user_question}' -> series_to_query: '{result}' (comparison detected)")
        return {"series_to_query": result}

    # 处理通用查询（需要跨多个系列查找）
    if is_general:
        # 检查问题中是否明确指定了型号范围
        if series_count > 0:
            multi_series = []
            if has_700:
                multi_series.append("700")
            if has_800B:
                multi_series.append("800B")
            if has_800P:
                multi_series.append("800P")
            if len(multi_series) > 1:
                result = f"multi:{','.join(multi_series)}"
                print(f"🎯 Route: '{user_question}' -> series_to_query: '{result}' (general query across series)")
                return {"series_to_query": result}
            if len(multi_series) == 1:
                result = multi_series[0]
                matched_kw = [kw for kw in ["700", "850 ", "850b", "750"] if kw in q.lower()][0]
                print(f"🎯 Route: '{user_question}' -> series_to_query: '{result}' (matched: {matched_kw})")
                return {"series_to_query": result}
        else:
            # 通用查询但未指定具体型号，查询所有系列
            result = "multi:700,800B,800P"
            print(f"🎯 Route: '{user_question}' -> series_to_query: '{result}' (general query across ALL series)")
            return {"series_to_query": result}

    # 单一系列查询
    if has_700:
        result = "700"
        matched_kw = [kw for kw in ["700", "750", "legacy"] if kw in q][0]
        print(f"🎯 Route: '{user_question}' -> series_to_query: '{result}' (matched: {matched_kw})")
        return {"series_to_query": result}
    if has_800B:
        result = "800B"
        # 找到匹配的关键词
        matched_kws = [kw for kw in ["800 ", "base", "800 base", "850", "ecu-850 "] if kw in q]
        matched_kw = matched_kws[0] if matched_kws else "850"
        print(f"🎯 Route: '{user_question}' -> series_to_query: '{result}' (matched: {matched_kw})")
        return {"series_to_query": result}
    if has_800P:
        result = "800P"
        matched_kws = [kw for kw in ["800p", "800 p", "plus", "800 plus", "850b", "ecu-850b "] if kw in q]
        matched_kw = matched_kws[0] if matched_kws else "850b"
        print(f"🎯 Route: '{user_question}' -> series_to_query: '{result}' (matched: {matched_kw})")
        return {"series_to_query": result}
    else:
        result = "unknown"
        print(f"🎯 Route: '{user_question}' -> series_to_query: '{result}' (no match found)")
        return {"series_to_query": result}

def retrieve_documents(state: ECUAgentState) -> dict:
    """从本地 ChromaDB 检索相关文档"""
    series = state["series_to_query"]
    all_docs=[]
    if series == "unknown":
        # 从所有系列中检索（用于通用查询）
        print("🔄 Unknown series detected - retrieving from ALL series")
        try:
            for s in ["700", "800B", "800P"]:
                print(f"  ➤ Retrieving from series {s}...")
                vectorstore = get_vectorstore(s)
                docs = vectorstore.similarity_search(
                    state["user_question"],
                    k=2  # 从每个系列取2个最相关的 chunks
                )
                # 添加系列标签到元数据，便于后续区分
                for doc in docs:
                    doc.metadata["series"] = s
                all_docs.extend(docs)
                print(f"    Retrieved {len(docs)} docs from series {s}")
        except Exception as e:
            print(f"⚠️ Universal retrieval failed: {e}")
            return {"retrieved_docs": []}

    # 检查是否为多系列查询格式
    elif series.startswith("multi:"):
        # 解析多个系列
        series_list = series[6:].split(",")  # "multi:800B,800P" -> ["800B", "800P"]
        print(f"🔄 Multi-series retrieval: {series_list}")

        try:
            for s in series_list:
                print(f"  ➤ multi: Retrieving from series {s}...")
                vectorstore = get_vectorstore(s)
                docs = vectorstore.similarity_search(
                    state["user_question"],
                    k=2  # 每个系列取 3 个最相关的 chunks
                )
                # 添加系列标签到元数据，便于后续区分
                for doc in docs:
                    doc.metadata["series"] = s
                all_docs.extend(docs)
                print(f" Retrieved {len(docs)} docs from series {s}")
        except Exception as e:
            print(f"⚠️ Multi-retrieval failed: {e}")
            return {"retrieved_docs": []}

    else:
        # 单一系列查询（原逻辑）
        try:
            vectorstore = get_vectorstore(series)
            docs = vectorstore.similarity_search(
                state["user_question"],
                k=2
            )
            # 为单一系列也添加系列标签
            for doc in docs:
                doc.metadata["series"] = series
            all_docs = docs
        except Exception as e:
            print(f"⚠️ Single-retrieval failed for series {series}: {e}")
            return {"retrieved_docs": []}

    print(f"✅ Retrieved {len(all_docs)} documents from {len(set(d.metadata.get('series') for d in all_docs))} series")
    return {"retrieved_docs": all_docs}

def generate_answer(state: ECUAgentState) -> dict:
    """基于检索结果生成最终回答"""
    question = state["user_question"]
    docs = state["retrieved_docs"]

    if not docs:
        answer = "I don't have technical information about this ECU model."
    else:
        context = docs[0].page_content

        prompt = ChatPromptTemplate.from_template(
            """You are an expert automotive engineer assistant.
            Answer the question based ONLY on the following context.
            Do not make up information. If unsure, say "I don't know".

            Context:
            {context}

            Question: {question}
            Answer:"""
        )

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

    return {"final_answer": answer}

# ======================
# 4. 构建并返回 LangGraph Agent
# ======================
def build_ecu_agent():
    """
    构建并返回一个可执行的 LangGraph ECU 技术问答 Agent。
    
    返回:
        Runnable: 可通过 .invoke({"user_question": "..."}) 调用的 Agent
    """
    workflow = StateGraph(ECUAgentState)

    # 添加节点
    workflow.add_node("route", route_question)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)

    # 设置入口和边
    workflow.set_entry_point("route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # 编译为可执行应用
    app = workflow.compile()
    return app

# 便捷回复
def query_ecu_agent(question: str) -> str:
    """便捷函数：输入问题，返回答案"""
    app = build_ecu_agent()
    result = app.invoke({
        "user_question": question,
        "series_to_query": "unknown",
        "retrieved_docs": [],
        "final_answer": ""
    })
    return result["final_answer"]

# ----------------------------
# 5. Orchestrator (with MLflow)
# ----------------------------
def run_ecu_agent_with_mlflow(user_question: str) -> dict:
    """完整执行流程 + MLflow 日志"""

    # 定义默认返回值
    final_state = {
        "user_question": user_question,
        "series_to_query": "unknown",
        "retrieved_docs": [],
        "final_answer": "处理中...",
        "success": False,
        "error": None
    }

    try:
        with mlflow.start_run(run_name=f"Q: {user_question[:40]}..."):
            # 记录开始时间
            start_time = time.time()

            # 记录参数
            mlflow.log_param("question", user_question)
            mlflow.log_param("question_length", len(user_question))

            # 构建代理
            app = build_ecu_agent()

            # 准备初始状态
            initial_state = {
                "user_question": user_question,
                "series_to_query": "unknown",
                "retrieved_docs": [],
                "final_answer": ""
            }

            # 执行代理
            agent_result = app.invoke(initial_state)  # 使用不同的变量名

            # 更新 final_state
            final_state.update(agent_result)
            final_state["success"] = True

            # 计算执行时间
            execution_time = time.time() - start_time
            final_state["execution_time"] = execution_time

            # 记录指标
            mlflow.log_metric("execution_time", execution_time)
            mlflow.log_metric("docs_retrieved", len(final_state.get("retrieved_docs", [])))
            mlflow.log_metric("answer_length", len(final_state.get("final_answer", "")))

            # 记录模型输入输出
            mlflow.log_dict({
                "input": initial_state,
                "output": final_state
            }, "input_output.json")

            # 记录最终状态
            mlflow.log_dict(final_state, "final_state.json")

    except Exception as error:
        # 错误处理
        final_state.update({
            "success": False,
            "error": str(error),
            "final_answer": f"抱歉，处理问题时出错: {error}"
        })

        # 记录错误
        if 'mlflow' in locals():
            mlflow.log_param("error", str(error))
            mlflow.log_metric("error_occurred", 1)

    # 确保总是返回 final_state
    return final_state

###################################################
# def build_ecu_agent_HF():
#     # 使用 flan-t5-small（更快，适合测试）
#     generator = pipeline(
#         "text2text-generation",
#         model="google/flan-t5-small",
#         max_new_tokens=200,
#         do_sample=False,
#         device=0 if torch.cuda.is_available() else -1
#     )
#     llm = HuggingFacePipeline(pipeline=generator)
#     tools = [query_ecu_700_series, query_ecu_800_series]
#     return create_react_agent(llm, tools)

# def build_ecu_agent_llama():
#     # 使用本地 Ollama 服务调用 llama3.1:8b
#     llm = ChatOllama(
#         model="llama3.1:8b",          # 必须与 ollama list 中的名称一致
#         base_url="http://localhost:11434",  # Ollama 默认地址
#         temperature=0.0,              # 降低随机性，适合问答
#         timeout=120,                  # 防止长响应超时
#         num_predict=256,               # 最大生成长度
#     )
#     tools = [query_ecu_700_series, query_ecu_800_series]
#     return langchain.agents.create_react_agent(llm,tools)
