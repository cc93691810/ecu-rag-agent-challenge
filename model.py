import mlflow.pyfunc
import os
from langchain_chroma import Chroma
from langchain_community.llms import Ollama

class ECURAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self, chroma_root="chroma_db"):
        self.chroma_root = chroma_root

    def load_context(self, context):
        """在模型加载时执行一次"""
        # 从 artifacts 获取实际路径（支持 Docker）
        if hasattr(context, "artifacts") and "chroma_root" in context.artifacts:
            self.chroma_root = context.artifacts["chroma_root"]

        # 初始化 LLM（宿主机运行 Ollama）
        self.llm = Ollama(model="llama3.1:8b")

        # 加载所有系列的向量库
        self.vectorstores = {}
        for series in ["700", "800B", "800P"]:
            path = os.path.join(self.chroma_root, f"ecu_{series.lower()}")
            if os.path.exists(path):
                self.vectorstores[series] = Chroma(persist_directory=path)

    def predict(self, context, model_input):
        """每次 API 调用时执行"""
        question = model_input["question"][0]
        
        # 1. 路由
        series = self._route_question(question)
        
        # 2. 检索
        docs = []
        if series == "all":
            for vs in self.vectorstores.values():
                docs.extend(vs.similarity_search(question, k=2))
        elif series in self.vectorstores:
            docs = self.vectorstores[series].similarity_search(question, k=3)
        
        # 3. 生成
        context_text = "\n".join([d.page_content for d in docs])
        prompt = f"Answer based on context:\n{context_text}\n\nQuestion: {question}"
        answer = self.llm.invoke(prompt)
        
        return [answer] 
    
    def _route_question(self, q):
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