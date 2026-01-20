# scripts/build_model.py
import mlflow
from model import ECURAGModel

# 创建模型实例
model = ECURAGModel()

# 保存到固定目录（非 mlruns！）
mlflow.pyfunc.save_model(
    path="models/ecu_rag_model",
    python_model=model,
    artifacts={"chroma_root": "chroma_db"},  # 注册外部文件
    code_path=["src"],                        # 包含源码
    conda_env=None                            # 避免 conda 依赖
)