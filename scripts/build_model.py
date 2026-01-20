# scripts/build_model.py
import mlflow
import sys
import os
from pathlib import Path

# 将 src/ 目录加入模块搜索路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
from model import testModel

def build_model():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    src_dir = project_root / "src"
    chroma_db_dir = project_root / "chroma_db"

    model=testModel()
    output_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)
    print(f"saving model to: {output_dir}")

    mlflow.pyfunc.save_model(
        path=str(output_dir / "test_model"),
        python_model=model,
        artifacts={"chroma_root": str(chroma_db_dir)},  # 注册外部文件
        code_paths=[str(src_dir)],                        # 当前目录包含源码
        conda_env=None,                      # 避免 conda 依赖
        pip_requirements=["mlflow>=2.0.0"]
    )

    # print(f"testModel is saved.")
    # print("\n📁 模型文件结构:")
    # model_path = output_dir / "test_model"
    # for root, dirs, files in os.walk(model_path):
    #     level = root.replace(str(model_path), '').count(os.sep)
    #     indent = ' ' * 2 * level
    #     print(f'{indent}{os.path.basename(root)}/')
    #     subindent = ' ' * 2 * (level + 1)
    #     for file in files:
    #         print(f'{subindent}{file}')

def test_model_loading():
    """测试加载保存的模型"""
    print("\n🧪 测试模型加载...")    
    model_path = Path(__file__).parent.parent / "models" / "test_model"    
    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return
    
    try:
        # 加载模型
        loaded_model = mlflow.pyfunc.load_model(str(model_path))
        print("✅ 模型加载成功!")
        
        # 测试预测
        test_input = ["Hello MLflow", "测试输入", "test input 123"]
        print(f"测试输入: {test_input}")
        
        result = loaded_model.predict(test_input)
        print(f"预测结果: {result}")
        
    except Exception as e:
        print(f"❌ 加载/测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 50)
    print("开始构建测试模型...")
    print("=" * 50)
    
    # 构建模型
    build_model()    
    print("\n" + "=" * 50)
    
    # 测试模型加载
    test_model_loading()
