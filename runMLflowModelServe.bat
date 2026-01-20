@echo off
echo ========================================
echo MLflow Model Serve Starter
echo ========================================

REM 设置环境变量
set __pyfunc_model_path__=%CD%\models\test_model
set MODEL_PATH=%CD%\models\test_model
set MLFLOW_ENABLE_CONDA=false
set MLFLOW_DISABLE_ENV_CREATION=true
set PYTHONPATH=C:\cc\AlexLearning\AIdemos\ECUagent\ecu_agent\models\test_model\code\src

echo.
echo   model_path: %MODEL_PATH%
echo   forbid Conda: true
echo  PYTHONPATH=C:\cc\AlexLearning\AIdemos\ECUagent\ecu_agent\models\test_model\code\src
echo.

REM 启动服务器
echo mlflow models serve -m models/test_model -p 5001 --no-conda
echo or
echo uvicorn --host 127.0.0.1 --port 5001 --workers 1 mlflow.pyfunc.scoring_server.app:app
echo.

uvicorn --host 127.0.0.1 --port 5001 --workers 1 --app-dir "C:\cc\AlexLearning\AIdemos\ECUagent\ecu_agent" mlflow.pyfunc.scoring_server.app:app
