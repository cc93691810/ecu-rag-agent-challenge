import mlflow

class testModel(mlflow.pyfunc.PythonModel):
    def predict(self, model_input: list[str], params=None) -> list[str]:
        return model_input