import os 
import mlflow
import pandas as pd

def init():
    global model 

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = mlflow.pyfunc.load_model(model_path)

    def run(mini_batch):
        print(f"run method start: {__file__}, run({len(mini_batch)} files)")
        resultList = []

        for filepath in mini_batch:
            data = pd.read_csv(filepath)
            pred = model.predict(data)

            df = pd.DataFrame(pred, columns=['Predictions'])
            df['file'] = os.path.basename(filepath)
            resultList.extend(df.values)

        return resultList
