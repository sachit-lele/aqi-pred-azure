# batch_inference.py

from azureml.core import Workspace, Dataset, Model, Experiment, Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
import os

ws = Workspace.get(
    name='myWorkSpace',
    subscription_id='946966e0-6b02-4b92-89d2-c2e3bb3604c7',
    resource_group='myResource',
)

experiment = Experiment(workspace=ws, name='air-quality-index-experiment01')

clustername = 'lelesachitcluster'
compute_target = ComputeTarget(workspace=ws, name=clustername)

env = Environment.get(workspace=ws, name='CustomEnv01')

input_data_path = (Datastore.get(ws, 'aqi_pred_datastore'), 'test.csv')


# Inference Config
inference_config = InferenceConfig(entry_script="inference.py",
                                   environment=env)

# Define PipelineData for output
predictions_output = PipelineData("predictions", datastore=datastore)

# Define Pipeline Step
inference_step = PythonScriptStep(
    name="Batch Inference",
    script_name="inference.py",
    arguments=["--input-data", dataset.as_named_input('input_dataset').as_mount(),
               "--model-name", model.name,
               "--output", predictions_output],
    outputs=[predictions_output],
    compute_target=compute_target,
    source_directory='src/model',  # Directory containing inference.py and other scripts
    runconfig=inference_config
)

# Create Pipeline
pipeline = Pipeline(workspace=ws, steps=[inference_step])

# Create Experiment
experiment = Experiment(ws, 'batch_inference_experiment')

# Submit Pipeline
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

# Download Predictions
predictions_path = predictions_output.download(target_path='.', overwrite=True)
print(f"Predictions downloaded to {predictions_path}")