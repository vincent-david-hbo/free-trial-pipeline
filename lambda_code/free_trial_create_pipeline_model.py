import boto3
import csv
import logging
import json
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm_client = boto3.client('sagemaker')

def collect_models():
    # query for the most recently trained models
        # initiate empty dictionary for response
    response = {}
    
        # free_trial_sklearn_featurizer
    search_params={
       "Resource": "TrainingJob",
       "SearchExpression": { 
          "Filters": [{ 
                "Name": "Tags.model",
                "Operator": "Equals",
                "Value": "free_trial_sklearn_featurizer"
             }]}
    }

    results = sm_client.search(**search_params)['Results']
    
        # find the most recently trained model
    most_recent_creation = results[0]['TrainingJob']['CreationTime']
    most_recent_item = 0
    for i in range(1, len(results)):
        creation = results[i]['TrainingJob']['CreationTime']
        if creation > most_recent_creation:
            most_recent_creation = most_recent_creation
            most_recent_item = i

    response['SKLearnModelData'] = results[most_recent_item]['TrainingJob']['ModelArtifacts']['S3ModelArtifacts']
    response['SKLearnImage'] = results[most_recent_item]['TrainingJob']['AlgorithmSpecification']['TrainingImage']

    
    # free_trial_xgboost
    search_params={
       "Resource": "TrainingJob",
       "SearchExpression": { 
          "Filters": [{ 
                "Name": "Tags.model",
                "Operator": "Equals",
                "Value": "free_trial_xgboost"
             }]}
    }

    results = sm_client.search(**search_params)['Results']
    
        # find the most recently trained model
    most_recent_creation = results[0]['TrainingJob']['CreationTime']
    most_recent_item = 0
    for i in range(1, len(results)):
        creation = results[i]['TrainingJob']['CreationTime']
        if creation > most_recent_creation:
            most_recent_creation = most_recent_creation
            most_recent_item = i

    response['XGBoostModelData'] = results[most_recent_item]['TrainingJob']['ModelArtifacts']['S3ModelArtifacts']
    response['XGBoostImage'] = results[most_recent_item]['TrainingJob']['AlgorithmSpecification']['TrainingImage']

    
    return response

def lambda_handler(event, context):
    PipelineModelName = event['PipelineModelName']
    Role = event['Role']
    
    # collect best model detail
    models = collect_models()
    
    # holder for env vars
    SAGEMAKER_SPARKML_SCHEMA = ''
    
    # create a new PipelineModel
    response = sm_client.create_model(
     ModelName=PipelineModelName
    ,ExecutionRoleArn=Role
    ,Containers=[
              {
                  'ContainerHostname': 'SKLearnPreProcessing'
                  ,'Image': models['SKLearnImage']
                  ,'Mode': 'SingleModel' # 'MultiModel'
                  ,'ModelDataUrl': models['SKLearnModelData']
                  ,'Environment': {
                      'SAGEMAKER_SPARKML_SCHEMA': SAGEMAKER_SPARKML_SCHEMA
                   }
              },
             {
                  'ContainerHostname': 'XGBoostModel'
                  ,'Image': models['XGBoostImage']
                  ,'Mode': 'SingleModel' # 'MultiModel'
                  ,'ModelDataUrl': models['XGBoostModelData']
              }
          ]
    ,Tags=[
        {
            'Key': 'model',
            'Value': 'free_trial_pipeline'
        }
    ]
    )
    
    return {
        'statusCode': 200,
        'model_name': PipelineModelName
    }