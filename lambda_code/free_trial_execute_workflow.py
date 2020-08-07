import boto3
from time import gmtime, strftime
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sfn_client = boto3.client('stepfunctions')

def lambda_handler(event, context):
    # collect the ARN
    response = sfn_client.list_state_machines()
    arn = [m['stateMachineArn'] for m in response['stateMachines'] if m['name'] == 'FTInferenceRoutine'][0]
    
    # define inputs
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    inputs={
        'SKLearnFeaturizerJobName': 'free-trial-sklearn-featurizer-{}'.format(timestamp_prefix),
        'TransformTrainJobName': 'free-trial-transform-train-{}'.format(timestamp_prefix),
        'TransformTestJobName': 'free-trial-transform-test-{}'.format(timestamp_prefix),
        'FeaturizerModelName': 'free-trial-featurizer-model-{}'.format(timestamp_prefix),
        'XGBModelName': 'free-trial-xgboost-model-{}'.format(timestamp_prefix),
        'TrainXGBoostJobName': 'free-trial-train-xgboost-{}'.format(timestamp_prefix),
        'PipelineModelName': 'free-trial-pipeline-model-{}'.format(timestamp_prefix),
        'TransformNewJobName': 'free-trial-transform-new-{}'.format(timestamp_prefix),
        'TransformXGBoostJobName': 'free-trial-transform-xgboost-{}'.format(timestamp_prefix),
        'TimestampPrefix': timestamp_prefix
    }
    
    # execute workflow
    response = sfn_client.start_execution(
        stateMachineArn=arn,
        input=json.dumps(inputs)
    )
    
    return {
        'statusCode': 200
    }