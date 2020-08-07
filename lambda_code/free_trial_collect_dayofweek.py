import boto3
import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

resource_client = boto3.client('resourcegroupstaggingapi')

def lambda_handler(event, context):
    dayofweek = datetime.datetime.today().weekday()
    
    if dayofweek == 4: # Sunday is 6
        retrain = 'True'
    else:
        retrain = 'False'
        
    # confirm that there exists an xgboost model 
    response = resource_client.get_resources(
    TagFilters=[
        {
            'Key': 'model',
            'Values': [
                'free_trial_xgboost'
            ]
        },
    ],
    ResourcesPerPage=100,
    ResourceTypeFilters=['sagemaker:model']    
    )
    
    if len(response['ResourceTagMappingList']) == 0: # no model exists
        retrain = 'True'

    return {
        'statusCode': 200,
        'retrain': retrain
    }