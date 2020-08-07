import boto3
import csv
import datetime
import logging
import json
import time
import re

from time import gmtime, strftime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm_client = boto3.client('sagemaker')
resource_client = boto3.client('resourcegroupstaggingapi')

def findLatestModel():
    # search for our tagged featurizer models
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
    
    # find the most recent entry
    most_recent_string = '2000-01-01-01-01-01'
    most_recent_obj = datetime.datetime.strptime(most_recent_string, "%Y-%m-%d-%H-%M-%S") 

    for model in response['ResourceTagMappingList']:
        arn = model['ResourceARN']
        date_string = re.search('[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])-(2[0-3]|[01][0-9])-[0-5][0-9]-[0-5][0-9]', arn).group()
        date_obj = datetime.datetime.strptime(date_string, "%Y-%m-%d-%H-%M-%S") 

        if date_obj > most_recent_obj:
            most_recent_obj = date_obj
            most_recent_string = date_string
            
    return most_recent_string

def lambda_handler(event, context):
    bucket = event['bucket']
    TransformJobName = event['TransformJobName']

    # collect most recent model detail
    most_recent_string = findLatestModel()

    response = sm_client.create_transform_job(
        TransformJobName=TransformJobName,
        ModelName=f'free-trial-xgboost-model-{most_recent_string}',
        MaxPayloadInMB=6, # default
        BatchStrategy='SingleRecord', #'MultiRecord'
        TransformInput={
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': f's3://{bucket}/free_trial_model/snowflake-hbomax-staging/inference/transformed' 
                }
            },
            'ContentType': 'text/csv',
            'SplitType': 'Line'
        },
        TransformOutput={
            'S3OutputPath': f's3://{bucket}/free_trial_model/snowflake-hbomax-staging/results',
            'Accept': 'text/csv',
            'AssembleWith': 'Line',
            #'KmsKeyId': 'alias/aws/s3'
        },
        TransformResources={
            'InstanceType': 'ml.m4.2xlarge', #'ml.m4.10xlarge' 
            'InstanceCount': 1,
            #'VolumeKmsKeyId': 'alias/aws/s3'
        },
        DataProcessing={
                'InputFilter': '$[1:]',
                #'OutputFilter': '$[0,-259:]',
                'JoinSource': 'Input'
            }
    )
    
    return {
        'statusCode': 200,
        'most_recent_string': most_recent_string
    }