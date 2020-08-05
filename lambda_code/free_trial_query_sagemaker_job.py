import boto3
import csv
import logging
import json
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm_client = boto3.client('sagemaker')
    
def lambda_handler(event, context):
    TransformJobName = event['TransformJobName']
    
    # query the transform job status
    response = sm_client.describe_transform_job(TransformJobName=TransformJobName)
    job_status = response['TransformJobStatus']

    return {
        'statusCode': 200,
        'JobStatus': job_status
    }