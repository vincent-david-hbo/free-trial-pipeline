import boto3
import csv
import logging
import json
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm_client = boto3.client('sagemaker')
    
def lambda_handler(event, context):
    job_name = event['job_name']
    job_type = event['job_type']
    
    if job_type == 'Transform':
        # query the transform job status
        response = sm_client.describe_transform_job(TransformJobName=job_name)
        job_status = response['TransformJobStatus']
        
    elif job_type == 'Train':
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        job_status = response['TrainingJobStatus']
        
        job_status = {}
        
        if 'FinalMetricDataList' in response:
            #We can't marshall datetime objects in JSON response. So convert
            #all datetime objects returned to unix time.
            for index, metric in enumerate(response['FinalMetricDataList']):
                metric['Timestamp'] = metric['Timestamp'].timestamp()

            job_status['trainingMetrics'] = response['FinalMetricDataList']
            job_status['TrainingJobName'] = response['TrainingJobName']
            job_status['TrainingJobArn'] = response['TrainingJobArn']
            job_status['ModelArtifacts'] = response['ModelArtifacts']
            job_status['HyperParameters'] = response['HyperParameters']

            test_auc = [m['Value'] for m in response['FinalMetricDataList'] if m['MetricName'] == 'validation:auc'][0]
            if test_auc > .1:
                job_status['MeetsThreshold'] = 'True'
            else:
                job_status['MeetsThreshold'] = 'False'
            
    return {
        'statusCode': 200,
        'JobStatus': job_status
    }