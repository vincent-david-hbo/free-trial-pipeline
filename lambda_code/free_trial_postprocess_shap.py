import boto3 
import logging
from botocore.client import Config

s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    bucket = event['bucket']
    
    # collect column headers
    s3_client.download_file(bucket, 'free_trial_model/snowflake-hbomax-staging/schema/column_headers.csv', '/tmp/column_headers.csv')
    column_headers_file = open('/tmp/column_headers.csv', 'r') 
    column_headers_lines = column_headers_file.readlines()
    column_headers = []
    for line in column_headers_lines:
        column_headers.append(line.strip())
        
    column_headers = ['HBO_UUID','PREDICTION','SHAP_EXPECTED'] + column_headers
    
    # collect files
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix ='free_trial_model/snowflake-hbomax-staging/results')
    
    files = [r['Key'] for r in response['Contents'] if r['Size'] > 0]
    
    for file in files:
        s3_client.download_file(bucket, file, '/tmp/results.csv')
    
    results_file = open('/tmp/results.csv', 'r') 
    results_lines = results_file.readlines()
    
    # format results
    output = []
    for line in results_lines:
        row = [x.strip() for x in line.split(',')]
        strip_row = row[:1] + row[-(len(column_headers) - 1):]
        output.append(dict(zip(column_headers, strip_row)))
        
    
    # write to S3
    filename = '/tmp/text.txt'

    with open(filename, mode="w") as outfile:  
        for item in output:
            outfile.write("%s\n" % item)
            
            
    # upload to S3
        # Braze
    response = s3_client.upload_file(filename, bucket, 'free_trial_model/snowflake-hbomax-staging/braze/SHAP_output', ExtraArgs={"ServerSideEncryption": "aws:kms"})
        # SnowFlake
    response = s3_client.upload_file(filename, bucket, 'free_trial_model/snowflake-hbomax-staging/snowflake/SHAP_output', ExtraArgs={"ServerSideEncryption": "aws:kms"})
    
    
    return {
        'statusCode': 200
    }