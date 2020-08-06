import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    dayofweek = datetime.datetime.today().weekday()
    
    if dayofweek == 2: # Sunday is 6
        retrain = 'True'
    else:
        retrain = 'False'

    return {
        'statusCode': 200,
        'retrain': retrain
    }