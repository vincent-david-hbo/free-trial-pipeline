from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = ['HBO_UUID',
 'PROVIDER',
 'PERIOD_RANK',
 'NUM_PROFILE',
 'NUM_ADULT_PROFILE',
 'NUM_KID_PROFILE',
 'FLG_TURN_OFF_AUTORENEW',
 'TOTAL_HBONOW_WATCH_SEC_ADJ',
 'ROKU_PERCENT_ADJ_NOW',
 'PS_PERCENT_ADJ_NOW',
 'IPHONE_PERCENT_ADJ_NOW',
 'ANDROIDTV_PERCENT_ADJ_NOW',
 'APPLETV_PERCENT_ADJ_NOW',
 'ANROID_PERCENT_ADJ_NOW',
 'DESKTOP_PERCENT_ADJ_NOW',
 'TIZENTV_PERCENT_ADJ_NOW',
 'XBOX_PERCENT_ADJ_NOW',
 'FIRETV_PERCENT_ADJ_NOW',
 'IPAD_PERCENT_ADJ_NOW',
 'TOTAL_WATCH_SEC_ADJ',
 'NUM_ENGAGED_STREAM',
 'NUM_NOT_ENGAGED_STREAM',
 'NUM_DAYS_WITH_VIEWING',
 'NUM_DAYS_WITH_VIEWING_ADJ',
 'MIN_SEQ_ENGAGED_STREAM',
 'NUM_STREAMS',
 'NUM_STREAMS_ADJ',
 'NUM_DEVICE_ADJ',
 'NUM_EDIT_ID_ADJ',
 'NUM_VIEWABLE_ID_ADJ',
 'NUM_SERIES_ID_ADJ',
 'NUM_SEASON_ID_ADJ',
 'NUM_SEASON_ID_PREMIERE_ADJ',
 'NUM_SEASON_ID_FINALE_ADJ',
 'NUM_SERIES_ID_PREMIERE_ADJ',
 'NUM_SERIES_ADJ',
 'NUM_MOVIES_ADJ',
 'NUM_SERIES_GENRE_ADJ',
 'NUM_MOVIE_GENRE_ADJ',
 'NUM_OVERALL_GENRE_ADJ',
 'AVG_DAYS_BTWN_RELEASE_WATCHED',
 'AVG_DAYS_BTWN_RELEASE_WATCHED_MOVIES',
 'AVG_DAYS_BTWN_RELEASE_WATCHED_SERIES',
 'PERCENT_SEASON_ID_PREMIERE',
 'PERCENT_SEASON_ID_FINALE',
 'PERCENT_SERIES_ID_PREMIERE',
 'ROKU_PERCENT_ADJ',
 'PS_PERCENT_ADJ',
 'IPHONE_PERCENT_ADJ',
 'ANDROIDTV_PERCENT_ADJ',
 'APPLETV_PERCENT_ADJ',
 'ANROID_PERCENT_ADJ',
 'DESKTOP_PERCENT_ADJ',
 'TIZENTV_PERCENT_ADJ',
 'XBOX_PERCENT_ADJ',
 'FIRETV_PERCENT_ADJ',
 'IPAD_PERCENT_ADJ',
 'LG_SCRN_PERCENT_ADJ',
 'SM_SCRN_PERCENT_ADJ',
 'ALL_MOVIES_PERCENT_ADJ',
 'ALL_SERIES_PERCENT_ADJ',
 'ALL_OTHER_CLASS_PERCENT_ADJ',
 'MOVIE_ACTN_ADVNTR_PERCENT_ADJ',
 'MOVIE_COMEDY_PERCENT_ADJ',
 'MOVIE_DOCUMENTARY_PERCENT_ADJ',
 'MOVIE_DRAMA_PERCENT_ADJ',
 'MOVIE_KIDS_FAM_PERCENT_ADJ',
 'MOVIE_FNTSY_SCIFI_PERCENT_ADJ',
 'MOVIE_THRILLER_PERCENT_ADJ',
 'MOVIE_MISC_PERCENT_ADJ',
 'SERIES_COMEDY_PERCENT_ADJ',
 'SERIES_ACTN_ADVNTR_PERCENT_ADJ',
 'SERIES_GOT_PERCENT_ADJ',
 'SERIES_DRAMA_PERCENT_ADJ',
 'SERIES_KIDS_FAM_PERCENT_ADJ',
 'SERIES_ENTMNT_NEWS_PERCENT_ADJ',
 'SERIES_MISC_PERCENT_ADJ',
 'OVERALL_DRAMA_PERCENT_ADJ',
 'OVERALL_COMEDY_PERCENT_ADJ',
 'OVERALL_GOT_PERCENT_ADJ',
 'OVERALL_KIDS_FAM_PERCENT_ADJ',
 'OVERALL_FNTSY_SCIFI_PERCENT_ADJ',
 'OVERALL_DOCUMENTARY_PERCENT_ADJ',
 'OVERALL_ENTMNT_NEWS_PERCENT_ADJ',
 'OVERALL_ACTN_ADVNTR_PERCENT_ADJ',
 'OVERALL_SPORTS_PERCENT_ADJ',
 'OVERALL_THRILLER_PERCENT_ADJ',
 'OVERALL_MISC_PERCENT_ADJ',
 'MOVIE_GENRE_ENTROPY',
 'SERIES_GENRE_ENTROPY',
 'OVERALL_GENRE_ENTROPY',
 'TOTAL_WATCH_MINS_PER_DAY',
 'TOTAL_WATCH_MINS_PER_DAY_ADJ',
 'AVG_MIN_PER_STREAMS',
 'NUM_DAYS_BTWN_FT_START_FIRST_STREAM',
 'NUM_DAYS_BTWN_FT_START_FIRST_STREAM_ADJ',
 'MOBILE_PERCENT',
 'PERC_TOP_3_SERIES_ADJ',
 'PERC_TOP_1_SERIES_ADJ',
 'AVG_SERIES_COMPLETION_RATE_MAXEPISODE',
 'AVG_SERIES_CONTINUE_WATCHING_RATE',
 'NUM_LOYALSERIES_2EP_90_CONTINUE_ONAIR',
 'NUM_LOYALSERIES_2EP',
 'NUM_EPISODES_COMPLETE_80_PERCENT_ADJ',
 'NUM_EPISODES_EVER_WATCHED_ADJ',
 'NUM_SERIES_LOYAL_W_EPISODE_ON_AIR_BILLINGDT',
 'FLG_COMP_2EP',
 'FLG_COMPLETE_SAMEDAY',
 'FLG_COMPLETE_SAMEWEEK',
 'FLG_COMPLETE_WITH2WEEK_SEASONEND',
 'NUM_SERIES_TRIED',
 'NUM_SERIES_LOYAL',
 'NUM_SERIES_LOYAL_START_IN_MIDDLE',
 'NUM_SERIES_LOYAL_COMPLETED_MAXEPISODE',
 'NUM_SERIES_LOYAL_NOTCOMPLETED_MAXEPISODE',
 'NUM_SERIES_LOYAL_COMPLETED_CNTEPISODE',
 'NUM_SERIES_LOYAL_NOTCOMPLETED_CNTEPISODE',
 'NUM_SERIES_LOYAL_INLIB_LATESTSEASON',
 'NUM_SERIES_LOYAL_INLIB_PREVSEASON',
 'PERC_BINGEING',
 'PERC_EPI_END_1DAY',
 'PERC_EPI_END_1WEEK_NOGFOT',
 'PERC_SEASON_END_2WEEK',
 'PERC_SEASON_END_1YEAR',
 'PERC_SEASON_END_OVER_1YEAR',
 'FLAG_SERIES_ENGAGED_FREE',
 'FIRST_WATCHED_ASSET_CLASS_SUB_ADJ',
 'NUM_SESSIONS',
 'NUM_SESSIONS_WITHOUT_STREAM',
 'PERC_SESSIONS_WITHOUT_STREAM',
 'AVG_TIME_TO_START_STREAM',
 'AVG_NAVIGATION_TIME',
 'AVG_SESSION_DURATION',
 'AVG_STREAMING_DURATION',
 'AVG_SESSION_UTILIZATION_PERC',
 'NUM_SESSIONS_W_SEARCH',
 'NUM_SESSIONS_W_STREAM_FOLLOWING_SEARCH',
 'NUM_SESSIONS_W_BROWSE',
 'NUM_SESSIONS_W_STREAM_FOLLOWING_BROWSE',
 'NUM_SESSIONS_W_TASTEMAKER',
 'NUM_SESSIONS_W_STREAM_FOLLOWING_TASTEMAKER',
 'NUM_SESSIONS_W_WATCHLIST',
 'NUM_SESSIONS_W_STREAM_FOLLOWING_WATCHLIST',
 'NUM_SESSIONS_W_ADD_TO_WATCHLIST',
 'NUM_SESSIONS_W_CONTINUE_WATCHING',
 'NUM_SESSIONS_W_MY_ACCOUNT',
 'NUM_SESSIONS_W_TURN_OFF_AUTO_RENEW',
 'NUM_SESSIONS_W_TURN_ON_AUTO_RENEW',
 'PERC_SESSIONS_W_SEARCH',
 'PERC_SESSIONS_W_STREAM_FOLLOWING_SEARCH_ALLSESSION',
 'PERC_SESSIONS_W_STREAM_FOLLOWING_SEARCH_PART',
 'PERC_SESSIONS_W_BROWSE',
 'PERC_SESSIONS_W_STREAM_FOLLOWING_BROWSE_ALLSESSION',
 'PERC_SESSIONS_W_STREAM_FOLLOWING_BROWSE_PART',
 'PERC_SESSIONS_W_TASTEMAKER',
 'PERC_SESSIONS_W_STREAM_FOLLOWING_TASTEMAKER_ALLSESSION',
 'PERC_SESSIONS_W_STREAM_FOLLOWING_TASTEMAKER_PART',
 'PERC_SESSIONS_W_WATCHLIST',
 'PERC_SESSIONS_W_STREAM_FOLLOWING_WATCHLIST_ALLSESSION',
 'PERC_SESSIONS_W_STREAM_FOLLOWING_WATCHLIST_PART',
 'PERC_SESSIONS_W_ADD_TO_WATCHLIST',
 'PERC_SESSIONS_W_CONTINUE_WATCHING',
 'PERC_SESSIONS_W_MY_ACCOUNT',
 'PERC_SESSIONS_W_TURN_OFF_AUTO_RENEW',
 'PERC_SESSIONS_W_TURN_ON_AUTO_RENEW',
 'FLAG_OF_STREAMING_VIDEOS_LAST_SESSION',
 'FLAG_OF_SEARCH_LAST_SESSION',
 'FLAG_OF_STREAM_FOLLOWING_SEARCH_LAST_SESSION',
 'FLAG_OF_BROWSE_LAST_SESSION',
 'FLAG_OF_STREAM_FOLLOWING_BROWSE_LAST_SESSION',
 'FLAG_OF_TASTEMAKER_LAST_SESSION',
 'FLAG_OF_STREAM_FOLLOWING_TASTEMAKER_LAST_SESSION',
 'FLAG_OF_WATCHLIST_LAST_SESSION',
 'FLAG_OF_STREAM_FOLLOWING_WATCHLIST_LAST_SESSION',
 'FLAG_OF_ADD_TO_WATCHLIST_LAST_SESSION',
 'FLAG_OF_CONTINUE_WATCHING_LAST_SESSION',
 'FLAG_OF_MY_ACCOUNT_LAST_SESSION',
 'FLAG_OF_TURN_OFF_AUTO_RENEW_LAST_SESSION',
 'FLAG_OF_TURN_ON_AUTO_RENEW_LAST_SESSION',
 'NUM_DAYS_SINCE_LAST_SESSION',
 'NUM_NAVIGATION_UI',
 'NUM_PROFILE_EVENTS',
 'NUM_COMMERCE_EVENTS',
 'NUM_MKTCAMPAIGN_SENT',
 'NUM_MKTCAMPAIGN_ENGAGED',
 'NUM_MKTCAMPAIGN_CONVERTED',
 'MKTCAMPAIGN_ENGAGEMENT_RATE',
 'MKTCAMPAIGN_CONVERSION_EVENT_RATE',
 'PROGRAM_TYPE_ORIGINAL_STREAMING_TIME_PERC_ADJ',
 'PROGRAM_TYPE_ORIGINAL_NUM_EPI_COMPLETED_80_PERC_ADJ',
 'PROGRAM_TYPE_ACQUIRED_STREAMING_TIME_PERC_ADJ',
 'PROGRAM_TYPE_ACQUIRED_NUM_EPI_COMPLETED_80_PERC_ADJ',
 'PROGRAM_TYPE_MISC_STREAMING_TIME_PERC_ADJ',
 'PROGRAM_TYPE_MISC_NUM_EPI_COMPLETED_80_PERC_ADJ',
 'CONTENT_SOURCE_HBO_STREAMING_TIME_PERC_ADJ',
 'CONTENT_SOURCE_HBO_NUM_EPI_COMPLETED_80_PERC_ADJ',
 'CONTENT_SOURCE_HBOMAX_STREAMING_TIME_PERC_ADJ',
 'CONTENT_SOURCE_HBOMAX_NUM_EPI_COMPLETED_80_PERC_ADJ',
 'CONTENT_SOURCE_MISC_STREAMING_TIME_PERC_ADJ',
 'CONTENT_SOURCE_MISC_NUM_EPI_COMPLETED_80_PERC_ADJ',
 'HBO_ORIGINAL_CONTENT_STREAMING_TIME_PERC_ADJ',
 'HBO_ORIGINAL_CONTENT_NUM_EPI_COMPLETED_80_PERC_ADJ',
 'HBOMAX_ORIGINAL_CONTENT_STREAMING_TIME_PERC_ADJ',
 'HBOMAX_ORIGINAL_CONTENT_NUM_EPI_COMPLETED_80_PERC_ADJ',
 'HBO_ACQUIRED_CONTENT_STREAMING_TIME_PERC_ADJ',
 'HBO_ACQUIRED_CONTENT_NUM_EPI_COMPLETED_80_PERC_ADJ',
 'HBOMAX_ACQUIRED_CONTENT_STREAMING_TIME_PERC_ADJ',
 'HBOMAX_ACQUIRED_CONTENT_NUM_EPI_COMPLETED_80_PERC_ADJ',
 'FRIENDS_STREAMING_SEC_ADJ',
 'FRIENDS_STREAMING_PERC_ADJ',
 'BBT_STREAMING_SEC_ADJ',
 'BBT_STREAMING_PERC_ADJ',
 'FT_SEGMENT',
 'FT_SUB_SEGMENT']

label_column = 'FLG_TARGET'



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe             
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) if '_COMPLETE' not in file]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    print('input files:', input_files)
        
    raw_data = [ pd.read_csv(
        file, 
        header=None, 
        #dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
        names=feature_columns_names + [label_column]) for file in input_files ]
    
    concat_data = pd.concat(raw_data)

    one_hot_encoder = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values='unknown', strategy='constant',fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    one_hot_encode_cols = ['PROVIDER', 'FIRST_WATCHED_ASSET_CLASS_SUB_ADJ', 'FT_SEGMENT', 'FT_SUB_SEGMENT']

    drop_cols = [label_column, 'HBO_UUID']
    
    preprocessor = ColumnTransformer(
            transformers=[
                ('one_hot', one_hot_encoder, one_hot_encode_cols),
                ('drop', 'drop', drop_cols)
            ],
            remainder='passthrough')


    preprocessor.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
    
def input_fn(input_data, content_type):
    """Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), 
                         header=None)
        
        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            df.columns = feature_columns_names
            
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        

def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """
    
    features = model.transform(input_data)

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        labeled_features = np.insert(features, 0, input_data[label_column], axis=1)

        # Drop the first row (headers)
        features_no_header = np.delete(labeled_features, 0, axis=0)

    else:
        # Drop the first row (headers)
        features_no_header = np.delete(features, 0, axis=0)

    return features_no_header
    

def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor