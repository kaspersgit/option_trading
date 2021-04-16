import boto3
import catboost as cb
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


def lambda_handler(event, context):
    message = 'Boto3 version: {}, PD version: {}, NP version: {}, catboost version: {}'.format(
        boto3.__version__,
        pd.__version__,
        np.__version__,
        cb.__version__
    )

    return {
        'message' : message
    }
