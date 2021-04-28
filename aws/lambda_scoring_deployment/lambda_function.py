# Load packages
import boto3
import pandas as pd
import numpy as np
import catboost as cb
from io import StringIO, BytesIO
from datetime import datetime
import os

# Create functions
def boolean2int(input_df):
    print('Casting booleans to integers')
    # Convert columns into wanted types (TODO automate)
    cols = input_df.columns[input_df.dtypes == "object"]
    for col in cols:
        if ("t" in input_df[col].unique()) & ("f" in input_df[col].unique()):
            input_df[col] = input_df[col].apply(lambda x: np.where(x == "t", 1, 0))
    return input_df

def fillNAvalues(input_df):
    print('Filling NaN values')
    # Fill NA
    # nr events set to 0 when NA
    month_events_cols = input_df.columns[input_df.columns.str.endswith("m")]
    input_df[month_events_cols] = input_df[month_events_cols].fillna(0)

    # last event days set to 9999 when NA (for tree based algorithm)
    nanTo9999 = input_df.columns[input_df.columns.str.endswith("days")]
    input_df[nanTo9999] = input_df[nanTo9999].fillna(9999)

    # Set individual features to a certain value
    # Persona phone number recency put to 9999
    input_df["number_recency_ranking"].fillna(9999, inplace=True)
    return input_df

def load_from_s3(
    s3_con,
    bucket,
    key_prefix,
    gzipped,
):
    # Reading in multiple files with same prefix
    try:
        df = pd.DataFrame()
        obj = s3_con.get_object(Bucket=bucket, Key=key_prefix)
        body = obj["Body"].read()
        if gzipped:
            gzipfile = BytesIO(body)
            gzipfile = gzip.GzipFile(fileobj=gzipfile)
            content = gzipfile.read()
        else:
            content = body
        s = str(content, "utf-8")
        data = StringIO(s)
        df_temp = pd.read_csv(data)
        df = pd.concat([df, df_temp], ignore_index=True)
        return df
    except ClientError as e:
            print("Unexpected error: %s" % e.response["Error"])

# Upload file to S3 bucket
def write_dataframe_to_csv_on_s3(s3_con, dataframe, filename, bucket):
    """
    Write a dataframe to a CSV on S3

    s3_con: boto3.resource("s3") connection type
    dataframe: pandas
    filename: path/key to file
    bucket: S3 bucket
    """
    # Create buffer
    csv_buffer = StringIO()
    print("buffer created")
    # Write dataframe to buffer
    dataframe.to_csv(csv_buffer, sep=",", index=False)
    print("dataframe written to buffer")
    # Write buffer to S3 object and give bucket owner full access
    s3_con.Object(bucket, f"{filename}").put(
        Body=csv_buffer.getvalue(), ACL="bucket-owner-full-control"
    )
    print("written to S3 object")

# Set up persistent variables
# Set working directory
os.chdir('/tmp')

# Set up connection to S3
s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")

# Load in the model
# Download CatBoost model from S3
s3_client.download_file('project-option-trading', 'trained_models/cb.cbm', 'cb.cbm')
pn_model = cb.CatBoostClassifier()
file_path = "cb.cbm"
model_version = 'cb_v1'
pn_model.load_model(file_path, format="cbm")

def lambda_handler(event, context):
    today = datetime.today().strftime("%Y-%m-%d")
    print("In handler")
    print("Todays date: {}".format(today))
    print(event)
    source_bucket = event['Records'][0]['s3']['bucket']['name']
    target_bucket = source_bucket
    print(source_bucket)
    key = event['Records'][0]['s3']['object']['key']

    print('Key: {}'.format(key))

    message = 'Boto3 version: {} \nPD version: {} \nNP version: {} \ncatboost version: {}'.format(
        boto3.__version__,
        pd.__version__,
        np.__version__,
        cb.__version__
    )
    print(message)

    print('Loading in data from s3 bucket {} and key {} ...'.format(source_bucket, key))
    # Load data
    df = load_from_s3(
        s3_client,
        bucket=source_bucket,
        key_prefix=key,
        gzipped=False,
    )

    print('Loading in data from s3 bucket {} and key {} ... Done'.format(source_bucket, key))

    print('DataFrame shape: {}'.format(df.shape))

    # Pre processing data
    print('Pre processing data ...')
    df = boolean2int(df)
    df = fillNAvalues(df)
    print('Pre processing data ... Done')

    # Score
    print('Scoring phone numbers ...')
    X = df[pn_model.feature_names_]
    # Thread count = 1 to make it work in lambda function
    y_pred = pn_model.predict_proba(X, thread_count=1)[:, 1]
    print('Scoring phone numbers ... Done')

    # Make scored dataframe
    df_scored = df[["id"]].copy()
    df_scored["score"] = y_pred
    df_scored["model"] = model_version
    df_scored["pred_date"] = today

    print('Scored dataFrame shape: {}'.format(df_scored.shape))

    write_dataframe_to_csv_on_s3(
        s3_con=s3_resource,
        dataframe=df_scored,
        filename=output_key,
        bucket=target_bucket,
    )
    print('Successfully scored {} phone numbers!'.format(df_scored.shape[0]))
