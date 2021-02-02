# Below code runs and seems to work, just remember to login with the aws-login-tool using
# aws-login-tool login -r iam-sync/lakehouse-redshift/lakehouse-redshift.IdP_risk_general_eu  -a klarna-data-production
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import gzip


def connect_to_s3(profile, type="client"):
    session = boto3.Session(region_name="eu-west-1", profile_name=profile)
    creds = session.get_credentials()
    if type == "client":
        s3 = boto3.client(
            "s3",
            aws_access_key_id=creds.access_key,
            aws_secret_access_key=creds.secret_key,
            aws_session_token=creds.token,
        )
    elif type == "resource":
        s3 = boto3.resource(
            "s3",
            aws_access_key_id=creds.access_key,
            aws_secret_access_key=creds.secret_key,
            aws_session_token=creds.token,
        )
    return s3


def load_from_s3(
    profile="default",
    bucket=None,
    key_prefix=None,
    gzipped=False,
):
    # Example usage
    # profile = "iam-sync/lakehouse-redshift/lakehouse-redshift.IdP_risk_general_eu@klarna-data-production"
    # bucket = 'eu-production-klarna-data-redshift-unload-eu'
    # key_prefix = 'access-purpose/risk_general_eu/cpd/dc_forecasting/data/daily000'
    # gzipped = False
    ##############
    # For the risk general S3 bucket
    s3_resource = connect_to_s3(profile, type="resource")
    # Reading in multiple files with same prefix
    try:
        df = pd.DataFrame()
        # List all objects in bucket
        bucket = s3_resource.Bucket(bucket)
        prefix_objs = bucket.objects.filter(Prefix=key_prefix)
        for obj in prefix_objs:
            key = obj.key
            body = obj.get()["Body"].read()
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
        if e.response["Error"]["Code"] == "ExpiredToken":
            print(
                "Token expired, login using the AWS login tool:\naws-login-tool login -r "
                + profile.split("@")[0]
                + " -a klarna-data-production"
            )
        if e.response["Error"]["Code"] == "InvalidToken":
            print(
                "Token invalid, login using the AWS login tool:\naws-login-tool login -r "
                + profile.split("@")[0]
                + " -a klarna-data-production"
            )
        else:
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