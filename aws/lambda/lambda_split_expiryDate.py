#%%
import csv
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError
from botocore.exceptions import ClientError

# filepath = "/Users/kasper.de-harder/gits/option_trading/data/barchart/barchart_unusual_activity_2020-10-07.csv"
# bucket = "project-option-trading"
# key = "raw_data/barchart/barchart_unusual_activity_2021-01-25.csv"
# os.chdir("/Users/kasper.de-harder/Documents/testing")

# function
def upload_to_s3(local_file, bucket, s3_file, acl_permission="authenticated-read"):
    s3 = boto3.client("s3")
    try:
        s3.upload_file(local_file, bucket, s3_file, ExtraArgs={"ACL": acl_permission})
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def download_from_s3(s3_file, bucket, local_file):
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, s3_file, local_file)
        print("Download Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


# main source
# https://stackoverflow.com/questions/46847803/splitting-csv-file-based-on-a-particular-column-using-python
def lambda_handler(event, context):
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]
    local_file = "/tmp/option_data.csv"
    print(
        "Downloading from bucket {} \nwith key {} \nto {} \n".format(
            bucket, key, local_file
        )
    )
    download_from_s3(key, bucket, local_file)

    with open(local_file) as fin:
        csvin = csv.DictReader(fin)
        # Category -> open file lookup
        outputs = {}
        expiryDates = []
        for row in csvin:
            expiryDate = row["expirationDate"]
            exportDate = row["exportedAt"]
            exportDate = datetime.strptime(exportDate, "%Y-%m-%d %H:%M:%S").strftime(
                "%Y-%m-%d"
            )
            # Open a new file and write the header
            if expiryDate not in outputs:
                expiryDates.append(expiryDate)
                filename = "exported_" + exportDate + "_expires_" + expiryDate
                fout = open("{}.csv".format(filename), "w")
                dw = csv.DictWriter(fout, fieldnames=csvin.fieldnames)
                dw.writeheader()
                outputs[expiryDate] = fout, dw
            # Always write the row
            _ = outputs[expiryDate][1].writerow(row)
        # Show how many files were created
        print(
            "Files created {} \nMin expiry date {} \nMax expiry date {}".format(
                len(expiryDates), min(expiryDates), max(expiryDates)
            )
        )
        # Close all the files
        for fout, _ in outputs.values():
            fout.close()
            # Upload all files to S3 bucket
            filename = fout.name
            s3_path = (
                "on_expiry_date/expires" + fout.name.split(".")[0].split("expires")[1]
            )
            bucket = "project-option-trading"
            print(
                "Uploading {} \nto bucket {} \nwith key {}".format(
                    filename, bucket, s3_path
                )
            )
            upload_to_s3(filename, bucket, s3_path)
