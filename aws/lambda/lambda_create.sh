# Create lambda function to split data on expiry date
aws lambda create-function  \
--function-name "project-option-splitExpiryDate"  \
--runtime "pthon3.7"    \
--role "arn:aws:eu-west-1:343302203904:role/LambdaOptions"  \
--handler "lambda_function.lambda_handler"  \
--timeout 5 \
--memory-size 512  \
--code "file://lambda_split_expiryDate.py"

# Add trigger to lambda function
aws s3api put-bucket-notification-configuration --bucket project-option-trading --notification-configuration file://notification.json

s3:ObjectCreated:*

 {
    "LambdaFunctionConfigurations": [
        {
            "Id": "kickSplitOnExpiryOff",
            "LambdaFunctionArn": "arn:aws:sns:eu-west-1:343302203904:s3-notification-topic",
            "Events": [
                "s3:ObjectCreated:*"
            ]
            "Filter": {
                "Key": {
                    "FilterRules": [
                        {
                        "Name": "prefix",
                        "Value": "rawdata/barchart/"
                        }
          ]
        }
      }
        }
    ]
}