# Create lambda function to split data on expiry date
aws lambda create-function  \
--function-name "project-option-splitExpiryDate"  \
--runtime "python3.7"    \
--role "arn:aws:iam::343302203904:role/LambdaS3dataTransfer"  \
--handler "lambda_split_expiryDate.lambda_handler"  \
--timeout 5 \
--memory-size 256  \
--zip-file "fileb://lambda_split_expiryDate.zip"

# give lambda invoke permissions to s3 bucket
aws lambda add-permission --function-name project-option-splitExpiryDate --principal s3.amazonaws.com \
--statement-id S3StatementId --action "lambda:InvokeFunction" \
--source-arn arn:aws:s3:::project-option-trading \
--source-account 343302203904

# Add trigger to lambda function
# Giving an Error
aws s3api put-bucket-notification-configuration \
    --bucket "project-option-trading" \
    --notification-configuration "file://lambdaS3trigger.json"

# Delete all bucket notification configurationss
aws s3api put-bucket-notification-configuration --bucket project-option-trading --notification-configuration="{}"
