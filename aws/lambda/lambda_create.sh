# Create lambda function to split data on expiry date
aws lambda create-function  \
--function-name "project-option-splitExpiryDate"  \
--runtime "python3.7"    \
--role "arn:aws:iam::343302203904:role/LambdaS3dataTransfer"  \
--handler "lambda_split_expiryDate.lambda_handler"  \
--timeout 5 \
--memory-size 256  \
--zip-file "fileb://lambda_split_expiryDate.zip"

# Add trigger to lambda function
# Giving an Error
aws s3api put-bucket-notification-configuration \
    --bucket "project-option-trading" \
    --notification-configuration "file://lambdaS3trigger.json"

# file notification.json and content:
