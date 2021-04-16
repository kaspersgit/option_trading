# Making CatBoost work in AWS lambda
Altough this is specific to catboost, the same approach should work to make
any package work which requires only numpy/pandas/scipy and is not more than 150 MB
on itself.

# This is a method which is a combination of different examples (check examples/ folder)
# Main idea
1. Create an AWS lambda like docker container
2. Install all the necessities
3. Shrink to 254 MB (262MB is the hard max limit)
4. create zip file
5. Upload to S3
6. Add to Lambda function

# Get to it
### Create a docker container which will run the build.sh file (takes a while)
docker run --rm -it -v $(pwd):/outputs lambci/lambda:build-python3.6 /bin/bash /outputs/build.sh

### Add your lambda function to zip package
zip -9 lambda-package.zip lambda_function.py

### Upload zip package to designated S3 location (Make sure you have a valid AWS token)
aws s3 cp lambda-package.zip s3://collection-pd-distress-handling/quality_phone_number/lambda-deployment/lambda-package.zip

### (if necessary) create lambda
sh lambda_create.sh

### Update Lambda function with package
aws lambda update-function-code --region eu-west-1 --function-name collection-pd-phonenumberPredict --s3-bucket collection-pd-distress-handling --s3-key quality_phone_number/lambda-deployment/lambda-package.zip

### Set lambda to be triggered from S3 put object
aws s3api put-bucket-notification-configuration --region eu-west-1 \
--bucket collection-pd-distress-handling \
--notification-configuration file://lambdaS3Trigger.json

### Test run lambda in console (make sure the memory is set to 1024MB)


## Credits to
https://marc-metz.medium.com/docker-run-rm-it-v-pwd-var-task-lambci-lambda-build-python3-7-bash-c7d53f3b7eb2
https://medium.com/@AlexeyButyrev/xgboost-on-aws-lambda-345f1394c2b
https://towardsdatascience.com/how-to-shrink-numpy-scipy-pandas-and-matplotlib-for-your-data-product-4ec8d7e86ee4
https://towardsdatascience.com/how-to-get-fbprophet-work-on-aws-lambda-c3a33a081aaf
