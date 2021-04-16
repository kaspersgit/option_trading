aws lambda create-function  \
--function-name "mrOption-barchart-option-predict"  \
--runtime "python3.6"    \
--role "arn:aws:eu-west-1:343302203904:role/mrOptionLambda"  \
--handler "lambda_function.lambda_handler"  \
--timeout 5 \
--memory-size 1024  \
--code "lambda_function.py"  
