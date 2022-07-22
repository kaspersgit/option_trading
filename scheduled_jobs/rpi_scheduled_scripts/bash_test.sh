#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/home/pi/Documents/python_scripts/option_trading/
echo "Kasper"
echo "is...... Great"
python3 -c "print('this is within python')"
printenv
python3 -c "import boto3, numpy; print('Succesfully imported two standard packages')"
echo "Starting s3 sync..."
