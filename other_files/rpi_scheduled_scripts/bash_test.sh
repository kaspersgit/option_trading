#!/bin/sh
echo "Kasper"
echo "is...... Great"
python3 "print('this is within python')"
echo "Starting s3 sync..."
aws s3 sync /home/pi/Documents/python_scripts/option_trading/data/marketbeat/. s3://project-option-trading/raw_data/marketbeat/ >> /home/pi/scripts/marketbeat_scraping.log
echo "s3 sync... Done"