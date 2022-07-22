#!/bin/bash
echo "Starting marketbeat scraping..."
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/scraping/marketbeat_scraping.py >> /home/pi/scripts/marketbeat_scraping.log
echo "Marketbeat scraping... Done"
echo "Starting s3 sync..."
/home/pi/.local/bin/aws s3 sync /home/pi/Documents/python_scripts/option_trading/data/marketbeat/. s3://project-option-trading/raw_data/marketbeat/ >> /home/pi/scripts/marketbeat_scraping.log --profile mrOption
echo "s3 sync... Done"