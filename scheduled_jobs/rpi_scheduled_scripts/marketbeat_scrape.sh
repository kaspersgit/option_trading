#!/bin/bash
echo "Starting marketbeat scraping..."
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/scraping/marketbeat_scraping.py
echo "Marketbeat scraping... Done"
echo "Starting s3 sync..."
/home/pi/.local/bin/aws s3 sync /home/pi/Documents/python_scripts/option_trading/data/marketbeat/. s3://project-option-trading/raw_data/marketbeat/ --profile mrOption
echo "s3 sync... Done"