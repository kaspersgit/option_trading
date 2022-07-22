#!/bin/bash
echo "Starting barchart login scraping..."
source ~/.bashrc
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/scraping/barchart_login_scraping.py >> /home/pi/scripts/barchart_scraping.log
echo "Barchart login scraping... Done"
echo "Starting s3 sync..."
/home/pi/.local/bin/aws s3 sync /home/pi/Documents/python_scripts/option_trading/data/barchart/. s3://project-option-trading/raw_data/barchart/ >> /home/pi/scripts/barchart_scraping.log --profile mrOption
echo "s3 sync... Done"
