#!/bin/bash
date +"%Y-%m-%d %T"
echo "Starting barchart login scraping..."
. /home/pi/.bashrc
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/scraping/barchart_login_scraping.py
echo "Barchart login scraping... Done"
echo "Starting s3 sync..."
/home/pi/.local/bin/aws s3 sync /home/pi/Documents/python_scripts/option_trading/data/barchart/. s3://project-option-trading/raw_data/barchart/ --profile mrOption
echo "s3 sync... Done"
