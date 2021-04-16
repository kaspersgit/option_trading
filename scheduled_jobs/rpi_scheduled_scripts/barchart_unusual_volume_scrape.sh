#!/bin/bash
echo "Starting barchart unusual volume scraping..."
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/scraping/barchart_unusual_volume_scraping.py >> /home/pi/scripts/barchart_unusual_volume_scraping.log
echo "Barchart unusual volume scraping... Done"
echo "Starting s3 sync..."
/home/pi/.local/bin/aws s3 sync /home/pi/Documents/python_scripts/option_trading/data/barchart_unusual_volume/. s3://project-option-trading/raw_data/barchart_unusual_volume/ >> /home/pi/scripts/barchart_unusual_volume_scraping.log
echo "s3 sync... Done"
