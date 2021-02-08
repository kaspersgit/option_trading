#!/bin/sh
python3 /home/pi/Documents/python_scripts/option_trading/scraping/marketbeat_scraping.py >> /home/pi/scripts/marketbeat_scraping.log
aws s3 sync /home/pi/Documents/python_scripts/option_trading/data/marketbeat/. s3://project-option-trading/raw_data/marketbeat/
