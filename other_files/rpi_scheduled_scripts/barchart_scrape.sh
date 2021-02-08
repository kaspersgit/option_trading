#!/bin/sh
python3 /home/pi/Documents/python_scripts/option_trading/scraping/barchart_scraping.py >> /home/pi/scripts/barchart_scraping.log
aws s3 sync /home/pi/Documents/python_scripts/option_trading/data/barchart/. s3://project-option-trading/raw_data/barchart/
