#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/home/pi/Documents/python_scripts/option_trading/
date +"%Y-%m-%d %T"
echo "Predicting with barchart data main model"
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/predicting/predicting_barchart.py DEV_c_GB64_v1x4 PRODUCTION
echo "Predicting with marketbeat data"
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/predicting/predicting_marketbeat.py PROD_MB_10p_GB32_v1x0 PRODUCTION
