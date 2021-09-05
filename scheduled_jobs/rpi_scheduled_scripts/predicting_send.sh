#!/bin/bash
export PYTHONPATH='/home/pi/Documents/python_scripts/option_trading/'
echo "Predicting with marketbeat data"
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/predicting/predicting_marketbeat.py PROD_MB_10p_GB32_v1x0 PRODUCTION
echo "Predicting with barchart data main model"
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/predicting/predicting_barchart.py PROD_c_GB32_v1x4 PRODUCTION
echo "Predicting with barchart data old model"
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/predicting/predicting_barchart.py PROD_c_AB32_v1x0 PRODUCTION
