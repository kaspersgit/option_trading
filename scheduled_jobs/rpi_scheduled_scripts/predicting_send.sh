#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/home/pi/Documents/python_scripts/option_trading/
echo "Predicting with barchart data main model"
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/predicting/predicting_barchart.py PROD_c_EBM64_v1x1 PRODUCTION
echo "Predicting with marketbeat data"
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/predicting/predicting_marketbeat.py PROD_MB_10p_GB32_v1x0 PRODUCTION
