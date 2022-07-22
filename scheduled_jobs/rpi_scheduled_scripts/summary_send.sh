#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/home/pi/Documents/python_scripts/option_trading/
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/weekly_summary/enrich_expired_options.py
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/weekly_summary/send_summary.py PROD_EBM64_v1x1 PRODUCTION
