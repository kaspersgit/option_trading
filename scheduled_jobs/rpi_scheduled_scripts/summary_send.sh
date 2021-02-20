#!/bin/bash
export PYTHONPATH='/home/pi/Documents/python_scripts/option_trading/'
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/enrich_expired_options.py
python3 /home/pi/Documents/python_scripts/option_trading/scheduled_jobs/send_summary.py PROD_c_AB32_v1x3 PRODUCTION
