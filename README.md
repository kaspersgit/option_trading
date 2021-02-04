# option_trading
Option trading based on a strategy

# Main model
AdaBoost model from sklearn package due to 32bit system on raspberry pi

# upcoming improvements
pre processing of data should not output duplicates or non mature options
models should be well documented
- name
- date
- details of data trained on
- features used
- performance

# notes
The following Errors have been encountered on Raspberry PI
when import numpy or matplotlib
ImportError: /lib/arm-linux-gnueabihf/libc.so.6: version `GLIBC_2.28' not found

Solved:
pip install numpy --global-option="-mfloat-abi=hard" --force-reinstall
