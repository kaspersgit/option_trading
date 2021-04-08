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

# hyperparamter search
- use optuna package for bayesian search with TPE surrogate

# notes
The following Errors have been encountered on Raspberry PI
when import numpy or matplotlib
ImportError: /lib/arm-linux-gnueabihf/libc.so.6: version `GLIBC_2.28' not found

Solved:
pip install numpy --global-option="-mfloat-abi=hard" --force-reinstall
now solved by updating raspberry version from 9 to 10 (Buster)

Installing chromium webdriver and browser to make scraping work
sudo apt-get install chromium-browser --yes
apt-get install chromium-chromedriver --yes
apt-get install xvfb --yes
pip install PyVirtualDisplay xvfbwrapper selenium

Swap file raspberry pi, increased size according to [these steps](https://nebl.io/neblio-university/enabling-increasing-raspberry-pi-swap/)

Due to memory leakage of lxpanel it might be wise to restart it periodically
`lxpanelctl restart`

Some basic commands to list raspberry pi usage:
</br>
disk space:     `df -h` 
</br>
memory usage:   `free -h`
</br>
processes:      `ps aux`