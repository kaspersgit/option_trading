#%%
## Checking out https://www.barchart.com/options/unusual-activity/stocks?page=1
# and https://www.marketbeat.com/market-data/unusual-call-options-volume/
# and https://marketchameleon.com/Reports/UnusualOptionVolumeReport
#### Using Selenium
#from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import numpy as np
from datetime import datetime
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from pyvirtualdisplay import Display
import time
import random
import platform, os

# Only when running in production
if platform.system() == 'Linux':
    display = Display(visible=0, size=(800,600))
    display.start()

    # To have some human kind of behaviour with visitin the website
    rand_wait=random.uniform(0,200)
    time.sleep(rand_wait)

# Load page
if platform.system() == 'Linux':
    options=webdriver.ChromeOptions()
    # Set download location
    download_loc = "/home/pi/Documents/python_scripts/option_trading/data/barchart"
    prefs = {"download.default_directory": download_loc}
    options.add_experimental_option("prefs", prefs)
    browser = webdriver.Chrome(options=options)

elif platform.system() == 'Windows':
    options=webdriver.ChromeOptions()
    # Set download location
    download_loc = "C:/Users/kaspe/Downloads/selenium_download"
    prefs = {"download.default_directory": download_loc}
    options.add_experimental_option("prefs", prefs)
    browser = webdriver.Chrome(
        executable_path=r'C:/Users/kaspe/Downloads/chromedriver_win32/chromedriver.exe', options=options)

### Let the scraping start
# Set some variables
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
today = datetime.today().strftime("%Y-%m-%d")
print('Script started at {}'.format(now))
url = 'https://www.barchart.com/login'
stocks_url = 'https://www.barchart.com/options/unusual-activity/stocks'

# try:
#     browser.get(url)
#     html = browser.page_source
# except Exception:
#     print('Failed, trying again')
#     browser.get(url)
#     html = browser.page_source

browser.get(url)

# Wait to let page load
# WebDriverWait(browser, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.bc-button.login-button')))
time.sleep(5)

# Deal with privacy / cookies acceptance
print('Attempting to deal with privacy questions')
try:
    # Click see more t&a
    more_terms = WebDriverWait(browser, 50).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Read more to accept preferences')]")))
    more_terms.click()
except NoSuchElementException:
    print('No read more button found')
except:
    print('Screen big enough to show all privacy text')
finally:
    try:
        # Click to reject all
        WebDriverWait(browser, 50).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Reject all')]"))).click()
        print('Rejected all privacy trackers')
    except NoSuchElementException:
        print('No reject all button found')
    except Exception as e:
        print(f'other error {e}')

# Click login button
# login = browser.find_element(By.CSS_SELECTOR, 'a.bc-user-block__button').click()

# Wait for iframe to load
wait = WebDriverWait(browser,20)

# fill in username and password
fill_username = wait.until(EC.element_to_be_clickable((By.NAME, "email")))
fill_password = wait.until(EC.element_to_be_clickable((By.NAME, "password")))

barchart_usr = os.getenv('barchart_usr')
barchart_pw = os.getenv('barchart_pw')

fill_username.send_keys(barchart_usr)
fill_password.send_keys(barchart_pw)

print('Filled username and password')

# Click the actual login button
login_button = browser.find_element(By.CSS_SELECTOR, 'button.bc-button.login-button')
login_button.click()

# Check if logged in
loggedin = browser.find_element(By.CSS_SELECTOR, 'span.bc-glyph-user')
if loggedin:
    print('Logged in successfully')
else:
    print('Could not locate My Accounts tab, indicating login was not successful')

# Double check url
print(browser.current_url)

# Switch to unusual stock option activity
print('Switch to option page')
browser.get(stocks_url)

# Wait 5 seconds
time.sleep(5)

# Double check url
print(browser.current_url)

# Explicit extra waiting
time.sleep(5)

# Click the download data button
print('Looking for the download button')
download_data = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "i.bc-glyph-download")))
download_data.click()

try:
    # if premier subscription alert shows click 'download anyway'
    download_anyway = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Download Anyway')]")))
    download_anyway.click()
    print('Data downloaded')
except:
    print('No premier subscription nudge is given')

# close headless display
display.stop()

# update file name to fit original standard
# Absolute path of a file
today_date_old = datetime.now().strftime('%m-%d-%Y')
today_date_new = datetime.now().strftime('%Y-%m-%d')
old_name = f"{download_loc}/unusual-stock-options-activity-{today_date_old}.csv"
new_name = f"{download_loc}/barchart_unusual_activity_{today_date_new}.csv"

# Renaming the file
os.rename(old_name, new_name)

print('File downloaded and renamed')
print('Loading csv to python for final adjustments')

# Check format of file content
df_total = pd.read_csv(new_name)

# adding timestamp for logging
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# renaming columns
df_total.columns = ['baseSymbol','baseLastPrice','symbolType','strikePrice','expirationDate','daysToExpiration','bidPrice','midpoint','askPrice','lastPrice','volume','openInterest','volumeOpenInterestRatio','volatility','delta','tradeTime']

# selecting only rows without Nan
df_total = df_total[df_total['expirationDate'].notna()]

# Cleaning and adding columns
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df_total['exportedAt'] = now
df_total['expirationDate'] = pd.to_datetime(df_total['expirationDate'])
df_total['tradeTime'] = pd.to_datetime(df_total['tradeTime'])

# Below are already correct type
# for col in ['baseLastPrice','strikePrice','volume','openInterest','bidPrice', 'midpoint','askPrice', 'lastPrice']:
#     df_total[col] = df_total[col].str.replace(",", "").str.replace('*', '').astype(float)

df_total['volatility'] = df_total["volatility"].str.replace(",", "", regex=True).str.replace("%", "", regex=True).str.replace('*', '', regex=True).astype(float)
df_total['daysToExpiration'] = df_total['daysToExpiration'].astype(int)

print('Extracted a total of {} records'.format(len(df_total)))

# Saving file again as CSV
df_total.to_csv(new_name, index=False)

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('Script finished at {}'.format(now))
