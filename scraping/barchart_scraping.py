#%%
## Checking out https://www.barchart.com/options/unusual-activity/stocks?page=1
# and https://www.marketbeat.com/market-data/unusual-call-options-volume/
# and https://marketchameleon.com/Reports/UnusualOptionVolumeReport
#### Using Selenium
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import numpy as np
from datetime import datetime
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from pyvirtualdisplay import Display
import time
import random
import platform

# Only when running in production
if platform.system() == 'Linux':
    display = Display(visible=0, size=(800,600))
    display.start()

    # To have some human kind of behaviour with visitin the website
    rand_wait=random.uniform(0,200)
    time.sleep(rand_wait)

def get_loaded_page(url, wait = 20):
    if platform.system() == 'Linux':
        browser = webdriver.Chrome()
    elif platform.system() == 'Windows':
        browser = webdriver.Firefox(
            executable_path='C:/Users/kaspe/Downloads/geckodriver-v0.26.0-win64/geckodriver.exe')

    browser.get(url)
    delay = wait # seconds
    try:
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, 'th span'))
        WebDriverWait(browser, delay).until(element_present)
        html = browser.page_source
    except TimeoutException:
        print("Loading took too much time!")
        raise Exception
    browser.quit()
    return(html)

# To get the volume
def get_column_values(columname,soup):
    values = list()
    if columname == 'baseSymbol':
        for td in soup.select('td.baseSymbol.text-left'):
            values.append(td.get_text(strip=True))
    input_tag = soup.find_all("td", {"class": columname})
    for i in range(len(input_tag)):
        span_tag = input_tag[i].find("span",{'data-ng-bind':'cell'})
        if not span_tag == None:
            values.append(span_tag.text)
    #print("{} column, {} values extracted".format(columname,len(values)))
    return(values)

def get_column_classes(soup, part = 'thead'):
    # Getting all column titles
    # to get all classes (columns are the interest)
    classes = []
    columntitles = []
    columns = soup.find_all(part)
    for element in columns[0].find_all(class_=True):
        ab = element['class']
        if 'baseSymbol' in ab[0]:
            ab = ['baseSymbol']
        classes.extend(ab)
        title = element.find_all("span", {"data-bs-tooltip": ""})
        if len(title) > 0:
            string_text = title[0].get_text(strip=True)
            columntitles.append(string_text)
    #print('Abundance of classes found, {} in total'.format(len(classes)))
    # try to get the class names (class of the column it seems)
    classnames = []
    for element in classes:
        classname = columns[0].find_all('th', {'class': element})
        if (not classname == None) & (
                element not in ['text-left', 'hide', 'barchart-sort-desc', 'barchart-sort-asc', 'bc-glyph-sort-desc',
                                'bc-glyph-sort-asc', 'quick-links', 'hide-for-print']):
            classnames.append(element)
    #print('Classes cleaned, {} columnclasses left'.format(len(classnames)))
    return(classnames, columntitles)

### Let the scraping start
# Set some variables
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
today = datetime.today().strftime("%Y-%m-%d")
print('Script started at {}'.format(now))
url = 'https://www.barchart.com/options/unusual-activity/stocks?page='
try:
    html = get_loaded_page('https://www.barchart.com/options/unusual-activity/stocks')
except Exception:
    print('Failed, trying again')
    html = get_loaded_page('https://www.barchart.com/options/unusual-activity/stocks')
soup = BeautifulSoup(html, 'html.parser')

# Get the number of pages to loop through
nr_pages = soup.select('div.bc-table-pagination')
if nr_pages:
    pages = nr_pages[0].get_text(strip=False)
    pages = pages.replace('\n',' ')
    nr_pages = [int(s) for s in pages.split() if s.isdigit()]
    nr_pages = int(np.ceil(nr_pages[0]/100))
else:
    nr_pages = 1
print("{} page(s) found".format(nr_pages))

# Scrape the table from every page and put together
# create empty dataframe to save all data for today in
df_total = pd.DataFrame()

for p in range(1, nr_pages+1):
    print('Working on page {} of {}'.format(p,nr_pages))
    if p > 1:
        try:
            html = get_loaded_page(url+str(p))
        except Exception as e:
            print('Skipping page due to:')
            print(e)
            continue
        soup = BeautifulSoup(html, 'html.parser')
    classnames, names = get_column_classes(soup)
    df = pd.DataFrame()
    # the baseSymbol is treated specially in the get_column_values function
    # and outputs an extra row which has to be deleted
    for class_ in classnames:
        class_list = get_column_values(columname=class_, soup=soup)
        if any(class_ in 'Symbol' for class_ in class_list):
            class_list.remove('Symbol')
        df[class_] = class_list
    print('Exctracted {} values for {} different columns'.format(len(df),len(classnames)))
    df_total = pd.concat([df_total, df])

# close headless display
display.stop()

# Cleaning and adding columns
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df_total['exportedAt'] = now
df_total['expirationDate'] = pd.to_datetime(df_total['expirationDate'])
df_total['tradeTime'] = pd.to_datetime(df_total['tradeTime'])
df_total['baseLastPrice'] = df_total["baseLastPrice"].str.replace(",", "").str.replace('*', '').astype(float)
df_total['strikePrice'] = df_total["strikePrice"].str.replace(",", "").str.replace('*', '').astype(float)
df_total['volume'] = df_total["volume"].str.replace(",", "").str.replace('*', '').astype(float)
df_total['openInterest'] = df_total["openInterest"].str.replace(",", "").str.replace('*', '').astype(float)
df_total['volatility'] = df_total["volatility"].str.replace(",", "").str.replace("%", "").str.replace('*', '').astype(float)
df_total['daysToExpiration'] = df_total['daysToExpiration'].astype(int)

# Saving file as CSV
df_total.to_csv('/home/pi/Documents/python_scripts/option_trading/data/barchart/barchart_unusual_activity_'+today+'.csv', index=False)

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('Script finished at {}'.format(now))
