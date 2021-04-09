#%%
## Checking out https://www.barchart.com/options/unusual-activity/stocks?page=1
# and https://www.barchart.com/options/volume-change/stocks?page=1
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
    # To have some human kind of behaviour with visit in the website
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
    if columname == 'symbol':
        for td in soup.select('td.symbol.text-left'):
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
        if 'symbol' in ab[0]:
            ab = ['symbol']
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
url = 'https://www.barchart.com/options/volume-change/stocks?page='
try:
    html = get_loaded_page('https://www.barchart.com/options/volume-change/stocks')
except Exception:
    print('Failed, trying again')
    html = get_loaded_page('https://www.barchart.com/options/volume-change/stocks')

soup = BeautifulSoup(html, 'html.parser')
# adding timestamp for loggin
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Get the number of pages to loop through
nr_pages = soup.select('div.bc-table-pagination')
if nr_pages:
    pages = nr_pages[0].get_text(strip=False)
    pages = pages.replace('\n',' ')
    nr_records = [int(s) for s in pages.split() if s.isdigit()]
    nr_pages = int(np.ceil(nr_records[0]/100))
else:
    nr_pages = 1
print("{} page(s) found".format(nr_pages))
print("Expecting around {} records".format(nr_records[0]))

# Scrape the table from every page and put together
# create empty dataframe to save all data for today in
df_total = pd.DataFrame()

for p in range(1, nr_pages+1):
    # adding timestamp for logging
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
    # the symbol is treated specially in the get_column_values function
    # and outputs an extra row which has to be deleted
    for class_ in classnames:
        class_list = get_column_values(columname=class_, soup=soup)
        # To remove possible bottom row which just states the column name again
        if any(class_v in names for class_v in class_list):
            for i in class_list:
                if i in names:
                    # print(i)
                    class_list.remove(i)
        df[class_] = class_list

    print('Exctracted {} values for {} different columns'.format(len(df),len(classnames)))
    df_total = pd.concat([df_total, df])

# close headless display
display.stop()

# adding timestamp for logging
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Cleaning and adding columns
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df_total['exportedAt'] = now
df_total['lastPrice'] = pd.to_numeric(df_total["lastPrice"].str.replace(",", "").str.replace('*', ''), errors = 'coerce')
df_total['priceChange'] = pd.to_numeric(df_total["priceChange"].str.replace("+", "").str.replace('*', ''), errors = 'coerce') # has plus or minus sign
df_total['percentChange'] = pd.to_numeric(df_total["percentChange"].str.replace("+", "").str.replace("%", "").str.replace('*', ''), errors = 'coerce') # has plus or minus sign
df_total['optionsTotalVolume'] = pd.to_numeric(df_total["optionsTotalVolume"].str.replace(",", "").str.replace('*', ''), errors = 'coerce')
df_total['optionsTotalOpenInterest'] = pd.to_numeric(df_total["optionsTotalOpenInterest"].str.replace(",", "").str.replace('*', ''), errors = 'coerce')
df_total['optionsImpliedVolatilityRank1y'] = pd.to_numeric(df_total["optionsImpliedVolatilityRank1y"].str.replace(",", "").str.replace("%", "").str.replace('*', ''), errors = 'coerce')
df_total['optionsTotalVolumePercentChange1m'] = pd.to_numeric(df_total["optionsTotalVolumePercentChange1m"].str.replace("+", "").str.replace(",", "").str.replace("%", "").str.replace('*', ''), errors = 'coerce')
df_total['optionsCallVolume'] = pd.to_numeric(df_total["optionsCallVolume"].str.replace(",", "").str.replace('*', ''), errors = 'coerce')
df_total['optionsPutVolume'] = pd.to_numeric(df_total["optionsPutVolume"].str.replace(",", "").str.replace('*', ''), errors = 'coerce')
df_total['optionsPutCallVolumeRatio'] = pd.to_numeric(df_total["optionsPutCallVolumeRatio"].str.replace(",", "").str.replace('*', ''), errors = 'coerce')

print('Extracted a total of {} records'.format(len(df_total)))

# Saving file as CSV
df_total.to_csv('/home/pi/Documents/python_scripts/option_trading/data/barchart_unusual_volume/barchart_unusual_volume_'+today+'.csv', index=False)

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('Script finished at {}'.format(now))
