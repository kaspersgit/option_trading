#%%
## Checking out https://www.barchart.com/options/unusual-activity/stocks?page=1
# and https://www.marketbeat.com/market-data/unusual-call-options-volume/
# and https://marketchameleon.com/Reports/UnusualOptionVolumeReport
#### Using Selenium
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
from datetime import datetime
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

def get_loaded_page(url, wait = 5):
    browser = webdriver.Firefox(executable_path='C:/Users/kaspe/Downloads/geckodriver-v0.26.0-win64/geckodriver.exe')
    browser.get(url)
    delay = wait # seconds
    try:
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, 'th.baseSymbol.text-left'))
        WebDriverWait(browser, delay).until(element_present)
        print("Page is ready!")
    except TimeoutException:
        print("Loading took too much time!")
    return(browser)

# To get the volume
def get_column_values(columname,soup):
    volume = list()
    if columname == 'baseSymbol':
        for td in soup.select('td.baseSymbol.text-left'):
            volume.append(td.get_text(strip=True))
    input_tag = soup.find_all("td", {"class": columname})
    for i in range(len(input_tag)):
        span_tag = input_tag[i].find("span",{'data-ng-bind':'cell'})
        if not span_tag == None:
            volume.append(span_tag.text)
    return(volume)

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

    # try to get the class names (class of the column it seems)
    classnames = []
    for element in classes:
        classname = columns[0].find_all('th', {'class': element})
        if (not classname == None) & (
                element not in ['text-left', 'hide', 'barchart-sort-desc', 'barchart-sort-asc', 'bc-glyph-sort-desc',
                                'bc-glyph-sort-asc', 'quick-links', 'hide-for-print']):
            print(element)
            classnames.append(element)

    return(classnames, columntitles)

"""
absolete
# try to get the class names (class of the column it seems)
classnames = []
for element in classes:
    classname = columns[0].find_all('th',{'class':element})
    if (not classname == None) & (element not in ['text-left','hide','barchart-sort-desc','barchart-sort-asc','bc-glyph-sort-desc','bc-glyph-sort-asc','quick-links','hide-for-print']):
        print(element)
        classnames.append(element)
"""

today = datetime.today().strftime("%Y-%m-%d")
url = 'https://www.barchart.com/options/unusual-activity/stocks?page='
page = get_loaded_page('https://www.barchart.com/options/unusual-activity/stocks')
html = page.page_source
soup = BeautifulSoup(html, 'html.parser')

nr_pages = soup.select('div.bc-table-pagination')
if nr_pages:
    pages = nr_pages[0].get_text(strip=False)
    pages = pages.replace('\n',' ')
    nr_pages = [int(s) for s in pages.split() if s.isdigit()]
    nr_pages = int(np.ceil(nr_pages[0]/100))
else:
    nr_pages = 1
print("{} page(s) found".format(nr_pages))

# create empty dataframe to save all data for today in
df_total = pd.DataFrame()

for p in range(1, nr_pages+1):
    print(p)
    if p > 1:
        page = get_loaded_page(url+str(p))
        html = page.page_source
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

    df_total = pd.concat([df_total, df])

# Cleaning and adding columns
df_total['current_date'] = today
df_total['current_date'] = pd.to_datetime(df_total['current_date'])
df_total['expirationDate'] = pd.to_datetime(df_total['expirationDate'])
df_total['baseLastPrice'] = df_total["baseLastPrice"].str.replace(",", "").astype(float)
df_total['strikePrice'] = df_total["strikePrice"].str.replace(",", "").astype(float)
df_total['daysToExpiration'] = df_total['daysToExpiration'].astype(int)

# Saving file as CSV
df_total.to_csv('barchart_unusual_activity_'+today+'.csv')





