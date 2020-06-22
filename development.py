#%%
## Checking out https://www.barchart.com/options/unusual-activity/stocks?page=1
# and https://www.marketbeat.com/market-data/unusual-call-options-volume/
# to get unusual option activity 
# load packages 
import pandas as pd
from bs4 import BeautifulSoup
import requests
 

#%%
# tryout
session = requests.Session()
headers={'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36 RuxitSynthetic/1.0 v5389158804 t38550'}
url = "https://www.barchart.com/options/unusual-activity"
page = session.get(url, headers=headers)

#%%
soup = BeautifulSoup(page.text, 'html.parser')
print(soup.prettify())

#%%
#### Using Selenium
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome(executable_path=r'/Users/kasper.de-harder/Documents/programs/chromedriver')

driver.get("https://www.barchart.com/options/unusual-activity")

#%%

heights = []
counter = 0
for i in range(1,300):
    bg = driver.find_element_by_css_selector('body')
    time.sleep(0.1)
    bg.send_keys(Keys.END)
    heights.append(driver.execute_script("return document.body.scrollHeight"))
    try :
        bottom = heights[i-16]
    except:
        pass
    if i%16 ==0:
        new_bottom = heights[i-1]
        if bottom == new_bottom:
            break
        
#%%
# Add headers
# Create session to work with 
# Adding information about user agent
def get_unusual_activi(session=None):
    if session is None:
        session = requests.Session()

    url = "https://www.barchart.com/options/unusual-activity/stocks?page=1"
    page = session.get(url)

    try:
        session = HTMLSession()
        response = session.get(url)
        
    except requests.exceptions.RequestException as e:
        print(e)

# chance otherwise is that website block when they see python trying to access
opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]