# selenium test 
from pyvirtualdisplay import Display
from selenium import webdriver
display = Display(visible=0, size=(800,600))
display.start()
print('Start')
browser = webdriver.Chrome()
print('Webdriver initiated succesfully')
