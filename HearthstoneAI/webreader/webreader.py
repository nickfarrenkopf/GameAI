import os
import urllib
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


### SELENIUM  ###

def init_driver():
    """ initialize webdriver """
    driver = webdriver.Chrome('Chromedriver')
    driver.wait = WebDriverWait(driver, 5)
    return driver


def download_hearthstone_images():
    """ """
    datas = []
    for i in range(21, 22): # 18
        print(i)
        url = 'https://www.hearthpwn.com/cards?display=3&filter-include-card-text=y&filter-premium=1' \
              '{}'.format('' if i == 1 else '&page={}'.format(i))
        driver.get(url)
        table_xpath = '//*[@id="content"]/section/div/div/div[2]'
        table = driver.find_elements_by_xpath(table_xpath)[0]
        anchors = table.find_elements_by_tag_name('a')
        for a in anchors:
            href = a.get_attribute('href')  
            name = '_'.join(href.split('-')[1:])
            # image
            img = a.find_elements_by_tag_name('img')[0].get_attribute('src')
            img_path = os.path.join(card_path, '{}.png'.format(name))
            if 'mecha-jaraxxus' not in img_path and 'sir-annoy-o' not in img_path :
                urllib.request.urlretrieve(img, img_path)
    return datas
            


def download_hearthstone_data():
    """ """
    data = []
    for i in range(1, 16):
        url = 'https://www.hearthpwn.com/cards?display=1&filter-include-card-text=y&filter-premium=1' \
              '{}'.format('' if i == 1 else '&page={}'.format(i))
        table_xpath = '//*[@id="cards"]/tbody'
        driver.get(url)
        table = driver.find_elements_by_xpath(table_xpath)[0]
        rows = table.find_elements_by_tag_name('tr')
        for j, r in enumerate(rows):
            name_xpath = '//*[@id="cards"]/tbody/tr[{}]/td[1]/a'.format(j + 1)
            type_xpath = '//*[@id="cards"]/tbody/tr[{}]/td[2]'.format(j + 1)
            class_xpath = '//*[@id="cards"]/tbody/tr[{}]/td[3]'.format(j + 1)
            card_name = r.find_elements_by_xpath(name_xpath)[0].get_attribute("text")
            card_name = '_'.join(card_name.lower().split(' '))
            card_type = r.find_elements_by_xpath(type_xpath)[0].text.lower()
            card_type = '_'.join(card_type.lower().split(' '))
            card_class = r.find_elements_by_xpath(class_xpath)[0].text.lower()
            card_class = card_class if card_class != '' else 'neutral'
            card_class = card_class.replace(' ', '')
            data.append([card_name, card_class, card_type])
    return data




### PARAMS ###

# location
base_path = os.path.dirname(os.path.realpath(__file__))
#driver_path = os.path.join(base_path)
driver_path = os.path.join(base_path, 'chromedriver')
card_path = os.path.join(base_path, 'cards')


### PROGRAM ###






driver = init_driver()
data = download_hearthstone_images()
#driver.quit()


##cards = os.listdir(cards_path)
##
##classes = list(set([d[1] for d in data]))
##types = list(set([d[2] for d in data]))
##
##my_data = dict((d[0], [d[1], d[2]]) for d in data)
##a = [os.makedirs(os.path.join(cards_path, c)) for c in classes]
##
##for c in cards:
##    if c not in classes and c in my_data.keys():
##        old_path = os.path.join(cards_path, c)
##        new_path = os.path.join(cards_path, my_data[c][0], c)
##        os.rename(old_path, new_path)


