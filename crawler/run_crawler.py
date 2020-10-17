import json
from threading import Thread
import queue

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import pandas as pd

class ThreadWithReturnValue(object):
    def __init__(self, target=None, args=(), **kwargs):
        self._que = queue.Queue()
        self._t = Thread(target=lambda q,arg1,kwargs1: q.put(target(*arg1, **kwargs1)) ,
                args=(self._que, args, kwargs), )
        self._t.start()

    def join(self):
        self._t.join()
        return self._que.get()

def crawlCases(driver, wait):
    url_list = driver.find_element_by_xpath('//*[@id="search_result"]/div/article/dl').find_elements_by_tag_name('a')

    contents = []
    for url in url_list:
        url.click()

        window_before = driver.window_handles[0]
        window_after = driver.window_handles[1]


        driver.switch_to.window(window_after)

        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'iframe')))
        iframes = driver.find_elements_by_tag_name('iframe')
        if len(iframes) != 1:
            print("iframe error")
            exit()
        driver.switch_to.frame(iframes[0])

        title = driver.find_element_by_xpath("/html/body")
        contents.append(title.text)

        driver.switch_to.window(window_before)

    return contents


def crawling(keyword) :
    print("Target Keyword : {}".format(keyword))

    all_data = []
    driver = webdriver.Chrome('./chromedriver.exe') # fixed to Window OS
    wait = WebDriverWait(driver, 10)
    driver.get(
        "https://legalsearch.kr/list/prec?cols=ALL_CONTENTS&keyword=" + keyword + "&court_code=400202&sort=score&pageSize=20&filter_search=true")

    url = driver.find_element_by_xpath('//*[@id="search_result"]/div/article/dl').find_elements_by_tag_name('a')

    cases = []
    prev = ""
    while True:
        if driver.current_url == prev:
            break
        prev = driver.current_url

        content = crawlCases(driver, wait)
        cases.extend(content)
        driver.find_element_by_xpath('/html/body/div[2]/section/div/article/div[2]').find_elements_by_tag_name('a')[-1].click()
        wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/section/div/article/div[2]')))

    driver.quit()
    all_data.extend(cases)
    return all_data


if __name__ == '__main__':
    keywords = ["강제추행","강간","유사강간","준강간","준강제추행","간음"]
    # keywords = ["유사강간","준강간","준강제추행"]

    # For Debugging
    # crawled_data = crawling(keywords[3])

    # Run Thread For each keyword
    twrv = [ThreadWithReturnValue(target=crawling, args=(keyword,)) for keyword in keywords]

    crawled_data = []
    for t in twrv:
        crawled_data.append(t.join())

    # For Dumping - Pandas DataFrame And Excel File
    df_construction = [['id', 'content']]
    for idx, content in enumerate(crawled_data):
        for sentence_idx, content_line in enumerate(content.split("\n")):
            if sentence_idx == 0:
                df_construction.append([str(idx), content_line])
            elif content_line == '\n' or content_line == ' ' or content_line == '':
                continue;
            else:
                df_construction.append(['', content_line])
        df_construction.append(['', ''])

    df = pd.DataFrame(df_construction)
    df.to_excel("./data.xlsx", index=False, header=False)
