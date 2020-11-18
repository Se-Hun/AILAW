# """
# Routes and views for the flask application.
# """
import copy
import pymongo
from datetime import date
from datetime import datetime
import re
from flask import Response 
import os
from datetime import datetime
from flask import render_template
from HelloWorld import app
from flask_restful import reqparse , abort ,Api , Resource 
from flask_cors import CORS 
from flask import request
import xlrd
import json
import ast
import pyexcel_ods
import sys
import codecs
from itertools import permutations
from collections import OrderedDict 
from orderedset import OrderedSet

from flask_jwt_extended import JWTManager
from flask_jwt_extended import create_access_token

from werkzeug.security import generate_password_hash , check_password_hash
from pymongo import MongoClient #MongoDB를 위해 import시킴(세훈)

import requests
import re
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait


jwt = JWTManager(app)
CORS(app)
api = Api(app)


#현재 시간 설정  2019-04-03 같이 넣기때문에  년월일을 넣어준다


def crawlCases(driver):
    url_list = driver.find_element_by_xpath('//*[@id="search_result"]/div/article/dl').find_elements_by_tag_name('a')

    contents = []
    url = url_list[0]
    url.click()

    window_before = driver.window_handles[0]
    window_after = driver.window_handles[1]

    driver.switch_to.window(window_after)
    return driver.current_url

# Press the green button in the gutter to run the script.

def getCrime(url) :
    response = requests.get(url).text
    # print(response)
    iframe_url = re.findall('<iframe(.*?)</iframe>', response, re.I|re.S)[0]
    iframe_url = re.findall('"([^"]*)"', iframe_url)[0]

    response = requests.get(iframe_url)
    response.encoding = 'utf-8'
    text = response.text

    try :
        crime = re.findall("<CASEREASONING>(.*)</CASEREASONING>", text, re.I|re.S)[0]
    except :
        return getCrime_2(url)

    crime = crime.split("</br>")
    for idx, sentence in enumerate(crime) :
        if sentence.find("증거") != -1 and sentence.find("요지") != -1:
            crime = crime[1:idx]

    crime = list(map(lambda x : x.strip(), crime))

    return crime


def getCrime_2(url):
    response = requests.get(url).text

    iframe_url = re.findall('<iframe(.*?)</iframe>', response, re.I | re.S)[0]
    iframe_url = re.findall('"([^"]*)"', iframe_url)[0]

    response = requests.get(iframe_url)
    response.encoding = 'utf-8'
    text = response.text

    crime = re.findall("<CONTENTSOFCASE>(.*)</CONTENTSOFCASE>", text, re.I | re.S)[0]
    crime = crime.split("<br/>")

    start = -1
    end = -1
    for idx, sentence in enumerate(crime):
        if sentence.find("증거") != -1 and sentence.find("요지") != -1 :
            end = idx
        elif sentence.find("범죄") != -1 and sentence.find("사실") != -1 and start == -1:
            start = idx

    crime = list(map(lambda x: x.strip(), crime[start:end]))
    # print(crime)
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    crime = list(map(lambda x : _RE_COMBINE_WHITESPACE.sub(" ", x).strip(), crime))
    # print("CRIME 2")
    return crime

@app.route('/extractCrime' , methods=['POST','OPTIONS'])
def paperSearchByPaperName():

    if request.method == 'POST' or request.method == 'OPTIONS':
        content = request.json

        if content :
            name = content.get('name')
            
    else :    
        print("No POST OPTIONS", file=sys.stderr)

    driver = webdriver.Chrome('./chromedriver')
    url = "https://legalsearch.kr/list/all?keyword=" + name
    wait = WebDriverWait(driver, 10)
    driver.get(url)

    case_url = crawlCases(driver)
    crime = getCrime(case_url)
    driver.quit()



