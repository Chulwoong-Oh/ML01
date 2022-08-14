import requests
import pandas as pd
import numpy as np
import urllib
from bs4 import BeautifulSoup

def making_newstbl(keyword, limit = 100):
    keyword_input = keyword
    keyword1 = urllib.parse.quote(keyword_input)
    base_url = "https://news.google.com"
    search_url = base_url + "/search?q=" + keyword1 + "&hl=ko&gl=KR&ceid=KR%3Ako"    
    resp = requests.get(search_url)
    html_src = resp.text
    soup = BeautifulSoup(html_src, 'html.parser')
    news_items = soup.select('div[class="xrnccd"]')

    df = pd.DataFrame()
    i = 0
    dic_news = {}

    for item in news_items[:limit]:
    # for item in news_items:
        ldata = []
        link = item.find('a', attrs={'class':'VDXfz'}).get('href')
        news_link = base_url + link[1:]
        # print(news_link)

        news_title = item.find('a', attrs={'class':'DY5T1d'}).getText()
        ldata.append(news_title)
        # print(news_title)
        # news_content = item.find('span', attrs={'class':'L8PZAb uG2FLd'}).text
        # print(news_content)

        news_agency = item.find('a', attrs={'class':'wEwyrc AVN2gc uQIVzc Sksgp slhocf'}).text
        # print(news_agency)   
        ldata.append(news_agency)

        news_reporting = item.find('time', attrs={'class':'WW6dff uQIVzc Sksgp slhocf'})
        try:
            news_reporting_datetime = news_reporting.get('datetime').split('T')
            news_reporting_date = news_reporting_datetime[0]
            news_reporting_time = news_reporting_datetime[1][:-1]
            news_reporting_text = news_reporting.text
            news_reporting_datetime1 = news_reporting.get('datetime')
        except:
            news_reporting_datetime1 = np.nan
        # print(news_reporting_date, news_reporting_time, news_reporting_text)    
        # print("\n")
        ldata.append(news_reporting_datetime1)
        dic_news[news_link] = ldata
        df = pd.DataFrame(dic_news).T.reset_index().rename(columns = {'index':'url', 0:'title', 1:'cc', 2:'datetime'}) 
    return df