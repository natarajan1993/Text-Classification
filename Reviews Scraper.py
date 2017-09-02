"""Simple review scraper from the Sears website"""
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re
pd.options.display.max_colwidth = 5000

pages = pd.read_csv(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Misc\Week 9\Sears_pages.csv')
rating_list,num_reviews_list,item_num_list,model_num_list = [],[],[],[]

for url in pages['Webpage']:
    soup = BeautifulSoup(urlopen(url), 'lxml')

    rating_list.append(soup.find('meta',itemprop = 'ratingValue')['content'])
    num_reviews_list.append(''.join(re.findall(r'\d+', soup.find('span',{'class':'product-ratings-count'}).text)))
    item_num_list.append(''.join(re.findall(r'\d+P',soup.find('small',itemprop = 'productID').text)))
    model_num_list.append(soup.find('span',itemprop = 'model').text)
    
result = pd.DataFrame({'Item Number':item_num_list,
                       'Model Number':model_num_list,
                       'Number of reviews':num_reviews_list,
                       'Rating':rating_list})
result
# result.to_csv(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Misc\Week 9\Sears_pages_results_071017.csv')
