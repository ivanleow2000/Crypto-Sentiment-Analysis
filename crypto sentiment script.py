# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:36:06 2022
Title: Crypto news sentiment analysis
@author: Ivan Leow
"""

from newsdataapi import NewsDataApiClient
import pandas as pd
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set working directory
WD = 'C:/Users/61435/Downloads/Others/Personal Projects/Crypto sentiment analysis/'

# Initialise client with key
api = NewsDataApiClient(apikey="") #Change accordingly
print("Connected to API")

# See documentation for full list of arguements and paramters at https://newsdata.io/docs
api_response = api.news_api(language = "en", q = "ethereum" and "bitcoin" and "cardano", country = None)

# Saving data in JSON format for record keeping
news_dump_df = pd.DataFrame(api_response)
now = datetime.now()
current_time = now.strftime("%d_%m_%Y_%H_%M_%S") # Day, Month, Year, Hour, Minute, Second
news_dump_df.to_json(WD + "News dump/" + current_time + '_news_dump.json')

#######################################################################################################################################

# Calling previously saved data
# Cleaning news_dump_df to show only results, converting dictionary in results column into a df
news_dump_df = pd.read_json(WD + "News dump/" + current_time + '_news_dump.json')
news_dump_df2 = news_dump_df['results']
news_dump_df2 = pd.DataFrame(news_dump_df2.tolist())

count = news_dump_df2['title'].count()
print(f'Number of News Items Today on New York Times: {count}')

### Finding vader scores ###

# Instantiate sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()
# Iterate through the headlines and get the polarity scores using vader
scores = news_dump_df2['title'].apply(vader.polarity_scores).tolist() #use 'title' or 'description'
# Convert the 'scores' list of dicts into a DataFrame
scores_df = pd.DataFrame(scores)
# Joins news dump with the vader scores calculated
final_df = news_dump_df2.join(scores_df, rsuffix='_right')
# Save df as CSV for closer analysis
final_df.to_csv(WD + 'vader_scores.csv')
# Averaging and totalling compound scores of every news item
compound = final_df['compound'].mean()
print(f'Overall Market Sentiment Today: {compound}')