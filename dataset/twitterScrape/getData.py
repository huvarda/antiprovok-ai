import twint
import csv

import requests


keywordList = []

def scrapePoliticalTweetReplies(hashtagList, outfile, debug=False):
    for hashtag in hashtagList:
        tweets = []

        repliesConf = twint.Config()
        repliesConf.Near = "Istanbul"
        repliesConf.Lang = "TR"
        
        repliesConf.Search = hashtag
        repliesConf.Hide_output = True
        repliesConf.Store_csv = True
        repliesConf.Output = "./newCsv/" + hashtag + outfile
        repliesConf.Show_hashtags = True
        
        repliesConf.Store_object = True
        repliesConf.Store_object_tweets_list = tweets
            
        twint.run.Search(repliesConf)


scrapePoliticalTweetReplies(keywordList, "Tweets.csv", debug=False)

