#----------------------------------------------
# Load dependencies
#----------------------------------------------
from ast import keyword
from configparser import RawConfigParser
from curses import KEY_A1
from locale import normalize
from posixpath import curdir
from tkinter.messagebox import RETRY
from typing import Text

from textblob import TextBlob

import sys

import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os

import nltk

import pycountry

import re
import string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from langdetect import detect

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import display, Image
from PIL import Image
import base64

#----------------------------------------------
# DEFINE VARIABLES
#----------------------------------------------
# def run():

# print('ENTERING RUN')


# !English stopwords
stopword = nltk.corpus.stopwords.words('english')

# ! FOR STEMMING
ps = nltk.PorterStemmer()

#----------------------------------------------
# DEFINE FUNCTIONS
#----------------------------------------------


# Function 1:  FUNCTION TO CALCULATE THE PERCENTAGE
#-----------------
def percentage(part, whole):
    return 100 * float(part)/float(whole)

# Function 2:   FUNCTION TO DOWNLOAD DATA INTO CSV FILE
#-----------------
def get_table_download_link(df):
    # Reference: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="tweets.csv">Download CSV file</a>'
    return href

# Function 3: GET THE DATA FROM TWITTER
#----------------
def twitter_get(keyword, noOfTweets):

    #! Authentication for the Twitter API
    consumerKey = 'eaimmSUr6UC12KRKjt2HFEe5h'
    consumerSecret = 'CLbjYeObiyF2dHet8YaU0mGBShMVLDKAlbnjycSzxeGmJrEIGb'
    accessToken = '1517195740380114944-XPZfdj1VaZ72QrXTq5es7lxwZzlvAO'
    accessTokenSecret = 'ZVpKmKHRIJpJ7HhuqOR20FiXlFeUZ1cDUQzDNL7XAc96U'
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    #!GETTING THE TWEETS WITH TEEPY
        #!fetched tweets are in the form of Cursor objects, still need to get the text of those tweets
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang='en').items(noOfTweets)

    #!SETTING THE VARIABLES UP
    positive = 0
    negative = 0
    neutral = 0
    polarity = 0
    tweet_list = []
    neutral_list = []
    negative_list = []
    positive_list = []

    #!from Cursoir object, ass the text form of the tweet to the tweet list of each cursor tweet object
    #!at the same time, we will get the polarity scores for each tweet
    for tweet in tweets:
        tweet_list.append(tweet.text)
        analysis = TextBlob(tweet.text)
        score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
        neg = score['neg']
        pos = score['pos']
        neu = score['neu']
        comp = score['compound']
        polarity += analysis.sentiment.polarity

        if neg > pos:
            negative_list.append(tweet.text)
            negative += 1
        
        elif pos > neg:
            positive_list.append(tweet.text)
            positive += 1
        
        elif pos == neg:
            neutral_list.append(tweet.text)
            neutral += 1
    
    positive = percentage(positive, noOfTweets)
    negative = percentage(negative, noOfTweets)
    neutral = percentage(neutral, noOfTweets)
    polarity = percentage(polarity, noOfTweets)
    positive = format(positive, '.1f')
    negative = format(negative, '.1f')
    neutral = format(neutral, '.1f')


    return positive, negative, neutral, polarity, tweet_list, positive_list, negative_list, neutral_list, neg, pos, neu, comp, keyword, noOfTweets
    # return tweet_list



# Function 4: CLEANING THE DATA
#-----------------
def clean_tweets(tweet_list):
    #!DROPS ALL DUPLCIATES IN THE LIST
    # tweet_list.drop_duplicates(inplace = True)
    #!CLEANING TEXT (RE, PUNCTUATION ETC)

    #!CREATING A NEW DATAFRAME (tw_list) AND NEW FEATURES (text)
    tw_list = pd.DataFrame(tweet_list)
    tw_list['text'] = tw_list[0]

    #!REMOVING RT, PUNCTUATION ETC BY LAMNDA FUNCTION
    remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
    #!SETTING UP ALL THE CHARACTERS TO REMOVE WITH LAMBDA
    rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)

    #!REMOVING ALL THE UNNEEDED CHARACTERS AND LOWERCASING THE TEXT
    tw_list['text'] = tw_list.text.map(remove_rt).map(rt)
    tw_list['text'] = tw_list.text.str.lower()
    # tw_list.head(10)
    # print(tw_list)

    return tw_list, tw_list['text']

# Function 5:   FUNCTION FOR GETTING NUMBER OF TWEETS
#----------------

#!NUMBER OF TWEETS (TOTAL, POS, NEG, NEU)

def numberOfTweets(tweet_list_in, neutral_list_in, negative_list_in, positive_list_in):

    tweet_list = pd.DataFrame(tweet_list_in)
    neutral_list = pd.DataFrame(neutral_list_in)
    negative_list = pd.DataFrame(negative_list_in)
    positive_list = pd.DataFrame(positive_list_in)

    len_tweets = len(tweet_list)
    len_positive = len(positive_list)
    len_negative = len(negative_list)
    len_neutral = len(neutral_list)

    print('Total number of tweets : ', len(tweet_list))
    print('Total number of positive tweets: ', len(positive_list))
    print('Total number of negative tweets: ', len(negative_list))
    print('Total number of neutral tweets: ', len(negative_list))

    # return len_tweets, len_positive, len_negative, len_neutral
    return tweet_list, neutral_list, negative_list, positive_list

# tweet_list


# Function 6:   FUCNTION FOR GETTING THE SENTIMENTS
#----------------

#!CALCULATING NEGATIVE, POSITIVE, NEUTRAL, AND COMPOUND VALUES
def sentiment(tw_list):

    tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
    for index, row in tw_list['text'].iteritems():
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        if neg > pos:
            tw_list.loc[index, 'sentiment'] = "negative"
        elif pos > neg:
            tw_list.loc[index, 'sentiment'] = "positive"
        elif pos == neg:
            tw_list.loc[index, 'sentiment'] = "neutral"
        
        tw_list.loc[index, 'neg'] = neg
        tw_list.loc[index, 'pos'] = pos
        tw_list.loc[index, 'neu'] = neu
        tw_list.loc[index, 'compound'] = comp

        #!CREATING NEW DATA FRAME FOR SENTIMENTS (POS, NEG, NEU) BY IMPORTING TW_LIST
        #AND SETTING EACH NEW DATA FRAME PER EMOTION
        tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
        tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
        tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]
    
    print(tw_list.head(10), '\n')


    return tw_list_negative, tw_list_positive, tw_list_neutral


    


# Function 7:   FUNCTION FOR GETTING THE PIE CHART
#----------------
def pieChart(positive, neutral, negative, keyword):
#!Creating a PieChart based on the data for the keyword
    labels = ['Positive ['+str(positive)+'%]', 'Neutral ['+str(neutral)+'%]', 'Negative ['+str(negative)+'%]']

    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'blue', 'red']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.style.use('default')
    plt.legend(labels)
    plt.title("Sentiment Analysis Result for "+keyword+"" )
    plt.axis('equal')
    #!----UNCOMMENT TO SHOW THE PIECHART-----
    # plt.show()
    fig = plt.show()
    return fig


# Function 8a:   FUNCTION FOR CREATING WORDCLOUDS
#----------------
def create_wordcloud(text, file_Name):
        mask = np.array(Image.open("/Users/alex/Documents/Personal Projects/twitterSentimentAnalysis/cloud.png"))
        stopwords = set(STOPWORDS)
        #!CREATING WORDCLOUD OBJECT
        wc = WordCloud(background_color="white",
        mask = mask,
        max_words=3000,
        stopwords=stopwords,
        repeat=True)
    
        wc.generate(str(text))

        #!SAVE TO absolute path -> wc.png
        wc.to_file("/Users/alex/Documents/Personal Projects/twitterSentimentAnalysis/"+file_Name+".png")
        print("Word Cloud Saved Successfully")
        path="/Users/alex/Documents/Personal Projects/twitterSentimentAnalysis/"+file_Name+".png"
        display(Image.open(path))

    #!----UNCOMMENT TO GET THE WORDCLOUD OUTPUTS------

    #!CREATING WORDCLOUD FOR ALL TWEETS, POS, NEG, NEU

    #? PUT THIS IN MAIN MAYBE?
    # create_wordcloud(tw_list['text'].values, 'total')
    # create_wordcloud(tw_list_positive["text"].values, 'positive')
    # create_wordcloud(tw_list_neutral["text"].values, 'neutal')
    # create_wordcloud(tw_list_negative["text"].values, 'negative')

# Function 8b:   FUNCTION FOR CREATING WORDCLOUDS FOR STREAMLIT
#----------------

def create_wordcloud_b(text, file_Name):
    mask = np.array(Image.open("/Users/alex/Documents/Personal Projects/twitterSentimentAnalysis/cloud.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
        mask = mask,
        max_words=3000,
        stopwords=stopwords,
        repeat=True)
    wc.generate(str(text))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

    cloud = plt.show()
    return cloud

# Function 9:   FUNCTION FOR CALUCLATING MEAN LENTGH OF TWEETS AND MEAN WORD COUNT
#----------------
    #!TIME TO CALCULATE THE LENTGH OF A TWEET AND ITS WORD COUNT
def calculate_text_mean(tw_list):
    tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
    tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))

    text_mean_len = round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()),2)
    text_mean_word_count = round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()),2)
    print(text_mean_len, '\n')
    print(text_mean_word_count, '\n')

    return text_mean_len, text_mean_word_count




# Function 10:   FUNCTION FOR COUNT VECTORISATION
#----------------

    #?THIS IS IF YOU WANT VECTORISATION BUT ALSO NEED THIS IF WANT TO SEE MOST USED WORDS AND N-GRAMS
#! Applying count vectorizer provides the capability to preprocess your
#! text data prior to generating the vector representation making 
#! it a highly flexible feature representation module for text. 
#! After count vectorizer, it is possible to analyze the words with 
#! two or three or whatever you want.


def count_vectorizer(tw_list):
    tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punc(x))
    tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))
    tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))
    # tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))

    #!Appliyng Countvectorizer
    countVectorizer = CountVectorizer(analyzer=clean_text) 
    countVector = countVectorizer.fit_transform(tw_list['text'])

    count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
    count_vect_df.head()

    # print("-------IN COUNT VECTORISER---------")
    # top = tw_list['stemmed'].value_counts().idxmax()
    # print()
    # print(top)

    return count_vect_df, tw_list['text']



# Function 11:   FUNCTIONS FOR CLEANING TEXT OF COUNT VECTORISATION
#----------------


#!#FUNCTION FOR REMOVING PUNCTUATION
def remove_punc(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


#! FUCNTION FOR APPLYING TOKENIZATION
def tokenization(text):
    text = re.split('\W+', text)
    return text


#!FUCNTION FOR REMOVING STOPWARDS
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]   
    return text


#!FUCNTION FOR APPLYING THE STEMMER TO GET ONLY THE ROOTS
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

#!CLEANING THE TEXT
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword] #REMOVES STOPWORDS AND STEMMING
    return text

#!PRINTS OUT THE TABLE
# print(tw_list.head())


# Function 12:   FUNCTIONS FOR MOST COMMON WORDS AFTER COUNT VECTORISATION
#----------------
def most_common(count_vect_df):
    # print('most common-------------')
#!GETTING THE MOST USED WORDS FROM ALL TWEETS COMBINED 
    count = pd.DataFrame(count_vect_df.sum())
    countdf = count.sort_values(0, ascending=False).head(20)
    # print(countdf[0], '\n')
    # top = countdf[0]

    # print(top)
    # print(countdf[1:11])

    return countdf


# Function 13:   FUNCTIONS FOR N-GRAMS
#----------------
#!INTERESTING TO SEE WHAT WORDS ARE USED IN COMBINATION
#!FUCNTION FOR N-GRAMS

def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    print(words_freq[:n])
    return words_freq[:n]

# Function 14: N-GRAM (THIS ONE WORKS WITH STREAMLIT)
#-----------------
# top_n = 5
# n = 2 ie bigram
def tweets_ngrams(n, top_n, tw_list):
    text = tw_list['text']
    words = clean_text(text)
    result = (pd.Series(data = nltk.ngrams(words, n), name = 'frequency').value_counts())[:top_n]
    return result.to_frame()

#?------------------------------------------

    # #STARTING THE SENTIMENT ANALYSIS
    # #!LET THE USER DECIDE WHAT THEY WANT TO SEARCH FOR AND HOW MANY TWEETS SHOULD BE SCRAPED
    # keyword = input("Enter keyword or hashtag to analyse: ")
    # print()
    # noOfTweets = int(input("Enter the amount of keyword tweets to analyse: "))


    #!THIS FUNCTION IS USED ONLY FOR THE CIRCLE CONSTRUCTION
    #!UNCOMMENT IF WANTING THO SHOW CIRCLE INSTEAD OF PIE CHART
    ##FUNCTION TO COUNT THE VALUES in COLUMNS
    # def count_values_in_column(data,feature):
    #     total=data.loc[:,feature].value_counts(dropna=False)
    #     percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    #     return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
    #!THIS CREATE A CIRCLE< NOT THE PIE CHART
    # #Count_values for sentiment
    # THIS IS USING THE ABOVE FUNTION, MAYBE TO MAIN.PY
    # pc = count_values_in_column(tw_list,"sentiment")
    # #CREATING A DATA POINT FOR A CIRCLE
    # piechart = count_values_in_column(tw_list, "sentiment")
    # names= pc.index
    # size= pc["Percentage"]
    # #CREATE A CIRCLE FOR THE CENTER OF THE PLOT
    # my_circle=plt.Circle( (0,0), 0.7, color='white')
    # plt.pie(size, labels=names, colors=['green', 'blue', 'red'])
    # p=plt.gcf()
    # p.gca().add_artist(my_circle)
    # plt.show()

    #?USE THIS IN THE MAIN?
    # #!2: Bigram
    # bigram = get_top_n_gram(tw_list['text'],(2,2), 20)
    # print(bigram, '\n')

    # #?USE THIS IN THE MAIN?
    # #!3 Trigram
    # trigram = get_top_n_gram(tw_list['text'], (3,3), 20)
    # print(trigram, '\n')


