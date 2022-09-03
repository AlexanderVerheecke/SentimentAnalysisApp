#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 1.1: Load libraries
#------------------------------------#
from nltk.featstruct import _default_fs_class
from numpy import e

import streamlit as st


import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
import tweepy as tw
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt

import altair as alt
import time


#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 1.2: Load custom library
#------------------------------------#
import twitterFunctions as tf # custom functions file

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=
##------SETTING UP GUI------


# 2.1: Main Panel Setup
#------------------------------------#

## 2.1.1: Main Layout
##----------------------------------##
st.set_page_config(layout="wide") # page expands to full width


## 2.1.2: Main Logo
##----------------------------------##
# image = Image.open('uob.png') #logo
# st.image(image, width = 350) #logo width

## 2.1.3: Main Title
##----------------------------------##
st.title('Sentiment Analysis for Twitter by Alexander Verheecke') #



#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 2.2: Sidebar Setup
#------------------------------------#

## 2.1.1: Sidebar Title
##----------------------------------##
st.sidebar.header("Input your search choices:")

## 2.2.2: Sidebar Input Fields
##----------------------------------##
with st.form(key ='form_1'):
    # user_word = ""
    # tweet_count = ""
    with st.sidebar:
        user_word = st.sidebar.text_input("Hashtag to anlayse", "London", help='Ensure that the field is not empty.')
        # num_of_tweets = st.sidebar.slider("Select the number of Latest Tweets to Analyze", 0, 50, 1)
        num_of_tweets = st.sidebar.number_input("Maximum number of tweets", min_value=20, max_value=500, value = 20, step = 1, help = 'Returns the specified amount of most recent tweets. A minimum of 20 and a maximum of 500 tweets can be analysed. The more tweets specified, the longer analysing will take.')
        option = st.selectbox('N-gram model',('Unigram', 'Bigram', 'Trigram'), help='N-gram: Most common continous sequence of words. Unigram: 1 word, Bigram: 2 words, Trigram: 3 words.')
        st.sidebar.text("") # spacing
        submit_button = st.form_submit_button(label = 'Analyse tweets', help = 'Re-run analyzer with the current inputs.')

# Loading message for users
if submit_button:
    if user_word == "":
        st.error("Type a hashtag!")
        st.stop()
    if num_of_tweets < 20 or  num_of_tweets > 500:
        st.error("Type a number between 20 and 500!")
        st.stop()
    if len(user_word) != 0:
        with st.spinner('Getting data from Twitter...'):
            time.sleep(10)
            # num_of_tweets = str(num_of_tweets)
            st.success('Analysis is done! You searched for the last ' + 
                        str(num_of_tweets) + 
                        ' tweets that used #' + 
                        user_word+". Please wait for data representation to be complete.")


## 2.2.3: Sidebar Information
##----------------------------------##
st.sidebar.header("Instructions:")
st.sidebar.write("For the best experience, select the dropdown icon in the upper right corner, go to 'Settings' and in 'Theme', select 'Light'.")
st.sidebar.write("To use the sentiment analyser, type the hashtag to analyse in the first field and the amount of tweets in the second field. "+
"Pressing 'Analyse tweets' will start the process. Please allow a couple of seconds for the algorithm to analyse the data show the results. "+
"For help, hover over the '?' next to the input field.")
st.sidebar.write("")
st.sidebar.write("As a demonstration, an intial analysis with the following query has been done: hashtag = 'London', number of tweets = 20, represented by a unigram.")
st.write("")
st.sidebar.header("About this app:")
st.sidebar.markdown("This application is build using Streamlit. At its core, the code runs on python and makes use of Twitters"
+" own API v2 to collect public Tweets. Various libraries have been imported to faciliate conducting sentiment analyisis tasks. "+ 
"Main packages include 'NLTK', 'NUMPY', 'PANDAS, 'TWEEPY', 'MATPLOTLIB', and 'TEXBLOB.")



#--------------------------------------------------
# PART 3: APP DATA SETUP
#--------------------------------------------------
# - 3.1: Twitter data ETL
# - 3.2: Define key variables
#--------------------------------------------------

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 3.1: Twitter Data ETL

# Layout
#------------------------------------#

# Run Function 3: GET THE DATA FROM TWITTER

# df_tweets, df_new = tf.twitter_get(user_word, num_of_tweets)
positive, negative, neutral, polarity, tweet_list, positive_list, negative_list, neutral_list, neg, pos, neu, comp, keyword, noOfTweets = tf.twitter_get(user_word, num_of_tweets)
# st.write('twitter_get done')

# Run Function 4: CLEANING THE DATA
tw_list, tw_list['text'] = tf.clean_tweets(tweet_list)
# df_tweets = tf.feature_extract(df_tweets)
# st.write('Cleaning data done')

# Function 5:   FUNCTION FOR GETTING NUMBER OF TWEETS
# len_tweets, len_positive, len_negative, len_neutral = tf.numberOfTweets(tweet_list, neutral_list, negative_list, positive_list)
tweet_list, neutral_list, negative_list, positive_list = tf.numberOfTweets(tweet_list, neutral_list, negative_list, positive_list)

# Function 10:   FUNCTION FOR COUNT VECTORISATION

count_vect_df, tw_list['text'] = tf.count_vectorizer(tw_list)




#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=


# 3.2: Define Key Variables
#------------------------------------#
user_num_tweets = str(num_of_tweets)
total_tweets = len(tweet_list)

# total_tweets = len(df_tweets['full_text'])
# highest_retweets = max(df_tweets['rt_count'])
# highest_likes = max(df_tweets['fav_count'])



#--------------------------------------------------
# PART 4: APP DATA & VISUALIZATIONS
#--------------------------------------------------
# - 4.1: UX messaging
# - 4.2: Sentiment analysis
# - 4.3: Descriptive analysis
# - 4.4: Topic model analysis
#--------------------------------------------------



# 4.1: UX Messaging
#------------------------------------#

# Loading message for users
# if submit_button:
#     if user_word == "":
#         st.error("Type a hashtag!")
#         st.stop()
#     if len(user_word) != 0:
#         with st.spinner('Getting data from Twitter...'):
#             time.sleep(8)
#             num_of_tweets = str(num_of_tweets)
#             st.success('Analysis is done! You searched for the last ' + 
#                         num_of_tweets + 
#                         ' tweets that used #' + 
#                         user_word)


#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=


# 4.2: Sentiment Analysis
#------------------------------------#

# # Subtitle 
# st.header('Sentiment Analysis')

# # Get sentiment
tw_list_positive, tw_list_neutral, tw_list_negative= tf.sentiment(tw_list)
# st.write('Getting sentiment done')

# # Get sentiment scores on raw tweets
# text_sentiment = tf.get_sentiment_scores(df_tweets, 'full_text')

# # Add sentiment classification
# text_sentiment = tf.sentiment_classifier(df_tweets, 'compound_score')

# # Select columns to output
# df_sentiment = df_tweets[['created_at', 'full_text', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'compound_score']]

# # Sentiment group dataframe
# sentiment_group = df_sentiment.groupby('sentiment').agg({'sentiment': 'count'}).transpose()s



# Function 9:   FUNCTION FOR CALUCLATING MEAN LENTGH OF TWEETS AND MEAN WORD COUNT


## 4.2.1: Summary Information
##----------------------------------##


# Sentiment percentage
st.subheader('Summary')
st.write("Out of "+ str(len(tweet_list))+" tweets, "+str(len(negative_list))+" were negative, " +str(len(neutral_list))+ " were neutral, and " +str(len(positive_list)) +" were positive.")

col1, col2, col3 = st.columns(3)
negative_percentage = tf.percentage(len(negative_list), len(tweet_list))
neutral_percentage = tf.percentage(len(neutral_list), len(tweet_list))
positive_percentage = tf.percentage(len(positive_list), len(tweet_list))
col1.metric("% Negative Tweets:", negative_percentage)
col2.metric("% Neutral Tweets:", neutral_percentage)
col3.metric("% Positive Tweets:", positive_percentage)

#Most common
# countdf = tf.most_common(count_vect_df)
# st.write("The most common words were: "+ countdf)


# n-grams-------
ngram_range = 2
# freq_words = tf.get_top_n_gram(tw_list,ngram_range,n=None)
# st.write("The most frequent words were: "+ freq_words)

if option == "Unigram":
    ngram_num = 1

if option == "Bigram":
    ngram_num = 2

if option == "Trigram":
    ngram_num = 3


ngram_visual = tf.tweets_ngrams(ngram_num, 5, tw_list)
ngram_visual['ngram'] = ngram_visual.index
ngram_bar = alt.Chart(ngram_visual).mark_bar().encode(
                    x = alt.X('frequency', axis = alt.Axis(title = 'Word Frequency')),
                    y = alt.Y('ngram', axis = alt.Axis(title = 'Ngram'), sort = '-x'),
                    tooltip = [alt.Tooltip('frequency', title = 'Ngram Frequency')],#,  alt.Tooltip('Ngram', title = 'Ngram Word(s)')] ,
                ).properties(
                    height = 350
                )
                
st.subheader('N-gram: 5 most common '+option+"s")
st.altair_chart(ngram_bar, use_container_width=True)



####PIE CHART
st.subheader('Pie Chart')
st.set_option('deprecation.showPyplotGlobalUse', False)
fig = tf.pieChart(positive, neutral, negative, keyword)
st.pyplot(fig)


#WORD CLOUD
st.subheader('Positive Wordcloud')
cloud = tf.create_wordcloud_b(tw_list_positive["text"].values, 'positive')
st.pyplot(cloud)
st.subheader('Neutral Wordcloud')
cloud = tf.create_wordcloud_b(tw_list_neutral["text"].values, 'neutal')
st.pyplot(cloud)
st.subheader('Negative Wordcloud')
cloud =tf.create_wordcloud_b(tw_list_negative["text"].values, 'negative')
st.pyplot(cloud)


# cloud = tf.create_wordcloud_b(tw_list['text'].values, 'total')
# st.pyplot(cloud)

# cloud = tf.create_wordcloud_b(tw_list_neutral["text"].values, 'neutal')
# st.pyplot(cloud)
# cloud =tf.create_wordcloud_b(tw_list_negative["text"].values, 'negative')
# cloud = tf.create_wordcloud_b(tw_list['text'].values)
# # cloud = tf.create_wordcloud
# st.pyplot(cloud)

