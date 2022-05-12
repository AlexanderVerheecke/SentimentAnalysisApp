# SentimentAnalysisApp
A Twitter sentiment analysis web application based on python and developed into a web application using streamlit. Public tweets are collected with the help of Twitter's own API v2. The user is required to indicate the hashtag and amount of tweets to analyse. A sentiment analysis will be conducted through a rule based sentiment analysis algorithm (VADER), resulting in the following features summarised in a summary:

1. Sentiment percentage: A percentage indicator showing how many tweets are classified as either positive, neutral, and negative.

2. N-gram diagram: The user specifies if the data is collected in either a unigram, bigram, or trigram. A grapg with the 5 most common words based on the n-gram will be shown.

3. Positive/Neutral/Negative wordcloud: A wordcloud with all the common vocabulary items is created based on its sentiment.

Link to the web application: https://share.streamlit.io/alexanderverheecke/sentimentanalysisapp/main/main.py


Future updates to the application planned once my schedule frees up:
- The ability to check tweets in different languages.
- The ability to specify what model to use for analyisis (Naïve Bayes, Deep Neural Network) and possibly compare the results.
- Lastly, the option to analyse posts from a different website such as Reddit.
