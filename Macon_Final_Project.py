'''
Author: Bethany Macon
Contact information: bcm316@live.unc.edu
Program title: Data set analyzer
Program description: This program runs a series of descriptive analyses on a data set
provided by the user. Specifically, it analyzes general characteristics of posts discussing a prominent makeup artist
and analyzes the sentiment analysis of posts in comparison to each other.
'''

import warnings
warnings.simplefilter(action='ignore')
import pandas as pd
from afinn import Afinn
from nltk.corpus import stopwords
from datetime import datetime
sentiment = Afinn()

print('This program seeks to analyze general descriptive characteristics of a Reddit data set.')
print('\n')
df = pd.read_csv(input('Please enter the name of your data set: '))

#this function gets rid of unecessary characters so that the program will run faster
def text_cleaning(df):
    df['clean_text'] = df.title
    df.clean_text = df.clean_text.str.lower().str.replace('[^a-z ]', '')
    stop = stopwords.words('english')
    df.clean_text = df.clean_text.apply(lambda x: ' '.join(i for i in x.split() if i not in stop))
    return df

#this function converts dates in timestamp form to a legible form to be able to compute longitudinal analyses
def date_converter(dates):
    dates_converted = []
    for date in dates:
        clean_date = datetime.fromtimestamp(date).isoformat()
        dates_converted.append(clean_date[0:10])
    return dates_converted

#the following subsets the data to only posts mentioning Jeffree Star (which is what is being used as the main data set)
star = df[df.title.str.lower().str.contains('jeffree star|jeffree|star')]

cleaned_dates = date_converter(star.created_utc)
star['cleaned_dates'] = cleaned_dates

print('\n')
print('Characteristics of Jeffree Star data set:')
print('A total of', len(star.title), 'posts discussed the makeup artist Jeffree Star, which is',
      str(round((len(star.title)/len(df.title)*100), 2))+ '% of the full data set.')
print('There are', len(star.author.drop_duplicates()), 'unique users who posted at least once in this data set.')
print('There are', len(star.subreddit.drop_duplicates()), 'unique subreddits that posts were published in.')
print('The average "score" across posts is:', round(star.score.mean(), 2), '(standard deviation = '+
      str(round(star.score.std(), 2))+ ').')
print('The average number of comments per post is:', round(star.num_comments.mean(), 2), '(standard deviation = '+
      str(round(star.num_comments.std(), 2))+ ').')

print('\n')
print('The Jeffree Star data set is split into two subsections based on sentiment scores of posts.')

#the following applies the sentiment score function to the Star data set and subsets the data based on positive or
#negative sentiment scores
star['sentiment'] = star.title.apply(sentiment.score)
star_positive = star[star.sentiment > 0]
star_negative = star[star.sentiment < 0]

print('\n')
print('Characteristics of positive Jeffree Star posts:')
print('There is a total of', (len(star_positive)), 'positive posts, which is',
      str(round((len(star_positive.title)/len(star.title)*100), 2))+ '% of the full Jeffree Star data set.')
print('The average sentiment of positive Jeffree Star posts is', str(round(star_positive.sentiment.mean(), 2))+ '.')
print('The top ten users whose posts about Jeffree Star had an overall positive sentiment were:')
print(star_positive.author.value_counts().head(10))
#the following subsets the data into posts with positive sentiments that also had high engagement based on score mean
high_pos = star_positive[star_positive.score >= star_positive.score.mean()]
print('A total of', len(high_pos.title), 'positive posts had high engagement (based on score), which is',
      str(round((len(high_pos.title)/len(star_positive.title)*100), 2))+ '% of the positive data set.')

print('\n')
print('Characteristics of negative Jeffree Star posts:')
print('There is a total of', (len(star_negative)), 'negative posts, which is',
      str(round((len(star_negative.title)/len(star.title)*100), 2))+ '% of the full Jeffree Star data set.')
print('The average sentiment of negative Jeffree Star posts is', str(round(star_negative.sentiment.mean(), 2))+ '.')
print('The top ten users whose posts about Jeffree Star had an overall negative sentiment were:')
print(star_negative.author.value_counts().head(10))
#the following subsets the data into posts with negative sentiments that also had high engagement based on score mean
high_neg = star_negative[star_negative.score >= star_negative.score.mean()]
print('A total of', len(high_neg.title), 'negative posts had high engagement (based on score), which is',
      str(round((len(high_neg.title)/len(star_negative.title)*100), 2))+ '% of the negative data set.')

print('\n')
print('The Jeffree Star full data set will now give characteristics of the Blood Lust Collection.')

#the following subsets the Star data into posts specifically mentioning Star and his new collection
blood_lust = star[star.title.str.lower().str.contains('blood|lust|collection|blood lust collection|blood lust palette|'  
                                                      'makeup|palette')]

print('\n')
print('Blood Lust Collection Characteristics:')
print('There are', len(blood_lust), 'posts that discuss Jeffree Star and his Blood Lust Collection, which is',
      str(round((len(blood_lust)/len(star))*100, 2))+ '% of the full Jeffree Star data set.')
print('The average sentiment score of posts describing the Blood Lust Collection is',
      str(round(blood_lust.title.apply(sentiment.score).mean(), 2))+ '.')
#the following is used to subset the blood lust data into posts mentioning 'love' and 'hate' to determine how often each
#was used
love = blood_lust[blood_lust.title.str.lower().str.contains('love')]
print("The term 'love' is used", len(love), 'times in posts discussing the Blood Lust Collection, which is',
       str(round((len(love)/len(blood_lust))*100, 2))+ '% of the total Blood Lust Collection posts.')
hate = blood_lust[blood_lust.title.str.lower().str.contains('hate')]
print("The term 'hate' is used", len(hate), 'times in posts discussing the Blood Lust Collection, which is',
       str(round((len(hate)/len(blood_lust))*100, 2))+ '% of the total Blood Lust Collection posts.')
print('The top ten dates with posts discussing the Blood Lust Collection are:')
print(blood_lust.cleaned_dates.value_counts().head(10))

#the following 4 lines take aggregate results from the comments and score per post from the full Star data set and turn
#them into new csv files
result1 = star.groupby(star.cleaned_dates).num_comments.agg(['sum', 'mean', 'max'])
result1.to_csv('star_results.csv')
result2 = star.groupby(star.cleaned_dates).score.agg(['sum', 'mean', 'max'])
result2.to_csv('star_results2.csv')

print('\n')
print('Thank you for using this program.')
