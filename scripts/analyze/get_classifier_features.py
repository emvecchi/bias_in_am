import pandas as pd
import numpy as np
import sys, json, os, ast
import seaborn as sns
import matplotlib.pyplot as plt
from functions import *


# Get features from automatic classifiers (toxicity, sentiment, emotion)

datafile = '/mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/intermediate-data/tal_elal_2016/train_period_data.annotations-ver2.masked.topics.jsonlist'
with open(datafile, 'r') as f:
	data = [json.loads(line) for line in f]

full_df = pd.DataFrame(data)

full_df.rename(columns={'selftext': 'text'}, inplace=True)
df = full_df[['id','text', 'masked_selftext']]

#clean up text a bit
df['tmp_text'] = df.loc[:,'text'].str.replace(r'\*Hello, users of CMV!.*', '', regex=True)
df['tmp_text'] = df.loc[:,'tmp_text'].str.replace(r'\n', '', regex=True)
#text_list = df['tmp_text'].tolist()

df['tmp_masked_selftext'] = df.loc[:,'masked_selftext'].str.replace(r'\*Hello, users of CMV!.*', '', regex=True)
df['tmp_masked_selftext'] = df.loc[:,'tmp_masked_selftext'].str.replace(r'\n', '', regex=True)

#### Get Toxicity scores for all datapoints, including for comments
toxicity_df = df
toxicity_scores = toxicity_classifier(df, 'tmp_text')
toxicity_df['toxicity_neutral'] = [x[0] for x in toxicity_scores]
toxicity_df['toxicity_toxic'] = [x[1] for x in toxicity_scores]
#toxicity_df['toxicity_scores'] = toxicity_scores

comment_info = pd.DataFrame(columns=['id', 'avg_comment_toxicity'])
for _, row in full_df.iterrows():
    comments_df = pd.json_normalize(row['comments'])
    if comments_df.empty:
    	new_entry = {'id':row['id'], 'avg_comment_toxicity':0.0}
    else:
    	comment_count = float(comments_df.shape[0])
    	toxicity_scores = toxicity_classifier(comments_df, 'body')
    	avg_toxicity = sum([x[1] for x in toxicity_scores]) / comment_count
    	new_entry = {'id':row['id'], 'avg_comment_toxicity':avg_toxicity}
    comment_info = comment_info.append(new_entry, ignore_index=True)


toxicity_df = pd.merge(toxicity_df, comment_info, on='id', how='left')
toxicity_df = toxicity_df.drop(columns=['tmp_text','text'])
toxicity_df.to_csv('data/bias_in_AM/train_toxicity_scores.csv', index=False)


sentiment_df = df
predicted_labels, sentiment_scores  = sentiment_classifier(df, 'tmp_text')
sentiment_df['sentiment_positive'] = [x[0] for x in sentiment_scores]
sentiment_df['sentiment_neutral'] = [x[1] for x in sentiment_scores]
sentiment_df['sentiment_negative'] = [x[2] for x in sentiment_scores]
sentiment_df["sentiment"] = predicted_labels

comment_info = pd.DataFrame(columns=['id', 'avg_comment_sentiment'])
for _, row in full_df.iterrows():
    comments_df = pd.json_normalize(row['comments'])
    if comments_df.empty:
    	new_entry = {'id':row['id'], 'avg_comment_sentiment':0.0}
    else:
    	comment_count = float(comments_df.shape[0])
    	predicted_labels, sentiment_scores = sentiment_classifier(comments_df, 'body')
    	avg_sentiment = sum([x[1] for x in sentiment_scores]) / comment_count
    	new_entry = {'id':row['id'], 'avg_comment_sentiment':avg_sentiment}
    comment_info = comment_info.append(new_entry, ignore_index=True)

sentiment_df = pd.merge(sentiment_df, comment_info, on='id', how='left')
sentiment_df = sentiment_df.drop(columns=['tmp_text','text'])
sentiment_df.to_csv('data/bias_in_AM/train_sentiment_scores.csv', index=False)


aq_df = df 
aq_predictions = aq_classifier(df,'tmp_text')
aq_df['aq_score'] = [x for x in aq_predictions]
masked_aq_predictions = aq_classifier(df,'tmp_masked_selftext')
aq_df['aq_masked_score'] = [x for x in masked_aq_predictions]

comment_info = pd.DataFrame(columns=['id', 'avg_comment_aq'])
for _, row in full_df.iterrows():
    comments_df = pd.json_normalize(row['comments'])
    if comments_df.empty:
    	new_entry = {'id':row['id'], 'avg_comment_aq':0.0}
    else:
    	comment_count = float(comments_df.shape[0])
    	aq_predictions = aq_classifier(comments_df, 'body')
    	avg_aq = sum([x for x in aq_predictions]) / comment_count
    	new_entry = {'id':row['id'], 'avg_comment_aq':avg_aq}
    comment_info = comment_info.append(new_entry, ignore_index=True)

aq_df = pd.merge(aq_df, comment_info, on='id', how='left')

masked_aq_df = df[df['tmp_masked_selftext'].notna()].copy()
masked_aq_predictions = aq_classifier(masked_aq_df,'tmp_masked_selftext')
masked_aq_df['aq_masked_score'] = [x for x in masked_aq_predictions]
aq_df = pd.merge(aq_df, masked_aq_df, on='id', how='left')

aq_df = aq_df.drop(columns=['tmp_text','text', 'masked_selftext', 'tmp_masked_selftext'])
aq_df.to_csv('data/bias_in_AM/train_aq_scores.csv', index=False)





