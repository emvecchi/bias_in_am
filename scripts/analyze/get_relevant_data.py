import pandas as pd
import numpy as np
import sys, json, os, ast
import seaborn as sns
import matplotlib.pyplot as plt
from functions import *

train_datafile = '/mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/intermediate-data/tal_elal_2016/train_period_data.annotations-ver2.topics.jsonlist'
train_featurefile='/mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/intermediate-data/tal_elal_2016/feature_sets/train_op_final.csv'

heldout_datafile = '/mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/intermediate-data/tal_elal_2016/heldout_period_data.annotations-ver2.topics.jsonlist'
heldout_featurefile='/mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/intermediate-data/tal_elal_2016/feature_sets/heldout_op_final.csv'

train_classifierfeaturefiles=['data/bias_in_AM/train_toxicity_scores.csv','data/bias_in_AM/train_sentiment_scores.csv', 'data/bias_in_AM/train_aq_scores.csv']
heldout_classifierfeaturefiles=['data/bias_in_AM/heldout_toxicity_scores.csv','data/bias_in_AM/heldout_sentiment_scores.csv', 'data/bias_in_AM/heldout_aq_scores.csv']

train_subset, train_feature_df = load_dataframes_new(train_datafile, train_featurefile)
heldout_subset, heldout_feature_df = load_dataframes_new(heldout_datafile, heldout_featurefile)

subset = pd.concat([train_subset,heldout_subset], ignore_index=True)
feature_df = pd.concat([train_feature_df,heldout_feature_df], ignore_index=True)

feature_df = get_tan_word_features(feature_df)
subset = pd.merge(subset, feature_df, on='id')

df_acl2022 = get_acl2022_feature_subset(subset)

comment_info = pd.DataFrame(columns=['id', 'perc_explicit_gender_in_comments', 'perc_explicit_gender_in_comments_m', 'perc_explicit_gender_in_comments_f', 'perc_author_gender_in_comments_m', 'perc_author_gender_in_comments_f', 'perc_author_gender_in_comments', 'avg_comment_score'])
for _, row in subset.iterrows():
    comments_df = pd.json_normalize(row['comments'])
    if comments_df.empty:
        new_entry = {'id':row['id'], 'perc_explicit_gender_in_comments':0.0, 'perc_explicit_gender_in_comments_m':0.0, 'perc_explicit_gender_in_comments_f':0.0, 'perc_author_gender_in_comments_m':0.0, 'perc_author_gender_in_comments_f':0.0, 'perc_author_gender_in_comments':0.0, 'avg_comment_score':0.0}    
    else:
        comment_count = float(comments_df.shape[0])
        explicit_gender_m = (comments_df['explicit_gender'] == 'M').sum() / comment_count
        explicit_gender_f = (comments_df['explicit_gender'] == 'F').sum() / comment_count
        perc_explicit_mentions = comments_df['explicit_gender'].isin(['M', 'F']).sum() / comment_count
        author_gender_m = (comments_df['author_gender'] == 'M').sum() / comment_count
        author_gender_f = (comments_df['author_gender'] == 'F').sum() / comment_count
        perc_author_mentions = comments_df['author_gender'].isin(['M', 'F']).sum() / comment_count
        avg_comment_score = comments_df['score'].sum() / comment_count
        new_entry = {'id':row['id'], 'perc_explicit_gender_in_comments':perc_explicit_mentions, 'perc_explicit_gender_in_comments_m':explicit_gender_m, 'perc_explicit_gender_in_comments_f':explicit_gender_f, 'perc_author_gender_in_comments_m':author_gender_m, 'perc_author_gender_in_comments_f':author_gender_f, 'perc_author_gender_in_comments':perc_author_mentions, 'avg_comment_score':avg_comment_score}
    comment_info = pd.concat([comment_info, pd.DataFrame([new_entry])], ignore_index=True)


full_tan2016_features = [
    'definite_article_perc',
    'indefinite_article_perc',
    'hu_liu_pos_nwords',
    'second_person',
    'hu_liu_neg_nwords',
    'hedge_words_perc',
    'first_person',
    'first_person_pl',
    'com_link_count',
    'example_count',
    'num_question',
    'pdf_link_count',
    'edu_link_count',
    'hu_liu_pos_perc',
    'num_quotations',
    'Valence',
    'entropy',
    'sentence_count',
    'ttr',
    'paragraph_count'
]

df_acl2022_tan2016 = pd.merge(df_acl2022, feature_df[['id']+full_tan2016_features], on='id', how='left')

df = pd.merge(df_acl2022_tan2016, subset[['id','author_gender', 'explicit_gender', 'num_comments', 'score', 'edited_binary', 'topic']], on='id', how='left')

df = pd.merge(df,comment_info, on='id', how='left')

for x in range(len(train_classifierfeaturefiles)):
    train_df_classifiers = pd.read_csv(train_classifierfeaturefiles[x], sep=',')
    heldout_df_classifiers = pd.read_csv(heldout_classifierfeaturefiles[x], sep=',')
    #heldout_df_classifiers = heldout_df_classifiers.drop(columns=['tmp_masked_selftext', 'masked_selftext'])
    df_classifiers = pd.concat([train_df_classifiers,heldout_df_classifiers], ignore_index=True)
    df = pd.merge(df, df_classifiers, on='id', how='left')

# add col for the source of the gender info (explicit vs implicit)
def determine_gender_source(row):
    if row['explicit_gender'] in [0, 1]:
        return 'explicit'
    else:
        return 'implicit'


df['gender_source'] = df.apply(determine_gender_source, axis=1)

df.to_csv('data/bias_in_AM/data_for_analysis.csv', index=False)
