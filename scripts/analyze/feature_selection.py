import pandas as pd
import numpy as np
import sys, json, os, ast
import seaborn as sns
import matplotlib.pyplot as plt
from functions import *


print('Loading dataframes...')
datafile = 'data/bias_in_AM/data_for_analysis.csv'
gender_var = 'author_gender'
dep_variable = 'edited_binary' # TO-DO: do a col with diff(aq_score, aq_masked_score) and col with abs(diff(aq_score, aq_masked_score))
model_type = 'LOG'

df = pd.read_csv(datafile)


df['sentiment'] = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})

if gender_var == 'author_gender':
    df = df.drop(columns='explicit_gender')
    df['gender_source'] = df['gender_source'].map({'implicit': 0, 'explicit': 1})
else:
    df = df.drop(columns=['author_gender','gender_source'])

df.dropna(subset=[gender_var], inplace=True)

all_zeros_mask = (df == 0).all()
columns_to_drop = all_zeros_mask[all_zeros_mask].index
df = df.drop(columns=columns_to_drop)
df = df.drop(columns='masked_selftext')
if gender_var == 'author_gender':
	df = df.drop(columns = 'aq_masked_score')

threshold = .8

print('Running heirarchical feature clustering based on Spearman correlations...')
print('\tDendrogram of feature clusters output to: feature_clustering.png')
print('\tSubclusters with Spearman distance under threshold of '+str(threshold))
print()
get_feature_hierarch_correlations(df, threshold, 'feature_clustering.png')

if gender_var == 'author_gender': 
	to_drop = ['basic_ntokens', 'basic_ntypes', 'basic_ncontent_tokens', 'basic_ncontent_types', 'basic_nfunction_tokens', 'basic_nfunction_types', 'entropy', 'sentence_count', 'ttr', 'syll_per_word', 'long_words', 'flesch',  'hu_liu_neg_nwords', 'hu_liu_pos_perc', 'perc_explicit_gender_in_comments', 'toxicity_neutral', 'sentiment_positive', 'sentiment_negative']
elif gender_var == 'explicit_gender':
	to_drop = ['basic_ntokens', 'basic_ntypes', 'basic_ncontent_tokens', 'basic_ncontent_types', 'basic_nfunction_tokens', 'basic_nfunction_types', 'entropy', 'sentence_count', 'syll_per_word', 'long_words', 'flesch', 'hu_liu_pos_perc', 'score', 'perc_explicit_gender_in_comments', 'toxicity_neutral', 'sentiment_positive', 'sentiment_negative', 'aq_masked_score', 'gunningFog']


tmp_df = df.drop(columns=to_drop)



print('Features to drop due to multicolinearity: ')
print(*to_drop, sep='\n\t')
print()

print('Perform step-wise model selection using stepAIC...')
print()
tmp_df['author_gender*gender_source'] = tmp_df['author_gender'] * tmp_df['gender_source']

#get a subset where males are random selection of 400 entries
tmp_df = pd.concat([tmp_df[tmp_	df['author_gender'] == 1], tmp_df[tmp_df['author_gender'] == 0].sample(n=400, random_state=42)], ignore_index=True)

if model_type == 'OLS':
	selected_features, final_model, final_result, vif = get_stepwise_selection_OLS2(tmp_df.iloc[:, 3:], dep_variable)
	#selected_features, final_model, final_result, interactions, vif = get_stepwise_selection_OLS_with_interactions(tmp_df.iloc[:, 3:], dep_variable)
	print('Selected Features:')
	print(selected_features)
	print(final_result.summary())
	print(vif)
	print(interactions)
elif model_type == 'LOG':
	features, model, result, vif, aic, bic, auc = get_stepwise_selection(tmp_df.iloc[:, 3:], dep_variable)
	print(result.summary())
	print(vif)
	print("AIC:", aic)
	print("BIC:", bic)
	print("AUC:", auc)

