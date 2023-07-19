import pandas as pd
import numpy as np
import sys, json, os, ast
import seaborn as sns
import matplotlib.pyplot as plt
from functions import *


print('Loading dataframes...')
datafile = '/mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/intermediate-data/tal_elal_2016//train_period_data.annotations.jsonlist'
featurefile='/mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/intermediate-data/tal_elal_2016/feature_sets/train_op_final.csv'
dep_variable='author_gender'

subset, feature_df = load_dataframes(datafile, featurefile, dep_variable)

# get the dataframe subset that has relevant CMV features, Tan et al (2016) text features, and ACL 2022 features
df = get_relevant_dataframe(subset, feature_df, dep_variable)

threshold = .8

print('Running heirarchical feature clustering based on Spearman correlations...')
print('\tDendrogram of feature clusters output to: feature_clustering.png')
print('\tSubclusters with Spearman distance under threshold of '+str(threshold))
print()
get_feature_hierarch_correlations(df, threshold, 'feature_clustering.png')

to_drop = ['entropy', 'sentence_count', 'mtld_original_aw', 'long_words', 'flesch', 'hdd42_aw', 'PRON', 'ttr', 'paragraph_count', 'syll_per_word', 'hu_liu_neg_nwords', 'hu_liu_pos_perc', 'certainty_component', 'num_quotations', 'joy_component', 'polarity_nouns_component', 'polarity_verbs_component', 'score', 'gunningFog']
print('Features to drop due to multicolinearity: ')
print(*to_drop, sep='\n\t')
print()

print('Perform step-wise model selection using stepAIC...')
print()
tmp_df = df.drop(columns=to_drop)
features, model, result, vif, aic, bic, auc = get_stepwise_selection(tmp_df.iloc[:, 3:], dep_variable)
print(result.summary())
print(vif)
print("AIC:", aic)
print("BIC:", bic)
print("AUC:", auc)

