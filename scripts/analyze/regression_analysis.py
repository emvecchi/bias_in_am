def usage():
    sys.stderr.write('\nUsage:\nregression_analysis.py <CMV_datafile> <feature_datafile> <dependent_var> <independent_variables> (-h)\n\n')
    sys.stderr.write('<CMV_datafile>\t\tFormat: .jsonlist\n')
    #sys.stderr.write('\t\t Eg: /mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/intermediate-data/tal_elal_2016/train_period_data.annotations.jsonlist\n')
    sys.stderr.write('<feature_datafile>\tFormat: .csv\n')
    #sys.stderr.write('\t\t Eg: /mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/intermediate-data/tal_elal_2016/feature_sets/train_op_final.csv\n')
    sys.stderr.write('<dependent_var>\t\tExample: author_gender, Note: binary M|F only\n')
    sys.stderr.write('<independent_variables>\tExample: score ups sentiment first_person [...]\n')
    sys.stderr.write('\t-h\t\t print this message\n\n')
    sys.exit(0)

def check_files(datafile, featurefile):
	if not os.path.exists(datafile):
		sys.stderr.write('File not found: ', datafile,'\n')
		sys.exit(0)
	if not os.path.exists(featurefile):
		sys.stderr.write('File not found: ', featurefile,'\n')
		sys.exit(0)

if __name__ == '__main__':
	import sys, json, os, ast
	import pandas as pd
	import numpy as np
	from pandas import read_csv
	import statsmodels.api as sm
	import statsmodels.formula.api as smf
	import scipy.stats as stats
	import seaborn as sns
	import matplotlib.pyplot as plt
	from functions import *

	if len(sys.argv) < 5:
		usage()
	else:
		datafile = sys.argv[1]
		featurefile = sys.argv[2]
		dep_variable=sys.argv[3]
		ind_variables=sys.argv[4:]
		check_files(datafile,featurefile)

	sys.stderr.write('\nLoading CMV data...\n')
	subset, feature_df = load_dataframes_new(datafile, featurefile)

	print('-----------------------------')
	print('::Model::')
	print('Dependent Variable: '+dep_variable)
	print('Independent Variables: '+str(ind_variables)[1:-1])
	print('')

	print('Distribution of Dependent Variable: (M:0; F:1)')
	#print(subset[dep_variable].describe(include='all'))
	print(subset[dep_variable].value_counts())
	print('')

	# get the dataframe subset that has relevant CMV features, Tan et al (2016) text features, and ACL 2022 features
	df = get_relevant_dataframe(subset, feature_df)

	#m,results,summary = get_log_regression2(df, dep_variable, ind_variables)
	#print('')
	#print('Variable Results:')
	#print(results)
	#print('')
	#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	#	print('Summary:')
	#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	#	print(summary)
	#print('')

	model, result = get_log_regression(df, dep_variable, ind_variables)
	print(result.summary())
	



	

