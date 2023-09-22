def get_regression(df, dep_variable, ind_variables):
	y = df[dep_variable]
	X = df[ind_variables]
	X = sm.add_constant(X)
	model = sm.Logit(y, X)
	result = model.fit()
	print('-----------------------------')
	print('::Model::')
	print('Dependent Variable: '+dep_variable)
	print('Independent Variables: '+str(ind_variables)[1:-1])
	print('')
	print(result.summary())
	return(result)

def get_dataframe(datafile, gender_var):
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
	# set interaction terms as columns
	df['num_comments*score']=df['score']*df['num_comments']
	return(df)

def check_file(datafile):
	if not os.path.exists(datafile):
		sys.stderr.write('File not found: ', datafile,'\n')
		sys.exit(0)

if __name__ == '__main__':
	import pandas as pd
	import numpy as np
	import sys, json, os, ast
	import seaborn as sns
	import statsmodels.api as sm

	datafile = sys.argv[1]
	gender_var = sys.argv[2]
	dep_variable = sys.argv[2]
	ind_variables = sys.argv[3:]

	check_file(datafile)
	df = get_dataframe(datafile, dep_variable)
	
	result = get_regression(df, dep_variable, ind_variables)
