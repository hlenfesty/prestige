#This is the final data file for the Prestige Model
#Collects model parameters and variables of interest

import pickle as pickle
import numpy as np
import pandas as pd

infile = open('pickled_models', 'rb')
models = pickle.load(infile)
infile.close()

#parameters
data_dict = {
	'num_agents': np.array([], dtype = int),
    'width' : np.array([], dtype = int),
    'height' : np.array([], dtype = int),
    'donut' : np.array([], dtype = int),
    'neighbor_distance' : np.array([], dtype = int),
    'innovate' : np.array([], dtype = float),
    'population' : np.array([], dtype = str), #need to change string to int index 
    'exponent' : np.array([], dtype = float),
   	'distance_penalty' : np.array([], dtype = float),
   	'sigma_local' : np.array([], dtype = float),
   	'sigma_global' : np.array([], dtype = float),
   	'sigma_ratio' : np.array([], dtype = float),
   	'corr_sigloc_avgdist' : np.array([], dtype = float),
   	'corr_copied_avgdist' : np.array([], dtype = float),
   	'corr_beliefchg_avgdist' : np.array([], dtype = float),
   	'mean' : np.array([], dtype = float),
   	'median' : np.array([], dtype = float),
   	'mode' : np.array([], dtype = float),
   	'SD' : np.array([], dtype = float),
   	'Pearson_1' : np.array([], dtype = float),
   	'Pearson_2' : np.array([], dtype = float),
   	'Pearson_3' : np.array([], dtype = float),
   	'gini' : np.array([], dtype = float),

}

#append parameters and variables from all models to data dictionary

for model in models:
	data_dict['num_agents'] = np.append(data_dict['num_agents'], model.num_agents)
	data_dict['width'] = np.append(data_dict['width'], model.num_agents)
	data_dict['height'] = np.append(data_dict['height'], model.num_agents)
	data_dict['donut'] = np.append(data_dict['donut'], model.num_agents)
	data_dict['neighbor_distance'] = np.append(data_dict['neighbor_distance'], model.num_agents)
	data_dict['innovate'] = np.append(data_dict['innovate'], model.num_agents)
	data_dict['population'] = np.append(data_dict['population'], model.num_agents)
	data_dict['exponent'] = np.append(data_dict['exponent'], model.num_agents)
	data_dict['distance_penalty'] = np.append(data_dict['distance_penalty'], model.num_agents)
	data_dict['sigma_local'] = np.append(data_dict['sigma_local'], model.num_agents)
	data_dict['sigma_global'] = np.append(data_dict['sigma_global'], model.num_agents)
	data_dict['sigma_ratio'] = np.append(data_dict['sigma_ratio'], model.num_agents)
	data_dict['corr_sigloc_avgdist'] = np.append(data_dict['corr_sigloc_avgdist'], model.num_agents)
	data_dict['corr_copied_avgdist'] = np.append(data_dict['corr_copied_avgdist'], model.num_agents)
	data_dict['corr_beliefchg_avgdist'] = np.append(data_dict['corr_beliefchg_avgdist'], model.num_agents)
	data_dict['mean'] = np.append(data_dict['mean'], model.num_agents)
	data_dict['median'] = np.append(data_dict['median'], model.num_agents)
	data_dict['mode'] = np.append(data_dict['mode'], model.num_agents)
	data_dict['SD'] = np.append(data_dict['SD'], model.num_agents)
	data_dict['Pearson_1'] = np.append(data_dict['Pearson_1'], model.num_agents)
	data_dict['Pearson_2'] = np.append(data_dict['Pearson_2'], model.num_agents)
	data_dict['Pearson_3'] = np.append(data_dict['Pearson_3'], model.num_agents)
	data_dict['gini'] = np.append(data_dict['gini'], model.num_agents)


filename = 'data'
outfile = open('data', 'wb') #write bytes
pickle.dump(data_dict, outfile)
outfile.close()

#To convert the data dictionary into pandas dataframe object later:

# dframe = pd.DataFrame(data_dict)
# dframe.index()
# dframe.columns()
# print(dframe)








