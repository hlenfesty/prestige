#This file add calculates and appends new variables created from data at the last step of each models
#calculate post-hoc variables here after each model run
#variables include local and global variation and their ratios (sigmas), mean, median, mode of the copies distributions
#as well as Pearson skewness measures, gini coefficient of copies (in)equality, and various correlations

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#calc "sigma local" and "sigma global" at the last step of the model only
#this used to live in the model.py file and was calc'd at every step
#calculate sigma global and sigma local, measures of similarity of beliefs to other agents from the persp of each focal.
#0 is complete variation, 1 is no variation

def process_model(model):

	model.sigma_global = np.zeros(shape=(model.num_agents), dtype=float)
	model.sigma_local = np.zeros(shape=(model.num_agents), dtype=float)

	for i in range(model.num_agents):
	    # sigma global
	    model.sigma_global[i] = np.mean(model.agents['belief'] != model.agents['belief'][i])
	    # sigma local
	    local_beliefs = np.array([b for b, d in zip(model.agents['belief'], model.agents['distance'][i, :]) if d <= model.neighbor_distance and d > 0])
	    model.sigma_local[i] = np.mean(local_beliefs != model.agents['belief'][i])

	#take the average of each agents's sigma ratio to get one value that's the overall model ratio
	# ratio if = 0 no loc sigma (divided by something); if =1, equal local and global; >1, more local var than global; >0 <1 local is less than global
	model.sigma_ratio = np.mean(model.sigma_local/model.sigma_global)

	# get the avg distance of each focal agent to all other agents by to taking the mean of each column in the dist matrix
	model.avg_dist = model.agents['distance'].mean(axis=0)     

	corr_sigloc_avgdist = np.corrcoef(model.sigma_local, model.avg_dist)
	model.corr_sigloc_avgdist = corr_sigloc_avgdist [0,1]

	#correlate copies w distance
	corr_copied_avgdist= np.corrcoef(model.agents['prestige'], model.avg_dist)
	model.corr_copied_avgdist = corr_copied_avgdist[0,1]

	#Skewness:
	#find the mean, median, mode and SD of the populations' distribution of prestige

	model.mean = np.average(model.agents['prestige'])
	model.mode = stats.mode(model.agents['prestige'])
	model.mode = float(np.mean(model.mode[0]))
	model.median = np.median(model.agents['prestige'])
	model.sd = np.std(model.agents['prestige'])

	#then a measure of the population's skewness of copies using the 3 Pearson's moments of skewness:
	#Pearson 1st skewness, mode method:  x-Mo/SD
	model.pearson_1 = (model.mean-model.mode)/model.sd

	#Pearson's 2nd skewness, median method: 3(x-Md)/SD
	model.pearson_2= (3*(model.mean-model.median))/model.sd

	#Pearson's 3, mean method: (x-M/SD)^3  
	#for every agent's number of copies (X) you subtract the mean (u) divide by the sd (sigma), and then cube the result. The coefficient of skewness is the average of these numbers
	model.pearson_3 = np.zeros(shape=(model.num_agents), dtype=float)

	for i in range(model.num_agents):
		# for every agent's number of copies (X) you subtract the mean (u) divide by the sd (sigma), and then cube the result
		model.pearson_3[i] = (((model.agents['prestige'][i])-model.mean)/model.sd)**3

	#The coefficient of skewness is the average of these numbers
	model.pearson_3 = np.average(model.pearson_3)

	#find the N of belief changes in the belief history matrix
	#convert the belief_history list into a pandas dataframe
	model.belief_change= pd.DataFrame(model.agents['belief_history'])
	model.belief_change = model.belief_change.nunique(axis=0)
	model.belief_change = np.array(model.belief_change)

	corr_beliefchg_avgdist = np.corrcoef(model.belief_change, model.avg_dist)
	model.corr_beliefchg_avgdist = corr_beliefchg_avgdist [0,1]

	#compute the gini coefficient of prestige inequality
	#sort the agents' prestige from low to high
	sorted_copies= np.sort(model.agents['prestige'])
	#find the proportion of copies that each agent has out of the total copies in the populatiom
	prop_copies = sorted_copies/sum(sorted_copies)
	#find the cumulative proportion of copies out of the entire population-- ea agent ascending from least to most copies
	cp_copies = np.cumsum(prop_copies)
	#duplicate cp_copies to shift append a 0 to the front in order to take the avg of the upper bounds of cp_copies
	cp_copies2 = np.append(np.array([0]), cp_copies[0:(model.num_agents-1)])
	#find the area under the Lorenz curve:
	#take the average of the upper bounds of agent's proportions of copies and multiply by the width of each agent's 'rectangle'
	lorenz_area = sum((cp_copies + cp_copies2)/2 * 1/model.num_agents)
	#calculate the Gini coeffecient by dividing the area between the line of equality and Lorenz curve (A)/ A + area under the curve
	model.gini = 1-2*(lorenz_area)
	return model

