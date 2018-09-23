#to open or de-pickle the file:
import pickle as pickle
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


infile = open('pickled_models', 'rb')
models = pickle.load(infile)
infile.close()

# print(models)
# print(models[0].donut)
# print(models[0].num_agents)
# print(models[0].agents)

#ex:  how to add something to the 'pickle jar':
#models[0].house = "house"
#print(models[0].house)


#calc "sigma local" and "sigma global" at the last step of the model only
#this used to live in the model.py file and calc'd at every step;
#calculate sigma global and sigma local, measures of similarity of beliefs to other agents from the persp of each focal.
#0 is complete variation, 1 is no variation

for model in models:
	model.sigma_global = np.zeros(shape=(model.num_agents), dtype=float)
	model.sigma_local = np.zeros(shape=(model.num_agents), dtype=float)

	for i in range(model.num_agents):
	    # sigma global
	    model.sigma_global[i] = np.mean(model.agents['belief'] != model.agents['belief'][i])
	    # sigma local
	    local_beliefs = np.array([b for b, d in zip(model.agents['belief'], model.agents['distance'][i, :]) if d <= model.neighbor_distance and d > 0])
	    model.sigma_local[i] = np.mean(local_beliefs != model.agents['belief'][i])

	#take the average of each agents's sigma ratio to get one value that's the overall model ratio
	print(model.sigma_local)
	print("break")

	# ratio = 0 no loc sigma (divided by something), =1, equal local and global, >1, more local var than global, >0 <1 local is less than global
	model.sigma_ratio = np.mean(model.sigma_local/model.sigma_global)
	print(model.sigma_ratio)
	# print(model.sigma_global)
	# print(model.sigma_local)


	print(model.agents['distance'])

	# get the avg distance of each focal agent to all other agents by to taking the mean of each column in the dist matrix
	model.avg_dist = model.agents['distance'].mean(axis=0)     
	print(model.avg_dist)

	corr1= np.corrcoef(model.sigma_local, model.avg_dist)
	print("correlation of sig local and dist")
	print(corr1)

	#plot the sig local vs. distance correlation
	matplotlib.style.use('ggplot')
	plt.scatter(model.sigma_local, model.avg_dist)
	plt.show()

	#correlate copies w distance

	corr2= np.corrcoef(model.agents['copied'], model.avg_dist)
	print ("correlation of copies and dist")
	print(corr2)

	matplotlib.style.use('ggplot')
	plt.scatter(model.agents['copied'], model.avg_dist)
	plt.show()


	#Skewness:
	#find the mean, median, mode and SD of the populations' copies distribution

	print("Copies", model.agents['copied'])

	model.mean = np.average(model.agents['copied'])
	print("Mean", model.mean,)

	model.mode = stats.mode(model.agents['copied'])
	print("Mode", model.mode)

	model.median = np.median(model.agents['copied'])
	print("Median", model.median)

	model.sd = np.std(model.agents['copied'])
	print("SD", model.sd)


	#then a measure of the population's skewness of copies using the 3 Pearson's moments of skewness:

	#Pearson 1st skewness, mode method:  x̅-Mo/SD
	model.pearson_1 = (model.mean-model.mode)/model.sd
	print("Pearson 1", model.pearson_1)

	#Pearson's 2nd skewness, median method:  3(x̅-Md)/SD
	model.pearson_2= (3*(model.mean-model.median))/model.sd
	print("Pearson 2", model.pearson_2)

	#Pearson's 3, mean method: (x̅-M/SD)^3  
	#for every agent's number of copies (X) you subtract the mean (u) divide by the sd (sigma), and then cube the result. The coefficient of skewness is the average of these numbers
	model.pearson_3 = np.zeros(shape=(model.num_agents), dtype=float)

	for i in range(model.num_agents):
		# for every agent's number of copies (X) you subtract the mean (u) divide by the sd (sigma), and then cube the result
		model.pearson_3[i] = (((model.agents['copied'][i])-model.mean)/model.sd)**3

	#The coefficient of skewness is the average of these numbers
	model.pearson_3 = np.average(model.pearson_3)
	print("Pearson 3", model.pearson_3)


	#find the N of belief changes in the belief history matrix
	#convert the belief_history list into a pandas dataframe
	model.belief_change= pd.DataFrame(model.agents['belief_history'])
	print(model.belief_change)

	model.belief_change = model.belief_change.nunique(axis=0)
	print(model.belief_change)

	model.belief_change = np.array(model.belief_change)
	print(model.belief_change)

	corr3= np.corrcoef(model.belief_change, model.avg_dist)
	print ("correlation of belief changes and dist")
	print(corr2)

	matplotlib.style.use('ggplot')
	plt.scatter(model.belief_change, model.avg_dist)
	plt.show()

	#compute the gini coefficient of prestige inequality

	print(model.agents['copied'])

	x= np.sort(model.agents['copied'])




	#then pickle close?
	#the re-open pickle file to check?