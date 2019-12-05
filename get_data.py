import pickle as pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from scipy import mean

#make the Gini heatmap
def plot_heatmap(x, y, z, others):

	unpickle_data = open("data","rb")
	data_in = pickle.load(unpickle_data)
	data= pd.DataFrame(data_in)
	print(data)

	# subset data
	sub_data = data
	for o in others:
		sub_data = sub_data.loc[data[o] == others[o]]
	# #get a particular value from a column in the dataframe
	# test= data.loc[data['population'] == "random"]
	# print(test)

	# #get multiple values of a specific column in a dataframe
	# test2= data.loc[data['population'].isin(["random", "villages"])]
	# print(test2)


	#copy the parameters over from run.py

	# innovs = np.sort(list(set(data['innovate'])))
	# exps = np.sort(list(set(data['exponent'])))
	xs = np.sort(list(set(data[x])))
	ys = np.sort(list(set(data[y])))
# print(innovs)
# exps = [0,4]
# n_repeats = 4

#load the data file


#set up the x,y axes

	x_data=sub_data[x]
	y_data=sub_data[y]

	z_data=sub_data[z]

	z_data=np.array(z_data)
	z_data=np.around(z_data, decimals=5)

	#reshape the gini values into an array the wide of the exps*innovs and the height of n_repeats
	#calc mean of gini value horizontally across columns for innov*exp combo starting with the lowest exponent value

	z_data = np.zeros(shape=(len(ys), len(xs)))
	#lower = np.zeros(shape=(len(ys), len(xs)))
	#upper = np.zeros(shape=(len(ys), len(xs)))
	
	for x_val in range(len(xs)):
		for y_val in range(len(ys)):
			zs = [zz for zz, xx, yy in zip(sub_data[z], sub_data[x], sub_data[y]) if xx == xs[x_val] and yy == ys[y_val]] 
			z_data[y_val, x_val] = round(np.mean(zs), 3)

	print(z_data, "Z matrix")


	#find the 95% confidence intervals +/- the mean
	margins = np.zeros(shape=(len(ys), len(xs)))
	
	for x_val in range(len(xs)):
		for y_val in range(len(ys)):
			zs = [zz for zz, xx, yy in zip(sub_data[z], sub_data[x], sub_data[y]) if xx == xs[x_val] and yy == ys[y_val]] 
			margins[y_val, x_val] = round(np.std(zs)*1.96, 2)


	print (margins, "margins")
	




	

	# z= np.reshape(z,(n_repeats,len(exps)*len(innovs)))


	

	#lower = z_data - margin of error
	#upper = z_data + margin of error

#get the mean of the rows ACROSS ALL ROWS- FIX THIS
# z= np.mean(z, axis=1)

# print(z, "Gini means")


# z=np.around(z, decimals=3)

# print(z, "Rounded Gini means")

# z= np.reshape(z,(len(exps), len(innovs)))
# print(z)

#z= np.flip(z, axis=0)
#print(z, "z flipped")

# z= np.transpose(z)
# print(z, "z transposed")

# z= np.flip(z, axis=0)
# print(z, "z flipped")


#data.shape for dataframe dimensions
#list.data gives list of column names


	fig, ax = plt.subplots()
	im = ax.imshow(z_data, origin='lower', vmin = 0, vmax = 1)
	fig.colorbar(im, ax=ax)
	ax.set_xticks(np.arange(len(xs)))
	ax.set_yticks(np.arange(len(ys)))
	ax.set_xticklabels(xs)
	ax.set_yticklabels(ys)
	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")
	for i in range(len(xs)):
	    for j in range(len(ys)):
	        text = ax.text(i, j, str(z_data[j, i]) + "\n" + "\n",
	                       ha="center", va="center", color="black", size=10, weight="bold")

	        text = ax.text(i, j, "[" +'+/-' + str(margins[j,i])+"]",
	                       ha="center", va="center", color="black", size=6)


	ax.set_title("Ratio of Local/Global Belief Variation" + "\n" + "Random, (N=400)", size=12, weight="bold")
	plt.xlabel("Prestige Exponent")
	plt.ylabel("Probability of Innovation")
	fig.tight_layout()
	plt.show()

#for discrete confidence intervals around the avgs, add in this line after "/n" linebreak text above
#+ " [" + str(lower[j,i]) + ", " + str(upper[j,i]) + "]"


# #get a particular value from a column in the dataframe
# test= data.loc[data['population'] == "random"]
# print(test)

# #get multiple values of a specific column in a dataframe
# test2= data.loc[data['population'].isin(["random", "villages"])]
# print(test2)

#MAIN FIG IN PAPER Gini plots:  set vmin/vmax to 0/1
#plot_heatmap(x='exponent', y='innovate', z='gini', others={'population': 'random'})
#plot_heatmap(x='exponent', y='innovate', z='gini', others={'population': 'villages'})
#plot_heatmap(x='exponent', y='innovate', z='gini', others={'population': 'grid'})
#plot_heatmap(x='exponent', y='innovate', z='gini', others={'population': 'city'})

#Corr_BeliefChng_AvgDist:  set vmin/vmax to -1/1
#plot_heatmap(x='exponent', y='innovate', z='corr_beliefchg_avgdist', others={'population': 'random'})
#plot_heatmap(x='exponent', y='innovate', z='corr_beliefchg_avgdist', others={'population': 'villages'})
#plot_heatmap(x='exponent', y='innovate', z='corr_beliefchg_avgdist', others={'population': 'grid'})
#plot_heatmap(x='exponent', y='innovate', z='corr_beliefchg_avgdist', others={'population': 'city'})

#Corr_SigmaLocal_Avg Dist: set vmin/vmax to -1/1
#plot_heatmap(x='exponent', y='innovate', z='corr_sigloc_avgdist', others={'population': 'random'})
#plot_heatmap(x='exponent', y='innovate', z='corr_sigloc_avgdist', others={'population': 'villages'})
#plot_heatmap(x='exponent', y='innovate', z='corr_sigloc_avgdist', others={'population': 'grid'})
#plot_heatmap(x='exponent', y='innovate', z='corr_sigloc_avgdist', others={'population': 'city'})

#corr_copied_avgdist: set vmin/vmax to -1/1
#plot_heatmap(x='exponent', y='innovate', z='corr_copied_avgdist', others={'population': 'random'})
#plot_heatmap(x='exponent', y='innovate', z='corr_copied_avgdist', others={'population': 'villages'})
#plot_heatmap(x='exponent', y='innovate', z='corr_copied_avgdist', others={'population': 'grid'})
#plot_heatmap(x='exponent', y='innovate', z='corr_copied_avgdist', others={'population': 'city'})

# MAIN FIG IN PAPER sigma_ratio_avgdist: set vmin/vmax to -1/1
#plot_heatmap(x='exponent', y='innovate', z='sigma_ratio', others={'population': 'random'})
#plot_heatmap(x='exponent', y='innovate', z='sigma_global', others={'population': 'villages'})
#plot_heatmap(x='exponent', y='innovate', z='sigma_ratio', others={'population': 'grid'})
#plot_heatmap(x='exponent', y='innovate', z='sigma_ratio', others={'population': 'city'})

#Corr_Gini_vs_Sigma_ratio: set vmin/vmax to -1/1
#need to calc the correlation
#plot_heatmap(x='exponent', y='innovate', z='sigma_ratio', others={'population': 'random'})
#plot_heatmap(x='exponent', y='innovate', z='sigma_global', others={'population': 'villages'})
#plot_heatmap(x='exponent', y='innovate', z='sigma_ratio', others={'population': 'grid'})
#plot_heatmap(x='exponent', y='innovate', z='sigma_ratio', others={'population': 'city'})