''' PARAMETERS '''

# how many repeat models
repeats = 2
# how many steps per model
steps = 100
# population size
Ns = [200]
# size of the world
width =20
height = 20
# is it a torroidal world?
donut = True
# how far away agents count as neighbors
neighbor_distance = 3 
# Probability of innovation
innovates = [0.001]
# population type: [random, grid, villages, city]
populations = ["villages", "random"]
# exponential increase of prestige
exponents = [1]
# penalize the distance of the agents
distance_penalties = [4]
# calculate sigmas
sigmas = False
# save a movie of the simulation?
save_movie = False
# save the model objects?
save_models = False
# plot figures? (only applies to final model run)
plot_figures = True
# print data dict at end?
print_data_dict = False


''' SIMULATION '''

# imports
#import matplotlib.cm as cm
from model import PrestigeModel
import pickle_in
import data
import numpy as np
import pickle as pickle
import plotting


# create empty arrays
all_copies = np.array([], dtype=int)
all_sigmas_local = np.array([], dtype=float)
all_sigmas_global = np.array([], dtype=float)
models = []

data_dict = data.new_data_dict()

# create and run models
# add 'for' layers here to vary other parameters e.g. exp 1:4
num_sims = len(Ns)*len(distance_penalties)*len(exponents)*len(innovates)*len(populations)*repeats
current_sim = 1
for N in Ns:
    for distance_penalty in distance_penalties:
        for exponent in exponents:
            for innovate in innovates:
                for population in populations:
                    for j in range(repeats):
                        print("Running model " + str(current_sim) + " of " + str(num_sims))
                        current_sim += 1
                        model = PrestigeModel(N, width, height, donut, neighbor_distance, innovate, population, exponent, distance_penalty, sigmas)
                        for i in range(steps):
                            model.step()
                        model = pickle_in.process_model(model)
                        data_dict = data.save_model_results(model, data_dict)
                        if save_models:
                            models.append(model)
                        else:
                            models = [model]
print("Done!")

filename = 'data'
outfile = open('data', 'wb')
pickle.dump(data_dict, outfile)
outfile.close()
if print_data_dict:
    print(data_dict)

if save_models:
    #save the output of the model by 'pickling' the list of model objects
    filename = 'pickled_models'
    outfile = open('pickled_models', 'wb') #write bytes
    pickle.dump(models, outfile)
    outfile.close()

if plot_figures:
    plotting.plot_figures(model, save_movie, sigmas)
