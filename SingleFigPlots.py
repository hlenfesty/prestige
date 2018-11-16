from model import PrestigeModel
import pickle_in
import data
import numpy as np
import pickle as pickle
import plotting



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