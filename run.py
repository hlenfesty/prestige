''' PARAMETERS '''

# how many repeat models
repeats = 1
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
populations = ["villages"]
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
print_data_dict = True


''' SIMULATION '''

# imports
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from model import PrestigeModel
import pickle_in
import data
import numpy as np
import pickle as pickle



# create empty arrays
all_copies = np.array([], dtype=int)
all_sigmas_local = np.array([], dtype=float)
all_sigmas_global = np.array([], dtype=float)
models = []

data_dict = data.new_data_dict()

# create and run models
# add 'for' layers here to vary other parameters e.g. exp 1:4
for N in Ns:
    for distance_penalty in distance_penalties:
        for exponent in exponents:
            for innovate in innovates:
                for population in populations:
                    for j in range(repeats):
                        model = PrestigeModel(N, width, height, donut, neighbor_distance, innovate, population, exponent, distance_penalty, sigmas)
                        for i in range(steps):
                            model.step()
                        model = pickle_in.process_model(model)
                        data_dict = data.save_model_results(model, data_dict)
                        if save_models:
                            models.append(model)
                        else:
                            models = [model]

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


''' PLOT FIGURES '''
if plot_figures:
    if sigmas:

        # plot sigmas over time
        plt.figure("variation over time")
        model = models[0]  # get the first model only
        agents = model.agents  # get the list of agents from the scheduler from the model
        avg_sigmas_local = agents['sigma_local_history'].mean(axis=1)
        avg_sigmas_global = agents['sigma_global_history'].mean(axis=1)

        time = range(steps + 1)
        plt.plot(time, avg_sigmas_local, color='orange')
        plt.plot(time, avg_sigmas_global, color='blue')
        plt.xlabel('time')
        plt.ylabel('local vs global variation')
        plt.title('local vs global variation over time')

        # #plot histogram of sigmas

        for model in models:
            all_sigmas_local = np.append(all_sigmas_local, model.agents['sigma_local'])
            all_sigmas_global = np.append(all_sigmas_global, model.agents['sigma_global'])

        plt.figure("global variance")
        plt.hist(all_sigmas_global)
        plt.figure("local variance")
        plt.hist(all_sigmas_local)

    # #plot the frequency of N copies

    for model in models:
        all_copies = np.append(all_copies, model.agents['copied'])

    plt.figure("copies")
    plt.hist(all_copies)

    #figure 2 plot the frequency of belief types

    plt.figure("beliefs")
    model = models[0]
    beliefs = model.agents['belief']  # get a list of the end beliefs
    plt.hist(beliefs)

    #plot beliefs and N copies together

    model = models[0]
    xs = model.agents['x']
    ys = model.agents['y']
    colors = model.agents['belief']
    area = model.agents['copied'] + 1 #makes the littlest dots visible


    plt.figure("heatmap")
    plt.scatter(xs, ys, s=area, c=colors, cmap='cool', alpha=0.5)
    cbar = plt.colorbar()

    plt.title("Bubble plot of Beliefs and Prestige: 'Big Man Societies in Villages'")
    plt.show()


    # try making a video

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation


    class AnimatedScatter(object):
        """An animated scatter plot using matplotlib.animations.FuncAnimation."""
        def __init__(self):
            self.numpoints = N

            # Setup the figure and axes...
            self.fig, self.ax = plt.subplots()
            # Then setup FuncAnimation.
            self.ani = animation.FuncAnimation(self.fig, self.update, frames=steps+1, interval=125,
                                               init_func=self.setup_plot, blit=True)


        def setup_plot(self):
            """Initial drawing of the scatter plot."""
            x, y, s, c = model.agents['x'], model.agents['y'], model.agents['copied_history'][0, :]+1, model.agents['belief_history'][0, :]
            self.scat = self.ax.scatter(x, y, c=c, cmap='cool', s=s, animated=True)
            return self.scat,


        def update(self, i):
            """Update the scatter plot."""
            s, c = model.agents['copied_history'][i, :]+1, model.agents['belief_history'][i, :]

            # Set sizes...
            self.scat._sizes = s
            # Set colors..
            self.scat.set_array(c)
            self.scat._alpha = 0.5

            return self.scat,

    a = AnimatedScatter()
    if save_movie:

        a.ani.save('clip.mp4', writer='ffmpeg')

    plt.title("Evolution of Prestige-Based Copying")
    plt.show()

