''' PARAMETERS '''

# how many repeat models
repeats = 1
# how many steps per model
steps = 100
# population size
N = 400
# size of the world
width = 20
height = 20
# is it a torroidal world?
donut = True
# how far away agents count as neighbors
neighbor_distance = 3
# calculate sigmas
sigmas = False
# save a movie of the simulation?
save_movie = False

''' SIMULATION '''

# imports
import matplotlib.pyplot as plt
from toms_model import PrestigeModel
import numpy as np

# create empty arrays
all_copies = np.array([], dtype=int)
all_sigmas_local = np.array([], dtype=float)
all_sigmas_global = np.array([], dtype=float)
models = []

# create and run models
for j in range(repeats):
    models.append(PrestigeModel(N, width, height, donut, neighbor_distance, sigmas))
    for i in range(steps):
        models[j].step()

''' PLOT FIGURES '''

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
area = model.agents['copied']

plt.figure("heatmap")
plt.scatter(xs, ys, s=area, c=colors, alpha=.5)  # alpha is transparency

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
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=steps+1, interval=50,
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = model.agents['x'], model.agents['y'], model.agents['copied_history'][0, :]+1, model.agents['belief_history'][0, :]
        self.scat = self.ax.scatter(x, y, c=c, s=s, animated=True)
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

plt.show()
