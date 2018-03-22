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
    models.append(PrestigeModel(N, width, height, donut, neighbor_distance))
    for i in range(steps):
        models[j].step()

''' PLOT FIGURES '''

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

# #plot histogram of sigmas

for model in models:
    all_sigmas_local = np.append(all_sigmas_local, model.agents['sigma_local'])
    all_sigmas_global = np.append(all_sigmas_global, model.agents['sigma_global'])

plt.figure("global variance")
plt.hist(all_sigmas_global)
plt.figure("local variance")
plt.hist(all_sigmas_local)

plt.show()
