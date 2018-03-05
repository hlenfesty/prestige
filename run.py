import matplotlib.pyplot as plt
from pmodel_6 import PrestigeModel
import numpy as np

all_copies = []
all_sigmas = []
all_sigmas_local = []
all_sigmas_global = []
models = []
steps = 100 #steps

for j in range(1):
    #Run the model
    models.append(PrestigeModel(1600, 40, 40, 3))
    for i in range(steps):
        models[j].step()

    for agent in models[j].schedule.agents:
        all_copies.append(agent.copies)
        all_sigmas.append(agent.sigma_ratio)
        all_sigmas_local.append(agent.sigma_local)
        all_sigmas_global.append(agent.sigma_global)


#plot sigmas over time
plt.figure("variation over time")
model = models[0] #get the first model only
agents = model.schedule.agents # get the list of agents from the scheduler from the model
avg_sigmas_local = np.zeros(steps) # make an empty array of zeros that is the length of the generations/n of agents, call is "avg_sigma_local"

for a in agents: 
    avg_sigmas_local += np.array(a.sigma_local_history) #get each agents' local sigma history, and add up the columns to the array of zeros

avg_sigmas_local /= model.num_agents #divide the summed list of agents' histories (added to zeros array) by num agents

avg_sigmas_global = np.zeros(steps)

for a in agents:
    avg_sigmas_global += np.array(a.sigma_global_history)

avg_sigmas_global /= model.num_agents

time = range(steps)
plt.plot(time, avg_sigmas_local, color = 'orange')
plt.plot(time, avg_sigmas_global, color = 'blue')
plt.xlabel('time')
plt.ylabel('local vs global variation')
plt.title ('local vs global variation over time')

#plot the frequency of N copies

plt.figure("copies")
plt.hist(all_copies, bins=range(max(all_copies)+1))

#figure 2 plot the frequency of belief types

plt.figure("beliefs")
model = models[0]
agents = model.schedule.agents  # get a list of the agents, that's already stored in the scheduler
beliefs = [a.belief for a in agents]  # get a list of the end beliefs
plt.hist(beliefs, bins=range(max(beliefs)+1))

#plot beliefs and N copies together

model = models[0]
agents = model.schedule.agents  # get a list of the agents, that's already stored in the scheduler
agents_pos = [a.pos for a in agents]
xs = [x[0] for x in agents_pos]
ys = [y[1] for y in agents_pos]
colors = [a.belief for a in agents]  # get a list of the end beliefs
area = [c.copies for c in agents]  # area will scale to N of copies

plt.figure("heatmap")
plt.scatter(xs, ys, s=area, c=colors, alpha=.5)  # alpha is transparency

#plot histogram of sigmas

plt.figure("variance")
time = models[0]
sigmas = all_sigmas
plt.hist(all_sigmas)
plt.show()

