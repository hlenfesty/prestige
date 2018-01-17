import matplotlib.pyplot as plt
from pmodel_5 import PrestigeModel

all_copies = []
models = []
for j in range(1):
    #Run the model
    models.append(PrestigeModel(400, 20, 20))
    for i in range(25):
        models[j].step()

    for agent in models[j].schedule.agents:
        all_copies.append(agent.copies)

plt.figure("copies")
plt.hist(all_copies, bins=range(max(all_copies)+1))

plt.figure("beliefs")
model = models[0]
agents = model.schedule.agents  # get a list of the agents, that's already stored in the scheduler
beliefs = [a.belief for a in agents]  # get a list of the end beliefs
plt.hist(beliefs, bins=range(max(beliefs)+1))

# figure 2

model = models[0]
agents = model.schedule.agents  # get a list of the agents, that's already stored in the scheduler
agents_pos = [a.pos for a in agents]
xs = [x[0] for x in agents_pos]
ys = [y[1] for y in agents_pos]
colors = [a.belief for a in agents]  # get a list of the end beliefs
area = [c.copies for c in agents]  # area will scale to N of copies

plt.figure("heatmap")
plt.scatter(xs, ys, s=area, c=colors, alpha=.5)  # alpha is transparency
plt.show()
