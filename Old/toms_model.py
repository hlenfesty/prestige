import numpy as np
from random import shuffle


class PrestigeModel():
    """A model of prestiged biased copying."""

    def __init__(self, N, width, height, donut, neighbor_distance, sigmas):
        # initialize the model
        self.num_agents = N
        self.width = width
        self.height = height
        self.donut = donut
        self.neighbor_distance = neighbor_distance
        self.sigmas = sigmas

        # agents are represented with a dictionary of arrays
        self.agents = {
            # the id of the agent
            'id': np.empty(shape=(self.num_agents), dtype=int),
            # the agent's x coordinate
            'x': np.empty(shape=(self.num_agents), dtype=int),
            # the agent's y coordinate
            'y': np.empty(shape=(self.num_agents), dtype=int),
            # a matrix of distances from agent i to agent j
            'distance': np.empty(shape=(self.num_agents, self.num_agents)),
            # the beliefs of each agent
            'belief': np.empty(shape=(self.num_agents), dtype=int),
            # the beliefs of each agent at each generation
            'belief_history': np.empty(shape=(self.num_agents), dtype=int),
            # the number of times each agent has been copied
            'copied': np.zeros(shape=(self.num_agents), dtype=int),
            # the number of times each agent had been coped at each generation
            'copied_history': np.zeros(shape=(self.num_agents), dtype=int),
            # the proportion of agents that have different beliefs to you
            'sigma_global': np.ones(shape=(self.num_agents), dtype=float),
            # the proportion of agents that had different beliefs to you at each generation
            'sigma_global_history': np.ones(shape=(self.num_agents), dtype=float),
            # the proportion of local agents that have different beliefs to you
            'sigma_local': np.ones(shape=(self.num_agents), dtype=float),
            # the propotion of local agents that had different beliefs to you at each generation
            'sigma_local_history': np.ones(shape=(self.num_agents), dtype=float)
        }

        # Create the agents
        # simply updates the id, x, y and belief values of the agent dictionary
        for i in range(self.num_agents):
            self.agents['id'][i] = round(i)
            self.agents['x'][i] = round((i % self.width))
            self.agents['y'][i] = round((i // self.height))
            self.agents['belief'][i] = round(i)

        # add the beliefs as the first row of the belief history matrix
        shuffle(self.agents['belief'])
        self.agents['belief_history'] = self.agents['belief']

        # create a distance matrix
        for i in range(self.num_agents):
            x = self.agents['x'][i]
            y = self.agents['y'][i]
            dx = np.absolute(x - self.agents['x'])
            dy = np.absolute(y - self.agents['y'])

            # if the world is toroidal its a little more complicated
            if self.donut:
                dx2 = self.width - dx
                dy2 = self.height - dy
                dx = np.minimum(dx, dx2)
                dy = np.minimum(dy, dy2)

            dist = (dx**2 + dy**2)**0.5
            self.agents['distance'][i, :] = dist

    def step(self):
        '''Advance the model by one step.'''

        # randomize the order of agents
        indexes = list(range(self.num_agents))
        shuffle(indexes)
        for i in indexes:

            # pick who to copy, 'copied' is a list of agents' copies
            probs = ((self.agents['copied']+1)**4)*np.exp(-self.agents['distance'][i, 
                :]*3)
            probs = probs / sum(probs)
            other_agent = list(range(self.num_agents))[np.random.multinomial(1, probs).argmax()]

            # if they aren't yourself, copy them
            if i != other_agent:
                self.agents['copied'][other_agent] = self.agents['copied'][other_agent] + 1
                self.agents['belief'][i] = self.agents['belief'][other_agent]

        self.agents['belief_history'] = np.vstack((self.agents['belief_history'], self.agents['belief']))
        self.agents['copied_history'] = np.vstack((self.agents['copied_history'], self.agents['copied']))

        if self.sigmas:
            # calculate sigma global and sigma local
            sigma_global = np.zeros(shape=(self.num_agents), dtype=float)
            sigma_local = np.zeros(shape=(self.num_agents), dtype=float)

            for i in list(range(self.num_agents)):
                # sigma global
                sigma_global[i] = np.mean(self.agents['belief'] != self.agents['belief'][i])
                # sigma local
                local_beliefs = np.array([b for b, d in zip(self.agents['belief'], self.agents['distance'][i, :]) if d <= self.neighbor_distance and d > 0])
                sigma_local[i] = np.mean(local_beliefs != self.agents['belief'][i])

            # this line takes into account that sigma global includes yourself and so removes you from the calculation
            sigma_global = (sigma_global)/(1-1/self.num_agents)

            # save it
            self.agents['sigma_global'] = sigma_global
            self.agents['sigma_local'] = sigma_local
            self.agents['sigma_global_history'] = np.vstack((self.agents['sigma_global_history'], self.agents['sigma_global']))
            self.agents['sigma_local_history'] = np.vstack((self.agents['sigma_local_history'], self.agents['sigma_local']))
