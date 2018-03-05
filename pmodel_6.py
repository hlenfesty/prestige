# model.py
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import numpy as np
from statistics import variance


class PrestigeAgent(Agent):
    """An agent with fixed initial number of copies."""
    def __init__(self, unique_id, belief, model):
        super().__init__(unique_id, model)
        self.belief = belief  # every agent starts out with a unique belief
        self.belief_history = []  # every agent's belief will be recorded in an empty list
        self.sigma_local_history = []
        self.sigma_global_history = []
        self.sigma_ratio_history = []
        self.agents_copied = [] #a list of IDs of everyone you've copied
        self.copies = 1  # every agent begins with N=1 copies

    def copy(self):
        #rules for when and who copying
        neighbors_copies = [n.copies for n in self.all_agents]  # returns a list of neighbors' copies
        nc = np.array(neighbors_copies)  # turns neighbors copies into an array so that we can divide by neighbors dist
        nd = np.array(self.agents_dist)  # turn agents_dist into an array so it can be a denominator
        #neighbors_probs = (nc/(nd+1))  # add one so that we don't divide by zero
        #neighbors_probs = nc/((nd+1)**2)
        neighbors_probs = (nc*np.exp(-nd*3))
        neighbors_probs = neighbors_probs/sum(neighbors_probs)  # normalizes the probs so that they to sum to 1; gives you the weighted neighbors probs
        other_agent = self.all_agents[np.random.multinomial(1, neighbors_probs).argmax()] #argmax returns the indices of the maximum values to get the highest neighbor probs for choosing the other agent to copy
        self.agents_copied.append(other_agent.unique_id) #append the id of agent copied to a list
        if other_agent.unique_id != self.unique_id: #if you don't end up copying yourself, 
            other_agent.copies += 1  # give the agent who was copied +1 more copy in their count
        self.belief = other_agent.belief  # now adopt the belief of the agent you copied as your own belief



    def step(self):
        #The agent's step will go here.
        self.belief_history.append(self.belief)
        
        # get the local and global variances and the ratio of the two

        self.sigma_local = len([n for n in self.neighbors if n.belief != self.belief])/len(self.neighbors) #get agents who have diff beliefs than focal
        self.sigma_global = len([n for n in self.other_agents if n.belief != self.belief])/len(self.other_agents) #get agents who have diff beliefs than focal
        
        if self.sigma_global == 0:
            self.sigma_ratio = 1
        else:
            self.sigma_ratio = self.sigma_local/(self.sigma_global) # add something because I got an error for float division by zero

        self.sigma_local_history.append(self.sigma_local)
        self.sigma_global_history.append(self.sigma_global)
        self.sigma_ratio_history.append(self.sigma_ratio)

        #self.sigma_local = variance([n.belief for n in neighbors]) #calculate the local variance of your neighbors
        #self.sigma_ratio = sigma_local/sigma_global #adding sigma ratio to self gives the model a property

        self.copy()

class PrestigeModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height, neighbor_distance):
        self.num_agents = N
        self.neighbor_distance = neighbor_distance
        self.grid = ContinuousSpace(width, height, True)  # True is for torroidal
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            agent = PrestigeAgent(i, i, self)
            self.schedule.add(agent)

            # Add the agent to a random grid cell
            #x = random.randrange(self.grid.width)
            x = i % self.grid.width
            #y = random.randrange(self.grid.height)
            y = i // self.grid.height

            self.grid.place_agent(agent, (x, y))

        agents_pos = [a.pos for a in self.schedule.agents]
        for a in self.schedule.agents:
            #all the agents including themselves
            a.all_agents = self.schedule.agents

            # all the agents, but not themselves -- if the other agents position isn't your your own position
            a.other_agents = [aa for aa in self.schedule.agents if aa != a]

            # use the get_distance function to calc distance from you to all the agents (including themselves)
            a.agents_dist = [self.get_distance(a.pos, p) for p in agents_pos]

            # all the agents close enough to you to count as a neighbor (but not yourself)
            a.neighbors = [aa for aa, d in zip(a.all_agents, a.agents_dist) if aa != a and d < self.neighbor_distance] 
                #zip together the two lists to iterate through both, then create new list of neighbors who meet neighbor_distance criteria

            #print("**********")
            #print(a.unique_id)
            #print([b.unique_id for b in a.all_agents])
            #print(a.agents_dist)



        self.datacollector = DataCollector(
            agent_reporters={"Copies": lambda a: a.copies})

    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()

    def get_distance(self, pos_1, pos_2):
        """ Get the distance between two point, accounting for toroidal space.

        Args:
            pos_1, pos_2: Coordinate tuples for both points.

        """
        pos_1 = np.array(pos_1) #convert pos tuple to an array to allow subtraction
        pos_2 = np.array(pos_2)
        delta_pos = abs(pos_1 - pos_2)
        if self.grid.torus:
            if delta_pos[0] > self.grid.width/2:
                delta_pos[0] = self.grid.width - delta_pos[0]
            if delta_pos[1] > self.grid.height/2:
                delta_pos[1] = self.grid.height - delta_pos[1]           
        return np.linalg.norm(delta_pos)
        
#### new iterator for returning N agents and their coordinates on the grid
