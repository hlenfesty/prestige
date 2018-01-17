# model.py
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import numpy as np


class PrestigeAgent(Agent):
    """An agent with fixed initial number of copies."""
    def __init__(self, unique_id, belief, model):
        super().__init__(unique_id, model)
        self.belief= belief #every agent starts out with a unique belief
        self.belief_history = [] #every agent's belief will be recorded in an empty list
        self.copies = 1 #every agent begins with N=1 copies

    def copy(self):
        #rules for when and who copying
        neighbors = self.model.schedule.agents
        # neighbors = self.model.grid.get_neighbors( #returns a list of neighbors
                    #     self.pos,
                    #     moore=True,
                    #     include_center=True, 
                    #     radius=20
                    # )

        neighbors_copies = [n.copies for n in neighbors] #returns a list of neighbors' copies
        neighbors_pos = [n.pos for n in neighbors]
        my_pos = self.pos
        neighbors_dist = [self.model.grid.get_distance(my_pos, p) for p in neighbors_pos]

        nc = np.array(neighbors_copies) #turns neighbors copies into an array so that we can divide by neighbors dist
        nd = np.array(neighbors_dist) #turn neighbors_dist into an array so it can be a denominator
        neighbors_probs = (nc/(nd+1)) #add one so that we don't divide by zero
        neighbors_probs = neighbors_probs/sum(neighbors_probs) #normalizes the probs to sum to 1
        other_agent = np.random.choice(neighbors, p=neighbors_probs) #weighted random choice of neighbors probs
        other_agent.copies +=1 #give the agent who was copies +1 more copy in their count
        self.belief = other_agent.belief #Here I would like to 'tag' the others' belief that was acquired by the agent, but also keep a record of the beliefs over time -- send beliefs to empty list?

    def step(self):
        #The agent's step will go here.
        self.belief_history.append(self.belief)
        self.copy()


class PrestigeModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = ContinuousSpace(width, height, True) #True is for torroidal
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = PrestigeAgent(i, i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            #x = random.randrange(self.grid.width)
            x = i % self.grid.width
            #y = random.randrange(self.grid.height)
            y = i // self.grid.height

            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"Copies": lambda a: a.copies})

    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()
        print([a.copies for a in self.schedule.agents])


#### new iterator for returning N agents and their coordinates on the grid
