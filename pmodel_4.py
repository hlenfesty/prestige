# model.py
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
import random
import numpy

class PrestigeAgent(Agent):
    """An agent with fixed initial number of copies."""
    def __init__(self, unique_id, belief, model):
        super().__init__(unique_id, model)
        self.belief= belief #every agent starts out with a unique belief
        self.belief_history = [] #every agent's belief will be recorded in an empty list
        self.copies = 1 #every agent begins with N=1 copies

    def copy(self):
        #rules for when and who copying
        neighbors = self.model.grid.get_neighbors( #returns a list of neighbors
                    self.pos,
                    moore=True,
                    include_center=True, 
                    radius=1)
        neighbors_copies = [n.copies for n in neighbors] 
        #max_copies = max(copies_list) #finds the max/largest N copies the contents of cellmates_copies.  Max could exist 2 or more times, but output is only one number (aka the max)
        #max_mates = [c for c in cellmates if c.copies == max_copies] #identifies who/which agent has the max copies
        neighbors_probs = [(n / (1.0 * sum(neighbors_copies))) for n in neighbors_copies] #calcuate ratio for agents of their copies relative to all copies
        other_agent = numpy.random.choice(neighbors, p=neighbors_probs)
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
        self.grid = MultiGrid(width, height, True) #True is for torroidal spal
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = PrestigeAgent(i, i, self)
            self.schedule.add(a)
            
            # Add the agent to a random grid cell
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x,y))
            
        self.datacollector = DataCollector(
            agent_reporters={"Copies": lambda a: a.copies})
            
    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()
        print([a.copies for a in self.schedule.agents])



       