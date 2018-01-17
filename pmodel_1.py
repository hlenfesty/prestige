# model.py
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random

class PrestigeAgent(Agent):
    """An agent with fixed initial number of copies."""
    def __init__(self, unique_id, belief, model):
        super().__init__(unique_id, model)
        self.belief= belief
        self.belief_history = []
        self.copies = 1

    def copy(self):
        #rules for when and who copying
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        copies_list = [c.copies for c in cellmates]
        print(copies_list)
        print("new line here")
        max_copies = max(copies_list)
        print("max copies", max_copies)
        max_mates = [c for c in cellmates if c.copies == max_copies]
        other_agent = random.choice(max_mates) 
        other_agent.copies +=1
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
       
        