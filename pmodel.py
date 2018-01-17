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
        self.copies = 0

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True, #moore includes diagonals vs. von neumann which doesn't
            include_center=False)
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position) 

    def copy(self):
        #rules for when and who copying
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other_agent = random.choice(cellmates) #first, choose a random neighbor to copy to get copying off the ground.  Then, copying neighbor with the most N copies.  if all neighbors have same N copies, THEN randm choice.  Either with a certain probability?
            other_agent.copies +=1
            self.belief = other_agent.belief #Here I would like to 'tag' the others' belief that was acquired by the agent, but also keep a record of the beliefs over time -- send beliefs to empty list?
      
    def step(self):
        #The agent's step will go here.
       self.move()
       if self.copies >= 0:
           self.copy()
    
def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)
    

class PrestigeModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True) #True is for torroidal spal
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = PrestigeAgent(i, self)
            self.schedule.add(a)
            
            # Add the agent to a random grid cell
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x,y))
            
        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": lambda a: a.wealth})
            
    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()
