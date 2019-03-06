import numpy as np
from random import shuffle
import pandas as pd


class PrestigeModel():
    """A model of prestiged biased copying."""

    def __init__(self, N, width, height, donut, neighbor_distance, innovate, population, exponent, distance_penalty, sigmas, gini_time, k): 
        # initialize the model
        self.num_agents = N
        self.width = width #THE WIDTH OF THE GRID
        self.height = height #THE HEIGHT OF THE GRID
        self.donut = donut
        self.neighbor_distance = neighbor_distance
        self.innovate = innovate
        self.population = population
        self.exponent = exponent
        self.distance_penalty = distance_penalty
        self.sigmas = sigmas
        self.gini_time= gini_time
        self.k = k
        self.steps = 0

        # if population is 'grid', make sure num_agents is a square number
        if population == "grid":
            dum = int((round(self.num_agents**0.5))**2)
            if dum != self.num_agents:
                print("Warning: grid structure requires N be a square number, but N is {}, setting it to {} instead".format(self.num_agents, dum))
                self.num_agents = dum

        # agents are represented with a dictionary of arrays
        self.agents = {
            # the id of the agent
            'id': np.empty(shape=(self.num_agents), dtype=int),
            # the agent's x coordinate
            'x': np.empty(shape=(self.num_agents), dtype=float),
            # the agent's y coordinate
            'y': np.empty(shape=(self.num_agents), dtype=float),
            # a matrix of distances from agent i to agent j
            'distance': np.empty(shape=(self.num_agents, self.num_agents)),
            # the beliefs of each agent
            'belief': np.empty(shape=(self.num_agents), dtype=int),
            # the beliefs of each agent at each generation
            'belief_history': np.empty(shape=(self.num_agents), dtype=int),
            # the number of times each agent has been copied
            'copied': np.zeros(shape=(self.num_agents), dtype=int),
            # a normalized variant of 'copied' for the purposes of plotting circles with a finite area
            # 'copied_norm': np.zeros(shape=(self.num_agents), dtype=float),
            # the number of times each agent had been copied at each generation
            'copied_history': np.zeros(shape=(self.num_agents), dtype=int),
            # agents prestige as a function of memory decay
            'prestige': np.zeros(shape=(self.num_agents), dtype=float),
            #agents' prestige history over all steps
            'prestige_history': np.zeros(shape=(self.num_agents), dtype=float)
        }



        # Create the agents
        # simply updates the id, x, y and belief values of the agent dictionary
        #to place agents evenly across the grid
        if population == "grid":

            x_step = self.width / (self.num_agents**0.5)
            y_step = self.height / (self.num_agents**0.5)
            for i in range(self.num_agents):

                self.agents['id'][i] = round(i) #WHY DO YOU HAVE TO ROUND WHAT'S ALREADY AN INTEGER?
                self.agents['x'][i] = i*x_step % self.width #TAKES THE AGENTS X COORDINATE, DIVIDES BY WIDTH OF GRID, GIVES REMAINDER
                self.agents['y'][i] = (i // int(self.num_agents**0.5))*y_step #// 'FLOOR DIVISIN': ROUNDS DOWN TO NEAREST WHOLE NUMBER

        elif population == "random":

            for i in range(self.num_agents):

                self.agents['id'][i] = round(i) #WHY DO YOU HAVE TO ROUND WHAT'S ALREADY AN INTEGER?
                self.agents['x'][i] = np.random.random()*self.width 
                self.agents['y'][i] = np.random.random()*self.height 

        elif population == "villages":

            self.agents['x'][0:int((N/4))]= np.random.normal(loc=self.width/4, scale=self.width/20, size=int(N/4))
            self.agents['y'][0:int((N/4))]= np.random.normal(loc=self.height/4, scale= self.height/20, size=int(N/4))

            self.agents['x'][int(N/4):int(N/2)]= np.random.normal(loc=self.width/4, scale=self.width/20, size=int(N/4))
            self.agents['y'][int(N/4):int(N/2)]= np.random.normal(loc=self.height*3/4, scale= self.height/20, size=int(N/4))

            self.agents['x'][int(N/2):int((N/4)*3)]= np.random.normal(loc=self.height*3/4, scale= self.height/20, size=int(N/4))
            self.agents['y'][int(N/2):int((N/4)*3)]= np.random.normal(loc=self.height/4, scale= self.height/20, size=int(N/4))

            self.agents['x'][int((N/4)*3):N]= np.random.normal(loc=self.height*3/4, scale= self.height/20, size=int(N/4))
            self.agents['y'][int((N/4)*3):N]= np.random.normal(loc=self.height*3/4, scale= self.height/20, size=int(N/4))

        elif population == "city":

            for i in range(self.num_agents):

                self.agents['x'][i] = np.random.normal(loc=self.width/2, scale = self.width/20) #TAKES THE AGENTS X COORDINATE, DIVIDES BY WIDTH OF GRID, GIVES REMAINDER
                self.agents['y'][i] = np.random.normal(loc=self.height/2, scale = self.height/20) #TAKES THE AGENTS X COORDINATE, DIVIDES BY WIDTH OF GRID, GIVES REMAINDER
                

        else:
            # print(population)
            raise ValueError()

        for i in range(self.num_agents):
            self.agents['belief'][i] = round(i)

        # add the beliefs as the first row of the belief history matrix
        #SHUFFLING IS NECESSARY BECAUSE IDS INITIALLY MATCH BELIEFS
        shuffle(self.agents['belief'])
        # print(self.agents['belief'])

        self.agents['belief_history'] = np.array(self.agents['belief'])

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
                dx = np.minimum(dx, dx2) #RETURNS THE LESSER VALUE OF THE DX VS DX2
                dy = np.minimum(dy, dy2)

            dist = (dx**2 + dy**2)**0.5
            self.agents['distance'][i, :] = dist

    def step(self):
        '''Advance the model by one step.'''
        self.steps += 1 
        # randomize the order of agents
        indexes = list(range(self.num_agents)) #TAKE THE # OF AGENTS AND MAKE THEM INTO A LIST
        shuffle(indexes) #SHUFFLE THIS LIST INTO A RANDOM ORDER

        #make a list called 'ages' that rank orders/enumerates the steps so far, w/the newest step being the largest, oldest step the smallest
        ages = np.flip(list(range(1, self.agents['copied_history'].shape[0]+1)), 0)
        #k is the decay rate
        k=self.k
        decay = np.exp(-ages*k)

        for i in indexes: #LOOP THROUGH THE LIST OF AGENTS CALLED INDEXES AND:

            if np.random.random() < self.innovate:
                #find the place of the Innovator in the belief array and give them a new belief 
                #find the highest belief so far and give a new belief 1 greater than that
                self.agents['belief'][i] = max(np.amax(self.agents['belief_history']), max(self.agents['belief'])) + 1

            else:
                #pick who to copy, 'copied' is a list of agents' copies
                
                if np.size(['copied_history', 0]) == 1 : #when copied_history is 0 rows long (first step of model)

                    probs = ((self.agents['copied']+1)**self.exponent)*np.exp(-self.agents['distance'][i, :]*self.distance_penalty)

                else: #when copied history array is more than 1 row (after first step of model)
                    #take each agents' copied_history, multiply it by our decay rule across each COLUMN (which is agents copied hist)
                    prestige_data = self.agents['copied_history']*decay[:, np.newaxis]
                    #take the sum of each agents prestige data column and 
                    self.agents['prestige'] = np.sum(prestige_data, axis = 0) + self.agents['copied']
                    
                    probs = ((self.agents['prestige']+1)**self.exponent)*np.exp(-self.agents['distance'][i, :]*self.distance_penalty)   

                # EXP'IATE COPIES MAKES AGENTS MORE INFLUENTIAL, EXP DIST ALLOWS DISTANT AGENTS TO BE COPIED AND GROUPS MORE CLUMPY
                
                #normalize the probs
                probs = probs / sum(probs) 

                #access the np.random library to use the multinomial
                other_agent = np.random.multinomial(1, probs).argmax()

                # if they aren't yourself, copy them
                if i != other_agent:
                    self.agents['copied'][other_agent] = self.agents['copied'][other_agent] + 1
                    self.agents['belief'][i] = self.agents['belief'][other_agent]

        #apppend the current beliefs, copies, and prestige to a historical list 
        self.agents['belief_history'] = np.vstack((self.agents['belief_history'], np.array(self.agents['belief'])))
        
        self.agents['copied_history'] = np.vstack((self.agents['copied_history'], self.agents['copied']))

        self.agents['prestige_history'] = np.vstack((self.agents['prestige_history'], self.agents['prestige']))

        #wipe the ['copied'] and ['prestige'] lists clean and reset to zeros

        #self.agents['prestige'] = np.zeros(shape=self.num_agents, dtype=float)
        self.agents['copied'] = np.zeros(shape=(self.num_agents), dtype=int)

        #print("Copied_History", self.agents['copied_history'])
      




        #create an innovate.history list?  keep a record of innovators somehow over time and space
