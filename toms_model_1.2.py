import numpy as np
from random import shuffle


class PrestigeModel():
    """A model of prestiged biased copying."""

    def __init__(self, N, width, height, donut, neighbor_distance, sigmas, innovate):
        # initialize the model
        self.num_agents = N
        self.width = width #THE WIDTH OF THE GRID
        self.height = height #THE HEIGHT OF THE GRID
        self.donut = donut
        self.neighbor_distance = neighbor_distance
        self.sigmas = sigmas
        self.innovate = innovate

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
            # a normalized variant of 'copied' for the purposes of plotting circles with a finite area
            'copied_norm': np.zeros(shape=(self.num_agents), dtype=float),
            # the number of times each agent had been copied at each generation
            'copied_history': np.zeros(shape=(self.num_agents), dtype=int),
            # the number of times each agent has been copied_norm at each generation
            'copied_norm_history': np.zeros(shape=(self.num_agents), dtype=int),
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
        # WHY DO WE NEED TO UPDATE IDS AND COORDINATES IF THOSE DON'T CHANGE?
        for i in range(self.num_agents):
            self.agents['id'][i] = round(i) #WHY DO YOU HAVE TO ROUND WHAT'S ALREADY AN INTEGER?
            self.agents['x'][i] = round((i % self.width)) #TAKES THE AGENTS X COORDINATE, DIVIDES BY WIDTH OF GRID, GIVES REMAINDER
            self.agents['y'][i] = round((i // self.height)) #// 'FLOOR DIVISIN': ROUNDS DOWN TO NEAREST WHOLE NUMBER
            self.agents['belief'][i] = round(i) #THIS MUST BE UPDATING ON EVERY MODEL STEP...

        # add the beliefs as the first row of the belief history matrix
        #SHUFFLING IS NECESSARY BECAUSE IDS INITIALLY MATCH BELIEFS
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
                dx = np.minimum(dx, dx2) #RETURNS THE LESSER VALUE OF THE DX VS DX2
                dy = np.minimum(dy, dy2)

            dist = (dx**2 + dy**2)**0.5
            self.agents['distance'][i, :] = dist

    def step(self):
        '''Advance the model by one step.'''

        # randomize the order of agents
        indexes = list(range(self.num_agents)) #TAKE THE # OF AGENTS AND MAKE THEM INTO A LIST
        shuffle(indexes) #SHUFFLE THIS LIST INTO A RANDOM ORDER
        for i in indexes: #LOOP THROUGH THE LIST OF AGENTS CALLED INDEXES AND:

            #pick who to copy, 'copied' is a list of agents' copies
            probs = ((self.agents['copied']+1)**4)*np.exp(-self.agents['distance'][i, 
                :]*3) #EXP'IATE COPIES MAKES AGENTS MORE INFLUENTIAL, EXP DIST ALLOWS DISTANT AGENTS TO BE COPIED AND GROUPS MORE CLUMPY
            probs = probs / sum(probs) #NORMALIZE THE PROBABILITES
            probs = ((self.agents[('copied'+1)/sum('copied'+1)]**4


            other_agent = list(range(self.num_agents))[np.random.multinomial(1, probs).argmax()]

            # if they aren't yourself, copy them
            if i != other_agent:
                self.agents['copied'][other_agent] = self.agents['copied'][other_agent] + 1
                self.agents['belief'][i] = self.agents['belief'][other_agent]

        self.agents['belief_history'] = np.vstack((self.agents['belief_history'], self.agents['belief']))
        self.agents['copied_history'] = np.vstack((self.agents['copied_history'], self.agents['copied']))
        #create an innovate.history list?  keep a record of innovators somehow over time and space


        if self.innovate:
            #randomly choose an agent to be an innovator
            I= np.random.choice(self.agents['id'])
            #find the place of the Innovator in the belief array and give them a new belief 
            self.agents['belief'[I-1]]= new_belief
            #give the new belief a number that is greater than any of the existing beliefs so far
            new_belief= max(self.agents['belief'])+1

        #self.agents[copied_b] = self.agents['copied'+1)/sum(copied+1)


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
