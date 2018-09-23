def compute_gini(model):
	#creat a new list of agents' wealth that is a sorted version of the agents' wealt
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)





#compute gini 

x= np.sort(model.agents['copied'])
N= model.num_agents

for i in range(model.num_agents)
	gini_copies = sum(xi * (n-i) for i, xi in enumerate(x))/(N*sum(x))



