def compute_gini(model):
	#creat a new list of agents' wealth that is a sorted version of the agents' wealt
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)





#compute gini 




for model in models:

	#sort the agents' number of copies from low to high
	model.copied_sorted= np.sort(model.agents['copied'])
	#calculate each agents' cumulative proportion of the population at the respective index-- but this has to match up 
 	model.prop_pop = np.zeros(shape=(model.num_agents), dtype=float)
	for i in models.copied_sorted:
		model.prop_pop[i]= ((N-i)+1)/N

	print(model.prop_pop)











