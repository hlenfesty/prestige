''' PLOT FIGURES '''

import matplotlib.patches as mpatches

def plot_figures(model, save_movie, sigmas, gini_time):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    
    if sigmas:
        avg_sigmas_local = np.zeros(shape=(model.steps + 1), dtype=float)
        avg_sigmas_global = np.zeros(shape=(model.steps + 1), dtype=float)
        
        for s in range(model.steps + 1):
            # calculate sigma global and sigma local
            sigma_global = np.zeros(shape=(model.num_agents), dtype=float)
            sigma_local = np.zeros(shape=(model.num_agents), dtype=float)

            for i in range(model.num_agents):
                # sigma global
                sigma_global[i] = np.mean(model.agents['belief_history'][s,] != model.agents['belief_history'][s, i])
                # sigma local
                local_beliefs = np.array([b for b, d in zip(model.agents['belief_history'][s,], model.agents['distance'][i, :]) if d <= model.neighbor_distance and d > 0])
                sigma_local[i] = np.mean(local_beliefs != model.agents['belief_history'][s, i])

            avg_sigmas_local[s] = np.mean(sigma_local)
            avg_sigmas_global[s] = np.mean(sigma_global)

        # plot sigmas over time
        plt.figure("Local vs Global Variation Over Time")
        time = range(model.steps + 1)
        plt.ylim(0, 1)
        plt.plot(time, avg_sigmas_local, color='orange')
        plt.plot(time, avg_sigmas_global, color='blue')
        plt.xlabel('Time')
        plt.ylabel('Variation of Beliefs')
        plt.title('Local vs Global Variation Over Time')
        plt.show()

        # # #plot histogram of sigmas
        plt.figure("global variance")
        plt.hist(sigma_global)
        plt.figure("local variance")
        plt.hist(sigma_local)


    if gini_time:

        gini_t = []

        for i in range(model.steps + 1):


            #compute the gini coefficient of prestige inequality of each model at each timestep

            #sort the agents' prestige from low to high
            sort = np.sort(model.agents['prestige_history'][i,])
            #find the proportion of copies that each agent has out of the total copies in the populatiom
            proportion = sort/sum(sort)
            #find the cumulative proportion of copies out of the entire population-- ea agent ascending from least to most copies
            cp = np.cumsum(proportion)
            #duplicate cp_copies to shift append a 0 to the front in order to take the avg of the upper bounds of cp_copies
            cp_2 = np.append(np.array([0]), cp[0:(model.num_agents-1)])
            #find the area under the Lorenz curve:
            #take the average of the upper bounds of agent's proportions of copies and multiply by the width of each agent's 'rectangle'
            lorenz= sum((cp + cp_2)/2 * 1/model.num_agents)
            #calculate the Gini coeffecient by dividing the area between the line of equality and Lorenz curve (A)/ A + area under the curve
            gini_t.append(1-2*(lorenz))


        #plot ginis over time
        plt.figure("Gini coefficients of prestige over time")
        time = range(model.steps + 1)
        plt.ylim(0, 1)
        plt.plot(time, gini_t, color='green')
        plt.xlabel('Time')
        plt.ylabel('Ginis')
        plt.title('Gini coefficeints over time')
        plt.show()

        prestige_t = []

        for i in range(model.steps + 1):
            
            prestige_t.append(sum(model.agents['prestige_history'][i,]))


        #plot ginis over time
        plt.figure("Sum of prestige over time")
        time = range(model.steps + 1)
        #plt.ylim(0, 1)
        plt.plot(time, prestige_t, color='green')
        plt.xlabel('Time')
        plt.ylabel('Total prestige')
        plt.title('Sum of prestige over time')
        plt.show()



    # plot the frequency of Prestige (used to be 'copied')
    all_prestige = model.agents['prestige']
    plt.figure("Prestige")
    plt.hist(all_prestige)

    #figure 2 plot the frequency of belief types
    plt.figure("Beliefs")
    beliefs = model.agents['belief']  # get a list of the end beliefs
    plt.hist(beliefs)

    #plot beliefs and Prestige (used to be 'copied') together
    xs = model.agents['x']
    ys = model.agents['y']
    colors = model.agents['belief']
    area = model.agents['prestige'] + 25 #makes the littlest dots visible


    plt.figure("Bubbleplot")
    plt.scatter(xs, ys, s=area, c=colors, cmap='cool', alpha=0.5)
    cbar = plt.colorbar()

    plt.title("Bubble plot of Beliefs and Prestige")
    plt.show()



    # try making a video

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation


    class AnimatedScatter(object):
        """An animated scatter plot using matplotlib.animations.FuncAnimation."""
        def __init__(self, model):
            self.numpoints = model.num_agents

            # Setup the figure and axes...
            self.fig, self.ax = plt.subplots()



            # Then setup FuncAnimation.
            self.ani = animation.FuncAnimation(self.fig, self.update, frames=model.steps+1, interval=100,
                                               init_func=self.setup_plot, blit=True)


        def setup_plot(self):
            """Initial drawing of the scatter plot."""
            x, y, s, c = model.agents['x'], model.agents['y'], model.agents['prestige_history'][0, :]+1, model.agents['belief_history'][0, :]
            self.scat = self.ax.scatter(x, y, c=c, cmap='cool', s=s, animated=True)
            return self.scat,


        def update(self, i):
            """Update the scatter plot."""
            s, c = ((model.agents['prestige_history'][i, :]+1)*1)**2, model.agents['belief_history'][i, :] % model.num_agents

            # Set sizes...
            self.scat._sizes = s
            # Set colors..
            self.scat.set_array(c)
            self.scat._alpha = 0.5

            return self.scat,

    a = AnimatedScatter(model)
    if save_movie:
        dpi=500
        a.ani.save('clip.mp4', writer='ffmpeg', dpi=dpi)

    plt.title("Evolution of Prestige-Based Copying")
    plt.show()


    # #plot the sig local vs. distance scatter plot
    matplotlib.style.use('ggplot')
    plt.scatter(model.sigma_local, model.avg_dist)
    plt.show()

    # plot the number of copies vs. avg distance scatter plot
    matplotlib.style.use('ggplot')
    plt.scatter(model.agents['prestige'], model.avg_dist)
    plt.show()

    # plot a scatter plot of number of times belief changed vs avg distance
    matplotlib.style.use('ggplot')
    plt.scatter(model.belief_change, model.avg_dist)
    plt.show()
