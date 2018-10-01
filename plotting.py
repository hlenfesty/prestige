''' PLOT FIGURES '''
def plot_figures(model, save_movie, sigmas):
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
        plt.figure("variation over time")
        time = range(model.steps + 1)
        plt.plot(time, avg_sigmas_local, color='orange')
        plt.plot(time, avg_sigmas_global, color='blue')
        plt.xlabel('time')
        plt.ylabel('local vs global variation')
        plt.title('local vs global variation over time')

        # # #plot histogram of sigmas
        plt.figure("global variance")
        plt.hist(sigma_global)
        plt.figure("local variance")
        plt.hist(sigma_local)

    # #plot the frequency of N copies
    all_copies = model.agents['copied']
    plt.figure("copies")
    plt.hist(all_copies)

    #figure 2 plot the frequency of belief types
    plt.figure("beliefs")
    beliefs = model.agents['belief']  # get a list of the end beliefs
    plt.hist(beliefs)

    #plot beliefs and N copies together
    xs = model.agents['x']
    ys = model.agents['y']
    colors = model.agents['belief']
    area = model.agents['copied'] + 1 #makes the littlest dots visible


    plt.figure("heatmap")
    plt.scatter(xs, ys, s=area, c=colors, cmap='cool', alpha=0.5)
    cbar = plt.colorbar()

    plt.title("Bubble plot of Beliefs and Prestige: 'Big Man Societies in Villages'")
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
            self.ani = animation.FuncAnimation(self.fig, self.update, frames=model.steps+1, interval=125,
                                               init_func=self.setup_plot, blit=True)


        def setup_plot(self):
            """Initial drawing of the scatter plot."""
            x, y, s, c = model.agents['x'], model.agents['y'], model.agents['copied_history'][0, :]+1, model.agents['belief_history'][0, :]
            self.scat = self.ax.scatter(x, y, c=c, cmap='cool', s=s, animated=True)
            return self.scat,


        def update(self, i):
            """Update the scatter plot."""
            s, c = model.agents['copied_history'][i, :]+1, model.agents['belief_history'][i, :]

            # Set sizes...
            self.scat._sizes = s
            # Set colors..
            self.scat.set_array(c)
            self.scat._alpha = 0.5

            return self.scat,

    a = AnimatedScatter(model)
    if save_movie:

        a.ani.save('clip.mp4', writer='ffmpeg')

    plt.title("Evolution of Prestige-Based Copying")
    plt.show()

    # #plot the sig local vs. distance scatter plot
    matplotlib.style.use('ggplot')
    plt.scatter(model.sigma_local, model.avg_dist)
    plt.show()

    # plot the number of copies vs. avg distance scatter plot
    matplotlib.style.use('ggplot')
    plt.scatter(model.agents['copied'], model.avg_dist)
    plt.show()

    # plot a scatter plot of number of times belief changed vs avg distance
    matplotlib.style.use('ggplot')
    plt.scatter(model.belief_change, model.avg_dist)
    plt.show()
