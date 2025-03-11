import matplotlib.pyplot as plt


class DataVisualization:
    def __init__(self, n_generation):
        self.n_generation = n_generation
        self.best_fitness_list = None
        self.plot_fitness_filename = 'fitness.png'

    def plot_fitness(self, best_fitness_list):
        self.best_fitness_list = best_fitness_list
        plt.plot(self.best_fitness_list)
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness Score')
        plt.title('Best Fitness Score per Generations')
        plt.savefig(self.plot_fitness_filename)
        plt.clf()
