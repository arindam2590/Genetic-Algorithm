import random
import json
import numpy as np


class GeneticAlgorithm:
    def __init__(self, pop_size, mut_rate):
        with open('Utils/config.json', 'r') as file:
            self.params = json.load(file)

        self.population_size = pop_size
        self.lower_bound = self.params['CHROMOSOME_LOWER']
        self.upper_bound = self.params['CHROMOSOME_UPPER']
        self.mutation_rate = mut_rate
        self.tour_size_percent = self.params['TOURNAMENT_PERCENT']
        self.parent_percent = self.params['PARENT_PERCENT']
        self.tournament_size = None
        self.n_parents = None

    def generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            path_length = np.random.randint(self.lower_bound, self.upper_bound)
            path = ''.join(random.choices(['U', 'D', 'L', 'R'], k=path_length))
            population.append(path)
        return population

    def select_parents(self, population, fitness_scores):
        self.population_size = len(population)
        self.tournament_size = int(self.tour_size_percent * self.population_size)
        self.n_parents = int(self.parent_percent * self.tournament_size)

        print(f'Info: Size of the Tournament - {self.tournament_size}')
        parent1, parent2 = [], []
        # Select two parents from the population using tournament selection
        for i in range(self.n_parents):
            parent1.append(self.tournament_selection(population, fitness_scores))
            parent2.append(self.tournament_selection(population, fitness_scores))
        return parent1, parent2

    def tournament_selection(self, population, fitness_scores):
        # Randomly select 'tournament_size' individuals (by index) from the population.
        contestant_indices = random.sample(range(self.population_size), self.tournament_size)

        best_index = contestant_indices[0]
        for idx in contestant_indices:
            if fitness_scores[idx] < fitness_scores[best_index]:
                best_index = idx
        best_parent = population[best_index]
        return best_parent

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents to create a child
        split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2

    def mutate(self, chromosome):
        new_chromosome = ''
        for gene in chromosome:
            if random.random() < self.mutation_rate:
                new_chromosome += random.choice(['U', 'D', 'L', 'R'])
            else:
                new_chromosome += gene
        return new_chromosome
